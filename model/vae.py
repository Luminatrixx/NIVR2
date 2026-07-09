"""
Causal video VAE for SeedVR2, ported to native ComfyUI module conventions.

Ported from ``seedvr2_videoupscaler/src/models/video_vae_v3/modules/attn_video_vae.py``
(class ``VideoAutoencoderKLWrapper``) and ``.../modules/causal_inflation_lib.py``
(class ``InflatedCausalConv3d``). The encoder/decoder block structure, the
causal-conv left-context cache, the temporal chunked ("slicing") streaming
encode/decode, and the per-frame spatial mid-block attention are ported
faithfully. Bespoke memory-management infrastructure (OOM-driven conv/norm
splitting, cuDNN workaround, tensor offloading, debug overlays) is dropped in
favor of ComfyUI-native tiling (this class is wired into
``comfy.sd.VAE.first_stage_model`` -- see package ``__init__.py`` -- so the
generic ``VAE.encode_tiled_3d``/``decode_tiled_3d``, built on
``comfy.utils.tiled_scale_multidim``, drive tiling) and the cuDNN fix already
built into ``comfy.ops.*.Conv3d``.

Architecture (from ``s8_c16_t4_inflation_sd3.yaml`` + ``configs_3b/main.yaml``):
    in_channels=3, out_channels=3, latent_channels=16
    block_out_channels=(128, 256, 512, 512), layers_per_block=2 (decoder: 3)
    norm_num_groups=32, act_fn=silu, temporal_scale_num=2
    spatial_downsample_factor=8 (3 of 4 stages downsample space x2)
    temporal_downsample_factor=4 (last 2 of the downsampling stages downsample
    time x2, causally)
    use_quant_conv=False, use_post_quant_conv=False (encoder's conv_out
    directly emits 32 channels = 16 mean + 16 logvar)
    scaling_factor=0.9152, shifting_factor=0.0 (no shift key in main.yaml)

The `time_receptive_field` switch in the source (`"half"` vs `"full"`) is
never overridden anywhere in the config chain that builds this checkpoint's
VAE (`VideoAutoencoderKL.__init__` default is `"full"`), so both resnet
convs always use a full (3,3,3) causal kernel here; the `"half"` (1,3,3)
branch is dead code for this checkpoint and was not ported.

Calling convention:
    - Input pixels to ``encode`` must already be normalized to [-1, 1]
      (the source wrapper applies ``Normalize(0.5, 0.5)`` *before* calling
      the VAE; that normalization is the caller's responsibility here, not
      this module's).
    - Input pixels are ``(B, 3, T, H, W)`` with any ``T >= 1``; ``encode``
      transparently pads to the nearest ``4n+1`` (repeating the last frame)
      and trims the corresponding output frames, since the causal temporal
      downsample requires it (mirrors the source's ``preprocess()``
      contract) but callers -- including ComfyUI's native tiled VAE encode,
      which hands arbitrary-length temporal tiles -- shouldn't have to know
      that.
    - ``encode`` returns the scaled 16-channel latent directly (deterministic
      / distribution mode, matching the source wrapper's
      ``posterior.mode()``): ``latent = (mean - shift) * scale``.
    - ``decode`` inverts the scale/shift (``z = latent / scale + shift``)
      and returns pixels in [-1, 1]; un-normalizing back to [0, 1] or
      [0, 255] is the caller's responsibility.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

import comfy.ops
from comfy.ldm.modules.attention import optimized_attention


class MemoryState(Enum):
    """
    DISABLED:     One-shot call, no causal cache kept (used when the whole
                  clip fits in a single chunk - mathematically identical to
                  INITIALIZING, just skips retaining `.memory` afterwards).
    INITIALIZING: First chunk of a streamed clip - primes the causal cache.
    ACTIVE:       Later chunk of a streamed clip - conv left-context comes
                  from the cache left behind by the previous chunk instead of
                  frame-0 replication, so chunked and one-shot processing of
                  the same clip give bit-identical results.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2


def _to_3tuple(value):
    if isinstance(value, (tuple, list)):
        assert len(value) == 3
        return tuple(value)
    return (value, value, value)


@dataclass
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    resnet_eps: float = 1e-6
    temporal_scale_num: int = 2
    slicing_sample_min_size: int = 4
    spatial_downsample_factor: int = 8
    temporal_downsample_factor: int = 4
    scaling_factor: float = 0.9152
    shifting_factor: float = 0.0
    mid_block_add_attention: bool = True
    double_z: bool = True


def group_norm_per_frame(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    GroupNorm statistics are computed per-frame (over C,H,W only), never
    across time - matches source `causal_norm_wrapper`'s
    "b c t h w -> (b t) c h w" reshape. A plain 5D GroupNorm call would
    instead normalize jointly over (T,H,W), which is a different (and wrong)
    set of statistics.
    """
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = norm(x)
    x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return x


def remove_head(x: torch.Tensor, times: int = 1) -> torch.Tensor:
    if times == 0:
        return x
    return torch.cat([x[:, :, :1], x[:, :, times + 1:]], dim=2)


def pixel_shuffle_3d(x: torch.Tensor, spatial_ratio: int, temporal_ratio: int, out_channels: int) -> torch.Tensor:
    """
    Inverse of packing `out_channels * spatial_ratio**2 * temporal_ratio`
    channels as `(x y z c)` (x outermost, c innermost); unpacks into a
    spatial/temporal pixel-shuffle upsample. Matches source's einops
    `"b (x y z c) f h w -> b c (f z) (h x) (w y)"`.
    """
    b, _, f, h, w = x.shape
    sx = sy = spatial_ratio
    sz = temporal_ratio
    x = x.view(b, sx, sy, sz, out_channels, f, h, w)
    x = x.permute(0, 4, 5, 3, 6, 1, 7, 2).contiguous()
    x = x.reshape(b, out_channels, f * sz, h * sx, w * sy)
    return x


class InflatedCausalConv3d(nn.Module):
    """
    Causal Conv3d: temporal padding is applied only on the left (past) side
    - either by replicating frame 0 (start of a clip / one-shot call) or by
    prepending the last `kernel_t - stride_t` frames cached from the
    previous chunk (streamed calls). Never pads on the right/future side,
    which is what makes the conv causal. Spatial padding is a normal
    symmetric conv padding, unaffected by any of this.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        kernel_size = _to_3tuple(kernel_size)
        stride = _to_3tuple(stride)
        padding = _to_3tuple(padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.temporal_padding = padding[0]
        self.memory: Optional[torch.Tensor] = None

        self.conv = operations.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, padding[1], padding[2]),
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def reset_memory(self):
        self.memory = None

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        if memory_state != MemoryState.ACTIVE:
            self.memory = None

        # Cache size for a stride-s, kernel-k causal conv is (k - s) frames;
        # written here as (stride - kernel) so a negative value indexes the
        # last |stride-kernel| frames of the (already left-padded) input.
        mem_size = self.stride[0] - self.kernel_size[0]

        if self.memory is not None and memory_state == MemoryState.ACTIVE:
            x = torch.cat([self.memory.to(dtype=x.dtype, device=x.device), x], dim=2)
        elif self.temporal_padding > 0:
            first_frame = x[:, :, :1].repeat(1, 1, self.temporal_padding * 2, 1, 1)
            x = torch.cat([first_frame, x], dim=2)

        if mem_size != 0 and memory_state != MemoryState.DISABLED:
            self.memory = x[:, :, mem_size:].detach()

        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int,
        eps: float,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm1 = operations.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True, dtype=dtype, device=device)
        self.conv1 = InflatedCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

        self.norm2 = operations.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=eps, affine=True, dtype=dtype, device=device)
        self.conv2 = InflatedCausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = in_channels != out_channels
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedCausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device, operations=operations)
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        residual = x

        h = group_norm_per_frame(self.norm1, x)
        h = self.nonlinearity(h)
        h = self.conv1(h, memory_state)

        h = group_norm_per_frame(self.norm2, h)
        h = self.nonlinearity(h)
        h = self.conv2(h, memory_state)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual, memory_state)

        return residual + h


class Downsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        temporal_down: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        temporal_kernel = 3 if temporal_down else 1
        temporal_stride = 2 if temporal_down else 1
        temporal_pad = 1 if temporal_down else 0

        self.conv = InflatedCausalConv3d(
            channels,
            out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            stride=(temporal_stride, 2, 2),
            padding=(temporal_pad, 0, 0),
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        # Asymmetric (right-only) spatial pad so a stride-2 kernel-3 conv
        # with padding=0 produces exactly H/2 x W/2 (the diffusers
        # Downsample2D "even size" trick). Temporal causal padding is
        # handled inside InflatedCausalConv3d.
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x, memory_state)


class Upsample3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        temporal_up: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.channels = channels
        self.spatial_ratio = 2
        self.temporal_up = temporal_up
        self.temporal_ratio = 2 if temporal_up else 1

        upscale_ratio = (self.spatial_ratio ** 2) * self.temporal_ratio
        # 1x1x1 conv whose output channels get pixel-shuffled into
        # spatial/temporal upsampling (MAGViT-v2 style), instead of a
        # transposed conv or nearest-neighbour interpolation.
        self.upscale_conv = operations.Conv3d(channels, channels * upscale_ratio, kernel_size=1, dtype=dtype, device=device)
        self.conv = InflatedCausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        x = self.upscale_conv(x)
        x = pixel_shuffle_3d(x, self.spatial_ratio, self.temporal_ratio, self.channels)
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            # The temporal pixel-shuffle duplicates the causal anchor frame
            # (frame 0) into two; drop the duplicate so a streamed clip keeps
            # exactly one frame 0, matching a one-shot pass over the full clip.
            x = remove_head(x)
        return self.conv(x, memory_state)


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        norm_num_groups: int,
        eps: float,
        add_downsample: bool,
        temporal_down: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock3D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    norm_num_groups,
                    eps,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsampler = Downsample3D(out_channels, out_channels, temporal_down, dtype=dtype, device=device, operations=operations)
        else:
            self.downsampler = None

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory_state)
        if self.downsampler is not None:
            x = self.downsampler(x, memory_state)
        return x


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        norm_num_groups: int,
        eps: float,
        add_upsample: bool,
        temporal_up: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            resnets.append(
                ResnetBlock3D(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    norm_num_groups,
                    eps,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = Upsample3D(out_channels, out_channels, temporal_up, dtype=dtype, device=device, operations=operations)
        else:
            self.upsampler = None

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, memory_state)
        if self.upsampler is not None:
            x = self.upsampler(x, memory_state)
        return x


class SpatialSelfAttention(nn.Module):
    """
    Single-head full self-attention over the H*W tokens of one frame
    (in_channels == dim_head, heads=1, matching the source's
    `Attention(in_channels, heads=in_channels//attention_head_dim=1,
    dim_head=attention_head_dim=in_channels)`), with GroupNorm pre-norm and
    a residual connection.
    """

    def __init__(
        self,
        channels: int,
        norm_num_groups: int,
        eps: float,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.heads = 1
        self.norm = operations.GroupNorm(num_groups=norm_num_groups, num_channels=channels, eps=eps, affine=True, dtype=dtype, device=device)
        self.to_q = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.to_k = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.to_v = operations.Linear(channels, channels, dtype=dtype, device=device)
        self.to_out = operations.Linear(channels, channels, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) - a single video frame.
        residual = x
        b, c, h, w = x.shape
        x = self.norm(x)
        x = x.reshape(b, c, h * w).transpose(1, 2)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out = optimized_attention(q, k, v, heads=self.heads)
        out = self.to_out(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return residual + out


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        norm_num_groups: int,
        eps: float,
        add_attention: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.resnet1 = ResnetBlock3D(channels, channels, norm_num_groups, eps, dtype=dtype, device=device, operations=operations)
        self.attention = (
            SpatialSelfAttention(channels, norm_num_groups, eps, dtype=dtype, device=device, operations=operations)
            if add_attention
            else None
        )
        self.resnet2 = ResnetBlock3D(channels, channels, norm_num_groups, eps, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        x = self.resnet1(x, memory_state)
        if self.attention is not None:
            # Attention is spatial-only: each frame attends only to itself,
            # never to neighbouring frames ("b c f h w -> (b f) c h w"), so
            # it stays causal (and chunk-invariant) for free - no left
            # context / memory cache is needed here.
            b, c, f, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
            x = self.attention(x)
            x = x.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        x = self.resnet2(x, memory_state)
        return x


class Encoder3D(nn.Module):
    def __init__(self, config: VAEConfig, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        boc = config.block_out_channels
        n_stages = len(boc)

        self.conv_in = InflatedCausalConv3d(config.in_channels, boc[0], kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

        down_blocks = []
        out_ch = boc[0]
        for i in range(n_stages):
            in_ch = out_ch
            out_ch = boc[i]
            is_final = i == n_stages - 1
            is_temporal_down = i >= n_stages - config.temporal_scale_num - 1
            down_blocks.append(
                DownEncoderBlock3D(
                    in_ch,
                    out_ch,
                    config.layers_per_block,
                    config.norm_num_groups,
                    config.resnet_eps,
                    add_downsample=not is_final,
                    temporal_down=is_temporal_down,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = UNetMidBlock3D(boc[-1], config.norm_num_groups, config.resnet_eps, config.mid_block_add_attention, dtype=dtype, device=device, operations=operations)

        self.conv_norm_out = operations.GroupNorm(num_groups=config.norm_num_groups, num_channels=boc[-1], eps=config.resnet_eps, affine=True, dtype=dtype, device=device)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * config.latent_channels if config.double_z else config.latent_channels
        self.conv_out = InflatedCausalConv3d(boc[-1], conv_out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        x = self.conv_in(x, memory_state)
        for block in self.down_blocks:
            x = block(x, memory_state)
        x = self.mid_block(x, memory_state)
        x = group_norm_per_frame(self.conv_norm_out, x)
        x = self.conv_act(x)
        x = self.conv_out(x, memory_state)
        return x


class Decoder3D(nn.Module):
    def __init__(self, config: VAEConfig, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        boc = config.block_out_channels
        n_stages = len(boc)
        reversed_boc = list(reversed(boc))

        self.conv_in = InflatedCausalConv3d(config.latent_channels, boc[-1], kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

        self.mid_block = UNetMidBlock3D(boc[-1], config.norm_num_groups, config.resnet_eps, config.mid_block_add_attention, dtype=dtype, device=device, operations=operations)

        up_blocks = []
        out_ch = reversed_boc[0]
        for i in range(n_stages):
            in_ch = out_ch
            out_ch = reversed_boc[i]
            is_final = i == n_stages - 1
            is_temporal_up = i < config.temporal_scale_num
            up_blocks.append(
                UpDecoderBlock3D(
                    in_ch,
                    out_ch,
                    config.layers_per_block + 1,
                    config.norm_num_groups,
                    config.resnet_eps,
                    add_upsample=not is_final,
                    temporal_up=is_temporal_up,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.conv_norm_out = operations.GroupNorm(num_groups=config.norm_num_groups, num_channels=boc[0], eps=config.resnet_eps, affine=True, dtype=dtype, device=device)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedCausalConv3d(boc[0], config.out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        x = self.conv_in(x, memory_state)
        x = self.mid_block(x, memory_state)
        for block in self.up_blocks:
            x = block(x, memory_state)
        x = group_norm_per_frame(self.conv_norm_out, x)
        x = self.conv_act(x)
        x = self.conv_out(x, memory_state)
        return x


class SeedVR2VAE(nn.Module):
    """
    Native-ComfyUI port of SeedVR2's causal video VAE
    (``VideoAutoencoderKLWrapper``). See module docstring for the exact
    scale/shift formula, the [-1, 1] / 4n+1 input contract, and what was
    intentionally dropped vs. ported from the source.
    """

    def __init__(self, dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init

        self.config = VAEConfig()
        self.encoder = Encoder3D(self.config, dtype=dtype, device=device, operations=operations)
        self.decoder = Decoder3D(self.config, dtype=dtype, device=device, operations=operations)

    def _reset_causal_memory(self):
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.reset_memory()

    def _chunked_run(self, x: torch.Tensor, run, chunk: int) -> torch.Tensor:
        """
        Shared causal-streaming driver for encode/decode: frame 0 is the
        causal anchor kept in every chunk's left context; the remaining
        frames are split into `chunk`-sized pieces and run through `run`
        (encoder or decoder forward) while `InflatedCausalConv3d` carries
        left-context state (`.memory`) across chunk boundaries. This is a
        pure memory/streaming optimization: because the conv only ever looks
        backward, the concatenated result is bit-identical to a single
        one-shot call over the full clip.
        """
        t = x.shape[2]
        self._reset_causal_memory()
        try:
            if t - 1 > chunk:
                rest = list(x[:, :, 1:].split(chunk, dim=2))
                outs = [run(torch.cat([x[:, :, :1], rest[0]], dim=2), MemoryState.INITIALIZING)]
                for piece in rest[1:]:
                    outs.append(run(piece, MemoryState.ACTIVE))
                out = torch.cat(outs, dim=2)
            else:
                out = run(x, MemoryState.DISABLED)
        finally:
            self._reset_causal_memory()
        return out

    def encode(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        pixel: (B, 3, T, H, W) in [-1, 1], any T >= 1. Internally padded to
        the nearest 4n+1 (by repeating the last frame) since the causal
        temporal downsample requires it; this is transparent to the caller
        -- unlike a hard T==4n+1 assertion, this must tolerate arbitrary
        T because ComfyUI's native tiled VAE encode (once this class is
        wired into ``comfy.sd.VAE`` -- see package ``__init__.py``) calls
        ``encode`` per temporal tile via ``comfy.utils.tiled_scale_multidim``,
        and a tile length isn't guaranteed to already be 4n+1.
        returns: (B, 16, T', H//8, W//8) scaled latent, T' = (T-1)//4 + 1
        (computed from the ORIGINAL, unpadded T).
        """
        assert pixel.ndim == 5, f"expected (B,C,T,H,W), got shape {tuple(pixel.shape)}"
        t = pixel.shape[2]
        if (t - 1) % 4 != 0:
            target = ((t - 1) // 4 + 1) * 4 + 1
            pad = target - t
            pixel = torch.cat([pixel, pixel[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        out_t = (t - 1) // 4 + 1

        raw = self._chunked_run(
            pixel,
            lambda chunk, state: self.encoder(chunk, state),
            self.config.slicing_sample_min_size,
        )
        mean, _logvar = raw.chunk(2, dim=1)
        mean = mean[:, :, :out_t]
        return (mean - self.config.shifting_factor) * self.config.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, 16, T', H', W') scaled latent (as produced by `encode`).
        returns: (B, 3, T, H, W) pixels in [-1, 1], T = (T'-1)*4 + 1.
        """
        assert latent.ndim == 5, f"expected (B,C,T,H,W), got shape {tuple(latent.shape)}"
        z = latent / self.config.scaling_factor + self.config.shifting_factor

        chunk = max(1, self.config.slicing_sample_min_size // (2 ** self.config.temporal_scale_num))
        return self._chunked_run(
            z,
            lambda piece, state: self.decoder(piece, state),
            chunk,
        )

    # Tiled encode/decode intentionally not implemented here: once this class
    # is wired in as ``comfy.sd.VAE.first_stage_model`` (see package
    # ``__init__.py``), the generic ``VAE.encode_tiled_3d``/``decode_tiled_3d``
    # (``comfy/sd.py``, built on ``comfy.utils.tiled_scale_multidim``) drive
    # tiling by calling this class's plain ``encode``/``decode`` per tile --
    # duplicating that logic here would just be two implementations to keep
    # in sync.


# Backward-compatible alias for any existing imports.
NIVR2VAE = SeedVR2VAE
