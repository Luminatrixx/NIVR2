"""
NaDiT (Native-resolution Diffusion Transformer): the top-level SeedVR2 3B/7B
video diffusion transformer, ported from dit_3b/nadit.py and dit_7b/nadit.py.

This is a single-video (batch=1), single-GPU (no sequence-parallel) port:
all of the original's multi-shape batch packing (``na.flatten/unflatten``),
sequence-parallel all-to-all/slice/gather ops, and varlen flash-attention
backend dispatch have been dropped -- those are no-ops at world_size=1 in the
source anyway. The per-element math (patchify, AdaLN-single modulation,
window partitioning, RoPE, attention, MLP) is ported exactly; see
``attention.py``, ``rope.py``, ``window.py`` and ``common.py`` for the
individual pieces.

forward() contract
-------------------
``NaDiT.forward`` is the ComfyUI diffusion-model calling convention entry
point (``forward(x, timestep, context=None, control=None,
transformer_options=None, **kwargs)``, batch size 1, channel-first),
invoked by ``comfy.model_base.BaseModel._apply_model`` -- see
``model/base_model.py`` for how ``x``'s 33 channels (16 noisy + 16 SR
condition + 1 task mask) get concatenated via native concat conditioning
before this is called. The original single-video, channel-last math (patchify,
AdaLN-single modulation, window partitioning, RoPE, attention, MLP) lives in
``_forward_single(vid, txt, timestep)`` and is unchanged; ``forward`` is a
thin shape-adapting wrapper around it. See ``attention.py``, ``rope.py``,
``window.py`` and ``common.py`` for the individual per-element pieces.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .common import AdaSingle, DualStream, TimeEmbedding, get_mlp_cls
from .config import DiTConfig
from .attention import NaSwinAttention


def _patchify(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    # (T, H, W, C) -> (T, H//ph, W//pw, ph*pw*C), matching the source's
    # "(H h) (W w) c -> H W (h w c)" einops pattern (t=1 always in our
    # configs, so the temporal patch axis is dropped entirely).
    T, H, W, C = x.shape
    x = x.view(T, H // ph, ph, W // pw, pw, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(T, H // ph, W // pw, ph * pw * C)


def _unpatchify(x: torch.Tensor, ph: int, pw: int, out_channels: int) -> torch.Tensor:
    # Inverse of _patchify: (T, H', W', ph*pw*C) -> (T, H'*ph, W'*pw, C).
    T, Hp, Wp, _ = x.shape
    x = x.view(T, Hp, Wp, ph, pw, out_channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(T, Hp * ph, Wp * pw, out_channels)


class NaDiTBlock(nn.Module):
    """One ``mmdit_sr`` transformer block: AdaLN-single-modulated windowed
    attention followed by an AdaLN-single-modulated MLP, both dual-stream
    (video/text) with residual connections. Ported from
    dit_3b/nablocks/mmsr_block.py and dit_7b/nablocks/mmsr_block.py, which
    are identical apart from ``is_last_layer``'s ``vid_only`` bookkeeping
    (3B only -- see NaDiT's module docstring for why it's harmless to skip
    the text branch there)."""

    def __init__(
        self,
        *,
        vid_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm_eps: float,
        qk_bias: bool,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        window_counts: Tuple[int, int, int],
        shifted: bool,
        is_last_layer: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        dim = vid_dim

        def norm_factory():
            return operations.RMSNorm(dim, eps=norm_eps, elementwise_affine=False, dtype=dtype, device=device)

        self.attn_norm = DualStream(norm_factory, shared_weights=shared_weights, vid_only=False)
        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=vid_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            norm_eps=norm_eps,
            shared_weights=shared_weights,
            rope_type=rope_type,
            rope_dim=rope_dim,
            window_counts=window_counts,
            shifted=shifted,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp_norm = DualStream(norm_factory, shared_weights=shared_weights, vid_only=is_last_layer)

        mlp_cls = get_mlp_cls(mlp_type)

        def mlp_factory():
            return mlp_cls(dim, expand_ratio, dtype=dtype, device=device, operations=operations)

        self.mlp = DualStream(mlp_factory, shared_weights=shared_weights, vid_only=is_last_layer)

        def ada_factory():
            return AdaSingle(dim, emb_dim, layers=["attn", "mlp"], dtype=dtype, device=device)

        self.ada = DualStream(ada_factory, shared_weights=shared_weights, vid_only=is_last_layer)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        emb: torch.Tensor,
        transformer_options=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(
            vid_attn, txt_attn,
            vid_kwargs=dict(emb=emb, layer="attn", mode="in"),
            txt_kwargs=dict(emb=emb, layer="attn", mode="in"),
        )
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, transformer_options=transformer_options)
        vid_attn, txt_attn = self.ada(
            vid_attn, txt_attn,
            vid_kwargs=dict(emb=emb, layer="attn", mode="out"),
            txt_kwargs=dict(emb=emb, layer="attn", mode="out"),
        )
        vid_attn, txt_attn = vid_attn + vid, txt_attn + txt

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(
            vid_mlp, txt_mlp,
            vid_kwargs=dict(emb=emb, layer="mlp", mode="in"),
            txt_kwargs=dict(emb=emb, layer="mlp", mode="in"),
        )
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(
            vid_mlp, txt_mlp,
            vid_kwargs=dict(emb=emb, layer="mlp", mode="out"),
            txt_kwargs=dict(emb=emb, layer="mlp", mode="out"),
        )
        vid_mlp, txt_mlp = vid_mlp + vid_attn, txt_mlp + txt_attn
        return vid_mlp, txt_mlp


class NaDiT(nn.Module):
    def __init__(self, config: DiTConfig, dtype=None, device=None, operations=None):
        super().__init__()
        assert config.patch_size[0] == 1, "temporal patchify is unused by both shipped configs and not ported"
        self.config = config
        ph, pw = config.patch_size[1], config.patch_size[2]
        self._ph, self._pw = ph, pw

        self.vid_in = operations.Linear(
            config.vid_in_channels * ph * pw, config.vid_dim, dtype=dtype, device=device
        )
        self.txt_in = (
            operations.Linear(config.txt_in_dim, config.txt_dim, dtype=dtype, device=device)
            if config.txt_in_dim != config.txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=config.sinusoidal_dim,
            hidden_dim=config.vid_dim,
            output_dim=config.emb_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        blocks = []
        for i in range(config.num_layers):
            if config.mm_layers is None:
                shared_weights = False
            else:
                shared_weights = not (i < config.mm_layers)
            is_last_layer = config.variant == "3b" and i == config.num_layers - 1
            shifted = i % 2 == 1
            blocks.append(
                NaDiTBlock(
                    vid_dim=config.vid_dim,
                    emb_dim=config.emb_dim,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    expand_ratio=config.expand_ratio,
                    norm_eps=config.norm_eps,
                    qk_bias=config.qk_bias,
                    mlp_type=config.mlp_type,
                    shared_weights=shared_weights,
                    rope_type=config.rope_type,
                    rope_dim=config.rope_dim,
                    window_counts=config.window_counts,
                    shifted=shifted,
                    is_last_layer=is_last_layer,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.vid_out_norm = None
        self.vid_out_ada = None
        if config.has_vid_out_norm:
            self.vid_out_norm = operations.RMSNorm(
                config.vid_dim, eps=config.norm_eps, elementwise_affine=True, dtype=dtype, device=device
            )
            self.vid_out_ada = AdaSingle(
                config.vid_dim, config.emb_dim, layers=["out"], modes=["in"], dtype=dtype, device=device
            )

        self.vid_out = operations.Linear(
            config.vid_dim, config.vid_out_channels * ph * pw, dtype=dtype, device=device
        )

    def _forward_single(self, vid: torch.Tensor, txt: torch.Tensor, timestep, transformer_options=None) -> torch.Tensor:
        ph, pw = self._ph, self._pw
        orig_h, orig_w = vid.shape[1], vid.shape[2]
        pad_h = (-orig_h) % ph
        pad_w = (-orig_w) % pw

        if pad_h or pad_w:
            # SeedVR2 patchifies 2D latent tiles directly, so odd latent sizes
            # need a small spatial pad before the transformer path.
            vid = vid.permute(3, 0, 1, 2).unsqueeze(0)
            vid = F.pad(vid, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
            vid = vid[0].permute(1, 2, 3, 0)

        vid = _patchify(vid, ph, pw)
        vid = self.vid_in(vid)
        txt = self.txt_in(txt)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=vid.device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep[None]
        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        for block in self.blocks:
            vid, txt = block(vid, txt, emb, transformer_options=transformer_options)

        if self.vid_out_norm is not None:
            vid = self.vid_out_norm(vid)
            vid = self.vid_out_ada(vid, emb, layer="out", mode="in")

        vid = self.vid_out(vid)
        vid = _unpatchify(vid, ph, pw, self.config.vid_out_channels)
        if pad_h or pad_w:
            vid = vid[:, :orig_h, :orig_w, :]
        return vid

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor = None,
                control=None, transformer_options=None, **kwargs) -> torch.Tensor:
        """
        ComfyUI diffusion-model calling convention, as invoked by
        ``comfy.model_base.BaseModel._apply_model`` (see ``model/base_model.py``).

        x: (B=1, vid_in_channels, T, H, W) channel-first -- the noisy latent
           already concatenated with the SR condition + mask channels via
           native concat conditioning (``BaseModel.concat_cond``). Batch size
           must be 1 (single video per KSampler call, matching how every
           other part of this port treats a video as one T-length sequence
           rather than a sample batch).
        timestep: (B=1,) tensor -- already converted from sigma to the
           lerp-schedule timestep by ``model_sampling.timestep()``.
        context: (B=1, L, txt_in_dim) channel-first text-embedding tokens
           (from CONDITIONING's cross_attn tensor).
        returns: (B=1, vid_out_channels, T, H, W) channel-first, matching x.
        """
        assert x.shape[0] == 1, "NaDiT only supports batch size 1 (one video per call); use the T dimension for multiple frames"
        vid = x[0].permute(1, 2, 3, 0)  # (T,H,W,C)
        if context is None:
            txt = torch.zeros((0, self.config.txt_in_dim), device=x.device, dtype=x.dtype)
        elif context.ndim == 3:
            txt = context[0]
        elif context.ndim == 2:
            txt = context
        elif context.ndim == 1:
            txt = context.unsqueeze(0)
        else:
            raise RuntimeError(f"Unsupported SeedVR2 text conditioning shape: {tuple(context.shape)}")

        out = self._forward_single(vid, txt, timestep, transformer_options=transformer_options)
        return out.permute(3, 0, 1, 2).unsqueeze(0)  # (1,C,T,H,W)
