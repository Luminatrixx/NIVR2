"""
Shared building blocks for NaDiT: sinusoidal timestep embedding, the
AdaLN-single modulation layer, the dual-stream (video/text) weight-sharing
wrapper, and the two MLP variants (SwiGLU for 3B, vanilla GELU-tanh for 7B).

Ported from dit_3b/{embedding,modulation,mm,mlp}.py and
dit_7b/{embedding,modulation,mm,mlp}.py, which are identical between the two
variants except for the extra ``vid_only`` bookkeeping needed by 3B's
mm-layer collapse. Every learnable layer is built through ``operations`` so
dtype casting and comfy-kitchen quantization work without any special-casing
here.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


def _init_parameter(shape, *, scale: float = 1.0, offset: float = 0.0, dtype=None, device=None) -> nn.Parameter:
    """
    Seed parameters in float32, then cast to the requested dtype. Some
    dtypes used for checkpoint storage/loading (notably float8 on CPU) do
    not implement random sampling kernels directly.
    """
    value = torch.randn(shape, dtype=torch.float32, device=device)
    if scale != 1.0:
        value = value / scale
    if offset != 0.0:
        value = value + offset
    if dtype is not None:
        value = value.to(dtype)
    return nn.Parameter(value)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """
    Matches ``diffusers.models.embeddings.get_timestep_embedding`` with
    ``flip_sin_to_cos=False, downscale_freq_shift=0`` exactly (dim is always
    even for our configs, so no odd-dim padding branch is needed).
    """
    half_dim = dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim
    freqs = torch.exp(exponent)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = operations.Linear(sinusoidal_dim, hidden_dim, dtype=dtype, device=device)
        self.proj_hid = operations.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden_dim, output_dim, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, timestep, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = sinusoidal_timestep_embedding(timestep, self.sinusoidal_dim)
        emb = emb.to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb


def _expand_leading(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Prepend singleton dims so ``x`` has ``ndim`` total dims, e.g.
    (dim, 3) -> (1, ..., 1, dim, 3)."""
    return x.reshape((1,) * (ndim - x.ndim) + tuple(x.shape))


class AdaSingle(nn.Module):
    """
    AdaLN-single modulation: a shared per-(vid|txt)-stream embedding table
    ``emb`` of size ``6 * dim`` is split into (shift, scale, gate) triples,
    one triple per named ``layer`` (e.g. "attn", "mlp"). ``modes`` controls
    whether the "in" (shift/scale) and/or "out" (gate) parameters exist at
    all -- the final vid_out_ada stage (3B only) only ever uses mode="in".

    Assumes a single sample (batch size 1): ``emb`` has shape (1, emb_dim).
    """

    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: Sequence[str],
        modes: Sequence[str] = ("in", "out"),
        dtype=None,
        device=None,
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = list(layers)
        self.modes = set(modes)
        for layer in self.layers:
            if "in" in self.modes:
                self.register_parameter(
                    f"{layer}_shift", _init_parameter(dim, scale=dim**0.5, dtype=dtype, device=device)
                )
                self.register_parameter(
                    f"{layer}_scale",
                    _init_parameter(dim, scale=dim**0.5, offset=1.0, dtype=dtype, device=device),
                )
            if "out" in self.modes:
                self.register_parameter(
                    f"{layer}_gate", _init_parameter(dim, scale=dim**0.5, dtype=dtype, device=device)
                )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        expected = set()
        for layer in self.layers:
            if "in" in self.modes:
                expected.add(f"{layer}_shift")
                expected.add(f"{layer}_scale")
            if "out" in self.modes:
                expected.add(f"{layer}_gate")

        prefix_len = len(prefix)
        seen = set()
        for full_key, value in state_dict.items():
            if not full_key.startswith(prefix):
                continue
            local_key = full_key[prefix_len:]
            if local_key not in expected:
                unexpected_keys.append(full_key)
                continue
            if not assign_to_params_buffers:
                value = value.clone()
            setattr(self, local_key, nn.Parameter(value, requires_grad=False))
            seen.add(local_key)

        for local_key in expected:
            if local_key not in seen:
                missing_keys.append(prefix + local_key)

    def forward(self, hid: torch.Tensor, emb: torch.Tensor, layer: str, mode: str) -> torch.Tensor:
        idx = self.layers.index(layer)
        # emb always encodes `emb_dim // (3 * dim)` (shift, scale, gate) slots
        # -- for the regular per-block ada this equals len(layers) (2:
        # "attn"/"mlp"), but for the 3B-only single-layer vid_out_ada
        # (layers=["out"], emb_dim still == 6 * dim) it is 2 slots wide even
        # though only slot 0 ("out") is named/consumed; deriving the slot
        # count from len(layers) instead (as the upstream source literally
        # does) makes that instance's rearrange come out at 2*dim instead of
        # dim, which mismatches the registered (dim,)-shaped parameters.
        num_slots = emb.shape[-1] // (self.dim * 3)
        e = emb.reshape(emb.shape[0], self.dim, num_slots, 3)[0, :, idx, :]
        e = _expand_leading(e, hid.ndim + 1)  # (1, ..., 1, dim, 3)

        if mode == "in":
            shiftA, scaleA, _ = e.unbind(-1)
            shiftB = getattr(self, f"{layer}_shift")
            scaleB = getattr(self, f"{layer}_scale")
            if shiftB.dtype != hid.dtype:
                shiftB = shiftB.to(hid.dtype)
            if scaleB.dtype != hid.dtype:
                scaleB = scaleB.to(hid.dtype)
            return hid * (scaleA + scaleB) + (shiftA + shiftB)
        if mode == "out":
            _, _, gateA = e.unbind(-1)
            gateB = getattr(self, f"{layer}_gate")
            if gateB.dtype != hid.dtype:
                gateB = gateB.to(hid.dtype)
            return hid * (gateA + gateB)
        raise NotImplementedError(mode)


class DualStream(nn.Module):
    """
    Wraps a zero-arg module factory to build either one shared submodule
    (``shared_weights=True``, stored as ``self.all``) or two independent
    "vid" / "txt" submodules. When ``vid_only`` is set, no txt submodule is
    built at all and the txt branch is passed through unchanged -- this
    mirrors the 3B last-layer behavior where the extra vid_out_norm/ada stage
    makes any further txt-branch computation dead weight.

    Ported from dit_3b/mm.py and dit_7b/mm.py's ``MMModule`` (dims are always
    equal between vid/txt in our configs, so the single-factory design here
    is equivalent and simpler than the original's parallel-arg plumbing).
    """

    def __init__(self, factory: Callable[[], nn.Module], shared_weights: bool = False, vid_only: bool = False):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if shared_weights:
            self.all = factory()
        else:
            self.vid = factory()
            self.txt = factory() if not vid_only else None

    def apply_vid(self, x, **kwargs):
        module = self.all if self.shared_weights else self.vid
        return module(x, **kwargs)

    def apply_txt(self, x, **kwargs):
        if self.vid_only:
            return x
        module = self.all if self.shared_weights else self.txt
        return module(x, **kwargs)

    def forward(
        self,
        vid: torch.Tensor,
        txt: torch.Tensor,
        vid_kwargs: Optional[Dict] = None,
        txt_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vid = self.apply_vid(vid, **(vid_kwargs or {}))
        txt = self.apply_txt(txt, **(txt_kwargs or {}))
        return vid, txt


class MLP(nn.Module):
    """Vanilla 2-layer GELU-tanh MLP (7B ``mlp_type="normal"``)."""

    def __init__(self, dim: int, expand_ratio: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj_in = operations.Linear(dim, dim * expand_ratio, dtype=dtype, device=device)
        self.act = nn.GELU("tanh")
        self.proj_out = operations.Linear(dim * expand_ratio, dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.act(self.proj_in(x)))


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP (3B ``mlp_type="swiglu"``)."""

    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.proj_in = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(nn.functional.silu(self.proj_in_gate(x)) * self.proj_in(x))


def get_mlp_cls(mlp_type: str):
    if mlp_type == "swiglu":
        return SwiGLUMLP
    if mlp_type == "normal":
        return MLP
    raise NotImplementedError(f"{mlp_type} is not supported")
