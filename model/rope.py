"""
3D rotary position embeddings for NaDiT, ported from dit_3b/rope.py and
dit_7b/rope.py. The original modules wrap the ``rotary_embedding_torch``
package (lucidrains); the relevant subset of that package's math
(``RotaryEmbedding.forward`` / ``get_axial_freqs`` / ``apply_rotary_emb``,
with the GPT-NeoX-style *interleaved* rotate-half convention) is reproduced
here directly so this node pack has no external rope dependency.

Two flavors are used:
- ``RotaryEmbedding3d`` (7B): pixel-style axial rope (``freqs_for="pixel"``,
  ``max_freq=256``) applied to video tokens only. Text tokens are not roped.
- ``NaMMRotaryEmbedding3d`` (3B): LLM-style axial rope (``freqs_for="lang"``,
  ``theta=10000``) applied jointly to video *and* text tokens, sharing one
  coordinate frame: text occupies temporal positions [0, txt_len) and video
  occupies [txt_len, txt_len + T), with H/W positions starting at 0 for
  video and text reusing the axis-0 (temporal) frequency table broadcast
  across all 3 axes (since text has no spatial extent of its own).

In both cases, when applied inside windowed attention the "video shape" fed
in is the *window-local* (t, h, w) extent (positions restart at 0 in every
window), matching the source's use of ``window_shape`` rather than the
global video shape.
"""

import math
from typing import Tuple

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # GPT-NeoX / lucidrains "interleaved" convention: pairs (x[2i], x[2i+1]).
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    freqs: (..., rot_dim), broadcastable against t's leading dims.
    t: (..., d) with d >= rot_dim; only the first rot_dim channels are rotated.
    """
    rot_dim = freqs.shape[-1]
    t_mid, t_right = t[..., :rot_dim], t[..., rot_dim:]
    out = t_mid * freqs.cos() + _rotate_half(t_mid) * freqs.sin()
    if t_right.shape[-1] > 0:
        out = torch.cat([out, t_right], dim=-1)
    return out


class AxialRotaryEmbedding(nn.Module):
    """
    Builds per-axis frequency tables and combines them into an N-axis axial
    frequency grid, matching ``rotary_embedding_torch.RotaryEmbedding``'s
    ``forward`` + ``get_axial_freqs`` for ``freqs_for in {"lang", "pixel"}``.
    """

    def __init__(
        self,
        dim: int,
        n_axes: int = 3,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 256.0,
        device=None,
    ):
        super().__init__()
        axis_dim = dim // n_axes
        if freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32)[: axis_dim // 2] / axis_dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, axis_dim // 2, dtype=torch.float32) * math.pi
        else:
            raise ValueError(f"unsupported freqs_for={freqs_for!r}")
        self.freqs_for = freqs_for
        self.n_axes = n_axes
        self.register_buffer("freqs", freqs, persistent=False)
        if device is not None:
            self.freqs = self.freqs.to(device)

    def _axis_freqs(self, pos: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("n,f->nf", pos.to(self.freqs.dtype), self.freqs)
        return freqs.repeat_interleave(2, dim=-1)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        device = self.freqs.device
        per_axis = []
        for ind, d in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1.0, 1.0, steps=d, device=device)
            else:
                pos = torch.arange(d, device=device, dtype=torch.float32)
            f = self._axis_freqs(pos)  # (d, axis_freq_len)
            shape = [1] * len(dims) + [f.shape[-1]]
            shape[ind] = d
            per_axis.append(f.reshape(shape))
        per_axis = torch.broadcast_tensors(*per_axis)
        return torch.cat(per_axis, dim=-1)


class RotaryEmbedding3d(nn.Module):
    """7B video-only pixel-style rope. ``dim`` is ``head_dim // 2``."""

    def __init__(self, dim: int, dtype=None, device=None):
        super().__init__()
        self.axial = AxialRotaryEmbedding(dim, n_axes=3, freqs_for="pixel", max_freq=256.0, device=device)

    def apply_vid(
        self,
        vid_q: torch.Tensor,  # (..., L, heads, d) with L == T*H*W
        vid_k: torch.Tensor,
        vid_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W = vid_size
        freqs = self.axial.get_axial_freqs(T, H, W).reshape(T * H * W, -1)
        freqs = freqs.unsqueeze(-2).to(vid_q.device)  # (L, 1, rot_dim)
        q = apply_rotary_emb(freqs, vid_q.float()).to(vid_q.dtype)
        k = apply_rotary_emb(freqs, vid_k.float()).to(vid_k.dtype)
        return q, k


class NaMMRotaryEmbedding3d(nn.Module):
    """3B joint video+text LLM-style rope. ``dim`` is the raw ``rope_dim`` config value."""

    def __init__(self, dim: int, dtype=None, device=None):
        super().__init__()
        self.axial = AxialRotaryEmbedding(dim, n_axes=3, freqs_for="lang", theta=10000.0, device=device)

    def apply_txt(
        self,
        txt_q: torch.Tensor,  # (L, heads, d)
        txt_k: torch.Tensor,
        txt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Text has no spatial extent: reuse the axis-0 (temporal) frequency
        # table and repeat it across all 3 axes to match video's rot_dim.
        table = self.axial.get_axial_freqs(txt_len)  # (txt_len, axis_freq_len)
        freqs = table.repeat(1, self.axial.n_axes)  # (txt_len, rot_dim)
        freqs = freqs.unsqueeze(-2).to(txt_q.device)  # (txt_len, 1, rot_dim)
        q = apply_rotary_emb(freqs, txt_q.float()).to(txt_q.dtype)
        k = apply_rotary_emb(freqs, txt_k.float()).to(txt_k.dtype)
        return q, k

    def apply_vid(
        self,
        vid_q: torch.Tensor,  # (..., L, heads, d) with L == T*H*W (window-local shape)
        vid_k: torch.Tensor,
        vid_size: Tuple[int, int, int],
        txt_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W = vid_size
        # Video shares the text's temporal coordinate frame, starting right
        # after the text positions: video occupies [txt_len, txt_len + T).
        table = self.axial.get_axial_freqs(txt_len + T, H, W)
        freqs = table[txt_len : txt_len + T, :H, :W].reshape(T * H * W, -1)
        freqs = freqs.unsqueeze(-2).to(vid_q.device)  # (L, 1, rot_dim)
        q = apply_rotary_emb(freqs, vid_q.float()).to(vid_q.dtype)
        k = apply_rotary_emb(freqs, vid_k.float()).to(vid_k.dtype)
        return q, k


def get_rope(rope_type: str, dim: int, dtype=None, device=None):
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim, dtype=dtype, device=device), True
    if rope_type == "rope3d":
        return RotaryEmbedding3d(dim, dtype=dtype, device=device), False
    raise NotImplementedError(f"{rope_type} is not supported")
