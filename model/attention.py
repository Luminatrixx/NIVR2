"""
Windowed multi-modal (video+text) attention for NaDiT, ported from
dit_3b/nablocks/attention/mmattn.py's ``NaSwinAttention`` and
dit_7b/nablocks/mmsr_block.py's ``NaSwinAttention`` (both variants only ever
instantiate the ``mmdit_sr`` block, which uses this windowed path -- the
plain full-attention ``NaMMAttention`` / the batch-oriented
``MMWindowAttention`` in dit_7b/blocks/mmdit_window_block.py are dead code
for the configs this pack ships and are intentionally not ported).

Single-video (batch=1) simplification of the original varlen/sequence-parallel
implementation:
- Video is windowed via ``window.compute_windows`` / ``group_windows``; each
  same-shape group of windows is processed as one batch through
  ``optimized_attention`` (no cu_seqlens/flash-varlen machinery needed).
- Text tokens are repeated (broadcast, not copied) into every window and
  concatenated for joint intra-window attention; the text *output* is the
  equal-weight average of its per-window attention outputs, matching the
  original's ``repeat_concat_idx`` / coalescing "unconcat" behavior.
- RoPE (when present) is applied per window using the window-local (t, h, w)
  shape as the coordinate frame, exactly as the source does via
  ``window_shape`` rather than the global video shape.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from comfy.ldm.modules.attention import optimized_attention

from .common import DualStream
from .rope import get_rope
from .window import compute_windows, group_windows


class NaSwinAttention(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        norm_eps: float,
        shared_weights: bool,
        rope_type: Optional[str],
        rope_dim: int,
        window_counts: Tuple[int, int, int],
        shifted: bool,
        dtype=None,
        device=None,
        operations=None,
    ):
        assert vid_dim == txt_dim, "NaSwinAttention assumes matching vid/txt dims"
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.window_counts = window_counts
        self.shifted = shifted

        dim = vid_dim

        self.proj_qkv = DualStream(
            lambda: operations.Linear(dim, self.inner_dim * 3, bias=qk_bias, dtype=dtype, device=device),
            shared_weights=shared_weights,
        )
        self.proj_out = DualStream(
            lambda: operations.Linear(self.inner_dim, dim, bias=True, dtype=dtype, device=device),
            shared_weights=shared_weights,
        )
        self.norm_q = DualStream(
            lambda: operations.RMSNorm(head_dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device),
            shared_weights=shared_weights,
        )
        self.norm_k = DualStream(
            lambda: operations.RMSNorm(head_dim, eps=norm_eps, elementwise_affine=True, dtype=dtype, device=device),
            shared_weights=shared_weights,
        )

        if rope_type is not None:
            self.rope, self.rope_is_mm = get_rope(rope_type, rope_dim, dtype=dtype, device=device)
        else:
            self.rope, self.rope_is_mm = None, False

    def forward(self, vid: torch.Tensor, txt: torch.Tensor, transformer_options=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # vid: (T, H, W, vid_dim) ; txt: (Lt, txt_dim)
        T, H, W, _ = vid.shape
        Lt = txt.shape[0]
        heads, head_dim, inner_dim = self.heads, self.head_dim, self.inner_dim

        vid_qkv = self.proj_qkv.apply_vid(vid)  # (T, H, W, 3*inner_dim)
        txt_qkv = self.proj_qkv.apply_txt(txt)  # (Lt, 3*inner_dim)

        txt_qkv = txt_qkv.view(Lt, 3, heads, head_dim)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)  # (Lt, heads, head_dim)
        txt_q = self.norm_q.apply_txt(txt_q)
        txt_k = self.norm_k.apply_txt(txt_k)

        if self.rope is not None and self.rope_is_mm:
            txt_q, txt_k = self.rope.apply_txt(txt_q, txt_k, Lt)

        windows = compute_windows(T, H, W, self.window_counts, shifted=self.shifted)
        groups = group_windows(windows)
        total_windows = len(windows)

        vid_out = torch.zeros(T, H, W, inner_dim, dtype=vid.dtype, device=vid.device)
        txt_out_sum = torch.zeros(Lt, inner_dim, dtype=txt.dtype, device=txt.device)

        for shape, idxs in groups.items():
            tw, hw, ww = shape
            n = len(idxs)
            Lw = tw * hw * ww

            batch_qkv = torch.stack(
                [vid_qkv[windows[i][0], windows[i][1], windows[i][2]].reshape(Lw, 3 * inner_dim) for i in idxs],
                dim=0,
            )  # (n, Lw, 3*inner_dim)
            batch_qkv = batch_qkv.view(n, Lw, 3, heads, head_dim)
            vid_q, vid_k, vid_v = batch_qkv.unbind(2)  # (n, Lw, heads, head_dim)

            vid_q = self.norm_q.apply_vid(vid_q)
            vid_k = self.norm_k.apply_vid(vid_k)

            if self.rope is not None:
                if self.rope_is_mm:
                    vid_q, vid_k = self.rope.apply_vid(vid_q, vid_k, (tw, hw, ww), Lt)
                else:
                    vid_q, vid_k = self.rope.apply_vid(vid_q, vid_k, (tw, hw, ww))

            txt_q_b = txt_q.unsqueeze(0).expand(n, -1, -1, -1)
            txt_k_b = txt_k.unsqueeze(0).expand(n, -1, -1, -1)
            txt_v_b = txt_v.unsqueeze(0).expand(n, -1, -1, -1)

            q = torch.cat([vid_q, txt_q_b], dim=1).reshape(n, Lw + Lt, inner_dim)
            k = torch.cat([vid_k, txt_k_b], dim=1).reshape(n, Lw + Lt, inner_dim)
            v = torch.cat([vid_v, txt_v_b], dim=1).reshape(n, Lw + Lt, inner_dim)

            out = optimized_attention(
                q, k, v, heads, mask=None, transformer_options=transformer_options
            ).type_as(vid_q)  # (n, Lw+Lt, inner_dim)

            vid_part = out[:, :Lw, :].reshape(n, tw, hw, ww, inner_dim)
            txt_part = out[:, Lw:, :]  # (n, Lt, inner_dim)

            for j, i in enumerate(idxs):
                ts, hs, ws = windows[i]
                vid_out[ts, hs, ws] = vid_part[j]

            txt_out_sum = txt_out_sum + txt_part.sum(dim=0)

        txt_out = txt_out_sum / total_windows

        vid_out = self.proj_out.apply_vid(vid_out)
        txt_out = self.proj_out.apply_txt(txt_out)
        return vid_out, txt_out
