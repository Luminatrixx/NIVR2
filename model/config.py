"""
Frozen hyperparameter configs for the SeedVR2 NaDiT (Native-resolution Diffusion
Transformer) 3B and 7B variants, ported from
``seedvr2_videoupscaler/configs_3b/main.yaml`` and
``seedvr2_videoupscaler/configs_7b/main.yaml``.

Both variants share the same overall architecture (AdaLN-single modulated
MM-DiT with alternating windowed / shifted-windowed 3D attention over a
space-only patchified video latent), but differ in width, depth, MLP type,
RoPE flavor and whether the dual-stream (video/text) weights ever collapse
into a single shared stream for the later layers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DiTConfig:
    variant: str  # "3b" or "7b"

    vid_in_channels: int
    vid_out_channels: int
    vid_dim: int
    txt_in_dim: int
    txt_dim: int
    emb_dim: int

    heads: int
    head_dim: int
    expand_ratio: int

    norm_eps: float
    qk_bias: bool

    patch_size: Tuple[int, int, int]  # (t, h, w), t must be 1 (space-only patchify)
    num_layers: int

    # Number of leading layers that use separate ("dual-stream") video/text
    # weights for attention qkv/out projections and mlp/mlp-norm/ada. Layers
    # at or beyond this index share a single weight set between video and
    # text. ``None`` means the stream never collapses (all layers dual-stream).
    mm_layers: Optional[int]

    mlp_type: str  # "swiglu" (3B) or "normal" (7B, vanilla GELU-tanh 2-layer)

    rope_type: str  # "mmrope3d" (3B, joint vid+txt rope) or "rope3d" (7B, vid-only pixel rope)
    rope_dim: int  # raw dim handed to the rotary embedding (divided by 3 axes internally)

    window_counts: Tuple[int, int, int]  # (nt, nh, nw) target window counts at 720p-equivalent scale

    # Only the 3B variant has an extra vid_out_norm + vid_out_ada stage right
    # before the final unpatchify projection.
    has_vid_out_norm: bool

    sinusoidal_dim: int = 256


DIT_3B_CONFIG = DiTConfig(
    variant="3b",
    vid_in_channels=33,
    vid_out_channels=16,
    vid_dim=2560,
    txt_in_dim=5120,
    txt_dim=2560,
    emb_dim=15360,  # 6 * vid_dim
    heads=20,
    head_dim=128,
    expand_ratio=4,
    norm_eps=1.0e-05,
    qk_bias=False,
    patch_size=(1, 2, 2),
    num_layers=32,
    mm_layers=10,
    mlp_type="swiglu",
    rope_type="mmrope3d",
    rope_dim=128,
    window_counts=(4, 3, 3),
    has_vid_out_norm=True,
)

DIT_7B_CONFIG = DiTConfig(
    variant="7b",
    vid_in_channels=33,
    vid_out_channels=16,
    vid_dim=3072,
    txt_in_dim=5120,
    txt_dim=3072,
    emb_dim=18432,  # 6 * vid_dim
    heads=24,
    head_dim=128,
    expand_ratio=4,
    norm_eps=1.0e-05,
    qk_bias=False,
    patch_size=(1, 2, 2),
    num_layers=36,
    mm_layers=None,  # dit_7b never collapses to a shared stream (shared_qkv=False, shared_mlp=False always)
    mlp_type="normal",
    rope_type="rope3d",
    rope_dim=64,  # head_dim // 2 (dit_7b NaRotaryEmbedding3d(dim=head_dim // 2), no explicit rope_dim in yaml)
    window_counts=(4, 3, 3),
    has_vid_out_norm=False,
)
