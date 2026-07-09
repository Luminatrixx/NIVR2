"""
SeedVR2's resolution-dependent timestep-shift transform and latent-space
noise augmentation.

Everything else in the original one-step rectified-flow (lerp) sampling
recipe is now reproduced by native KSampler + comfy.model_sampling.CONST/
ModelSamplingDiscreteFlow (see ``model/base_model.py``'s module docstring
for the exact correspondence). What's left here is a genuinely SeedVR2-
specific *pre*-sampling knob with no native equivalent: optionally blending
some noise into the SR condition latent before it's attached as concat
conditioning (ported from ``generation_phases.py``'s ``_add_noise``), used
by the "SeedVR2 DiT Settings" node's ``latent_noise_scale`` input.

Schedule: lerp (rectified flow), ``x_t = A(t)*x_0 + B(t)*x_T`` with
``A(t) = 1 - t/T``, ``B(t) = t/T``, ``T = 1000.0`` (ported from
``common/diffusion/schedules/lerp.py``) -- matches native
ModelSamplingDiscreteFlow's default ``multiplier=1000, shift=1.0``.
"""

import torch

SCHEDULE_T = 1000.0


def _schedule_A(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - t / SCHEDULE_T


def _schedule_B(t: torch.Tensor) -> torch.Tensor:
    return t / SCHEDULE_T


def _schedule_forward(x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return _schedule_A(t) * x_0 + _schedule_B(t) * x_T


def timestep_transform(t: torch.Tensor, latent_thw: tuple, temporal_downsample: int = 4,
                        spatial_downsample: int = 8) -> torch.Tensor:
    """
    Resolution-dependent timestep shift, ported from
    ``VideoDiffusionInfer.timestep_transform`` (``core/infer.py``).

    latent_thw: (T, H, W) of the latent tensor (channel/batch dims excluded).
    """
    t_latent, h_latent, w_latent = latent_thw
    frames = (t_latent - 1) * temporal_downsample + 1
    height = h_latent * spatial_downsample
    width = w_latent * spatial_downsample

    def lin(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    img_shift_fn = lin(256 * 256, 1.0, 1024 * 1024, 3.2)
    vid_shift_fn = lin(256 * 256 * 37, 1.0, 1280 * 720 * 145, 5.0)
    shift = vid_shift_fn(height * width * frames) if frames > 1 else img_shift_fn(height * width)

    t_norm = t / SCHEDULE_T
    t_shifted = shift * t_norm / (1 + (shift - 1) * t_norm)
    return t_shifted * SCHEDULE_T


def augment_condition_latent(latent: torch.Tensor, latent_noise_scale: float) -> torch.Tensor:
    """
    Optional noise augmentation of the SR condition latent, ported from
    ``generation_phases.py``'s ``_add_noise``/``aug_noises`` construction.
    The ``*0.1``/``*0.05`` augmentation-noise mix constants are undocumented
    in any config but load-bearing in the shipped recipe -- preserved
    exactly. No-op when ``latent_noise_scale`` is 0.

    latent: (B, C, T, H, W) channel-first condition latent (native LATENT
        tensor convention, e.g. straight from VAE Encode).
    """
    if latent_noise_scale == 0.0:
        return latent
    base_noise = torch.randn_like(latent)
    aug_noise = base_noise * 0.1 + torch.randn_like(base_noise) * 0.05
    t = torch.tensor([SCHEDULE_T], device=latent.device, dtype=latent.dtype) * latent_noise_scale
    t = timestep_transform(t, tuple(latent.shape[-3:]))
    return _schedule_forward(latent, aug_noise, t)
