"""SeedVR2 DiT Settings — SeedVR2-specific pre-sampling preparation for a native MODEL."""

from typing import Any, Callable, Dict, List, Tuple

import torch
import node_helpers
from comfy_api.latest import io

import comfy.model_sampling
from comfy.ldm.modules import attention as comfy_attention

from ..model.sampling import augment_condition_latent

Conditioning = List[Tuple[torch.Tensor, Dict[str, Any]]]

_ATTENTION_BACKEND_OPTIONS = [
    "none",
    "sageattn (2 and under)",
    "sageattn 3",
    "flash attention 2",
]
_ATTENTION_BACKEND_MAP = {
    "none": None,
    "sageattn (2 and under)": "sage",
    "sageattn 3": "sage3",
    "flash attention 2": "flash",
}


def _resolve_attention_override(label: str) -> Callable | None:
    backend_name = _ATTENTION_BACKEND_MAP.get(label)
    if backend_name is None:
        return None

    attention_func = comfy_attention.get_attention_function(backend_name, default=None)
    if attention_func is None:
        raise RuntimeError(
            f"SeedVR2 attention backend '{label}' is not available in this ComfyUI install."
        )

    def attention_override(_current_func, *args, **kwargs):
        return attention_func(*args, **kwargs)

    return attention_override


class LuminaNIVR2PatchDiT(io.ComfyNode):
    """
    Prepares a native-loaded SeedVR2 MODEL for native KSampler: installs the
    rectified-flow model_sampling (shift-tunable, mirrors ModelSamplingSD3),
    and attaches the SR condition latent (+ optional noise augmentation) to
    positive/negative CONDITIONING via the same concat-conditioning
    mechanism native inpainting models use.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LuminaNIVR2PatchDiT",
            display_name="SeedVR2 DiT Settings",
            category="Lumina NIVR2",
            description=(
                "Prepares a SeedVR2 MODEL (from native Load Diffusion Model) for native KSampler. "
                "Installs the rectified-flow sampling schedule and attaches the low-res condition "
                "latent (from native VAE Encode) as concat conditioning -- the 33-channel DiT input "
                "(16 noisy + 16 condition + 1 constant task mask) is built entirely through native "
                "concat-conditioning, the same mechanism inpainting models use."
            ),
            inputs=[
                io.Model.Input("model", tooltip="SeedVR2 DiT model from native Load Diffusion Model."),
                io.Latent.Input("condition_latent",
                    tooltip="Low-res latent to upscale: resize your input image to the target "
                            "resolution (native Image Scale) then VAE-encode it (native VAE Encode) "
                            "using the SeedVR2 VAE from native Load VAE."
                ),
                io.Conditioning.Input("positive", tooltip="Positive conditioning from SeedVR2 Text Conditioning."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning from SeedVR2 Text Conditioning."),
                io.Float.Input("shift", default=1.0, min=0.0, max=100.0, step=0.01, optional=True,
                    tooltip="Rectified-flow timestep shift. 1.0 matches the original one-step recipe."
                ),
                io.Float.Input("latent_noise_scale", default=0.0, min=0.0, max=1.0, step=0.001, optional=True,
                    tooltip="Latent-space noise augmentation of the SR condition before sampling (0.0 = disabled)."
                ),
                io.Combo.Input(
                    "attention_backend",
                    options=_ATTENTION_BACKEND_OPTIONS,
                    default="none",
                    optional=True,
                    tooltip=(
                        "Overrides the DiT attention backend for this model only. "
                        "'none' keeps ComfyUI's current default attention selection."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model", tooltip="Patched MODEL, ready for KSampler."),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                    tooltip="Starting latent for KSampler's latent_image input (use denoise=1.0)."),
            ]
        )

    @classmethod
    def execute(cls, model, condition_latent: Dict[str, Any], positive: Conditioning, negative: Conditioning,
                shift: float = 1.0, latent_noise_scale: float = 0.0,
                attention_backend: str = "none") -> io.NodeOutput:
        cond_samples = augment_condition_latent(condition_latent["samples"], latent_noise_scale)

        m = model.clone()

        class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=1000)
        m.add_object_patch("model_sampling", model_sampling)

        attention_override = _resolve_attention_override(attention_backend)
        if attention_override is None:
            m.model_options["transformer_options"].pop("optimized_attention_override", None)
        else:
            m.model_options["transformer_options"]["optimized_attention_override"] = attention_override

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": cond_samples})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": cond_samples})

        empty_latent = {"samples": torch.zeros_like(cond_samples)}

        return io.NodeOutput(m, positive, negative, empty_latent)
