"""
Minimal comfy.model_base.BaseModel wiring for NaDiT.

This does NOT go through comfy.model_detection's generic architecture-
guessing heuristics -- SeedVR2's own detection/registration happens once,
at pack-import time, in the top-level __init__.py, which wraps
comfy.model_detection.model_config_from_unet so it returns a
SeedVR2ModelConfig instance (below) for a recognized SeedVR2 checkpoint.
That's what makes the native "Load Diffusion Model" node work: once
model_config_from_unet returns our config, comfy.sd.load_diffusion_model_
state_dict's generic flow takes over unmodified -- it calls
model_config.set_inference_dtype(...) (resolving weight/manual-cast dtype
exactly as for any other native model, respecting UNETLoader's weight_dtype
override) and then model_config.get_model(state_dict, "") (below), then
wraps the result in a standard ModelPatcher and loads weights via
BaseModel.load_model_weights -- all unmodified core code.

Everything else is reused verbatim from BaseModel:
  - Sampling math: model_type=FLOW gives comfy.model_sampling.CONST +
    ModelSamplingDiscreteFlow (default shift=1.0, multiplier=1000), whose
    calculate_denoised (model_input - model_output*sigma) reproduces
    SeedVR2's rectified-flow x_0 = x_T - pred at t=T exactly, with zero
    custom sampling code (comfy/model_sampling.py:86-100,284-319).
  - Concat conditioning: self.concat_keys = ("concat_image", "mask")
    reproduces the 16(noise)+16(condition)+1(mask)=33 channel DiT input via
    BaseModel.concat_cond (comfy/model_base.py:257-312) -- "mask" resolves
    to a constant torch.ones_like(noise)[:, :1] for free whenever no
    denoise_mask is attached (SeedVR2's task mask is always 1.0), and
    "concat_image" appends whatever a CONDITIONING entry's
    "concat_latent_image" key carries. Channel ORDER matters here: this
    tuple's iteration order is the torch.cat order, so it must produce
    [noise, condition, mask] to match NaDiT's vid_in_channels layout.
  - Cross-attention text embedding: BaseModel.extra_conds already turns a
    CONDITIONING entry's tensor (surfaced as kwargs["cross_attn"]) into
    c_crossattn -> NaDiT.forward's `context` argument (comfy/model_base.py:
    320-332) -- no override needed.
  - Weight loading / quantization / VRAM offload: comfy.ops.pick_operations
    + comfy.model_patcher.ModelPatcher, exactly as for any other native
    model -- both driven by core, not by this file.
"""

import json

import torch

import comfy.model_base
import comfy.model_management
import comfy.ops
from comfy.latent_formats import LatentFormat

from .config import DiTConfig
from .dit import NaDiT


class SeedVR2LatentFormat(LatentFormat):
    """Identity scaling: SeedVR2VAE.encode already applies SeedVR2's own
    scaling_factor/shifting_factor, so no further latent_format scaling
    should be layered on top (default LatentFormat.scale_factor=1.0)."""

    latent_channels = 16
    latent_dimensions = 3


def _seedvr2_dit_vid_dim_from_state_dict(state_dict: dict, prefix: str = ""):
    for key in (f"{prefix}vid_in.weight", f"{prefix}vid_in.proj.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return tensor.shape[0]
    return None


def _seedvr2_raw_float8_weight_dtype(state_dict: dict, prefix: str = ""):
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        if not key.endswith(".weight"):
            continue
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return tensor.dtype
    return None


def _remap_seedvr2_dit_key(key: str) -> str:
    if key.startswith("vid_in.proj."):
        return "vid_in." + key[len("vid_in.proj."):]
    if key.startswith("vid_out.proj."):
        return "vid_out." + key[len("vid_out.proj."):]
    return key


def _remap_seedvr2_dit_key_for_saving(key: str) -> str:
    if key.startswith("vid_in."):
        return "vid_in.proj." + key[len("vid_in."):]
    if key.startswith("vid_out."):
        return "vid_out.proj." + key[len("vid_out."):]
    return key


def _seedvr2_float8_quant_marker(dtype: torch.dtype) -> torch.Tensor:
    format_name = str(dtype).split(".")[-1]
    payload = json.dumps({"format": format_name}).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _seedvr2_is_quantized_linear_weight_key(key: str) -> bool:
    if key in {"vid_in.weight", "txt_in.weight", "vid_out.weight"}:
        return True
    linear_fragments = (
        ".proj_in.weight",
        ".proj_in_gate.weight",
        ".proj_hid.weight",
        ".proj_out.weight",
        ".proj_qkv.vid.weight",
        ".proj_qkv.txt.weight",
    )
    return key.endswith(linear_fragments)


class SeedVR2ModelConfig:
    """
    Minimal comfy.supported_models_base.BASE-compatible config, built
    directly (not as a comfy.supported_models.models list entry) -- only
    the attributes/methods comfy.sd.load_diffusion_model_state_dict and
    comfy.model_base.BaseModel.__init__ actually read/call are provided:
    unet_config, latent_format, sampling_settings, manual_cast_dtype,
    custom_operations, optimizations, memory_usage_factor, quant_config,
    supported_inference_dtypes, set_inference_dtype(), get_model().
    """

    def __init__(self, dit_config: DiTConfig):
        self.dit_config = dit_config
        self.unet_config = {"disable_unet_model_creation": True}
        self.latent_format = SeedVR2LatentFormat()
        self.sampling_settings = {}
        self.manual_cast_dtype = None
        self.custom_operations = None
        self.optimizations = {"fp8": False}
        self.memory_usage_factor = 2.0
        self.quant_config = None
        self.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype

    def process_unet_state_dict(self, state_dict):
        remapped = {}
        for key, value in state_dict.items():
            # Rope tables in the published checkpoints are deterministic
            # caches, not trainable parameters, so the port recomputes them.
            if key.endswith(".attn.rope.rope.freqs"):
                continue
            mapped_key = _remap_seedvr2_dit_key(key)
            remapped[mapped_key] = value
            if (
                self.quant_config
                and _seedvr2_is_quantized_linear_weight_key(mapped_key)
                and value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            ):
                key_prefix = mapped_key[:-len("weight")]
                quant_key = key_prefix + "comfy_quant"
                scale_key = key_prefix + "weight_scale"
                if quant_key not in remapped:
                    remapped[quant_key] = _seedvr2_float8_quant_marker(value.dtype)
                if scale_key not in remapped:
                    remapped[scale_key] = torch.tensor(1.0, dtype=torch.float32)
        return remapped

    def process_unet_state_dict_for_saving(self, state_dict):
        return {_remap_seedvr2_dit_key_for_saving(key): value for key, value in state_dict.items()}

    def get_model(self, state_dict, prefix="", device=None):
        model = SeedVR2Model(self, device=device)

        param_dtype = self.unet_config.get("dtype", None)
        operations = comfy.ops.pick_operations(
            param_dtype, self.manual_cast_dtype,
            load_device=comfy.model_management.get_torch_device(),
            fp8_optimizations=self.optimizations.get("fp8", False),
            model_config=self,
        )
        model.diffusion_model = NaDiT(self.dit_config, dtype=param_dtype, device=device, operations=operations)
        model.diffusion_model.eval()
        model.diffusion_model.dtype = param_dtype
        return model


class SeedVR2Model(comfy.model_base.BaseModel):
    def __init__(self, model_config: SeedVR2ModelConfig, device=None):
        super().__init__(model_config, model_type=comfy.model_base.ModelType.FLOW, device=device)
        self.concat_keys = ("concat_image", "mask")
