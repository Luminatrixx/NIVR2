"""
Lumina NIVR2 - Native Implementation of SeedVR2

Registers SeedVR2 checkpoints with ComfyUI's native "Load Diffusion Model"
and "Load VAE" nodes (via two narrowly-scoped monkeypatches -- see
_register_dit_detection/_register_vae_detection below) so the rest of the
graph is built entirely from native nodes: VAE Encode, VAE Decode, KSampler.
Only two small "Patch" nodes (dit_patch.py, vae_patch.py) plus two tiny
loader nodes for pieces with no native equivalent (a fixed-embedding
CONDITIONING source, and color correction) are custom. The published
SeedVR2 VAE checkpoint uses diffusers-style parameter names, so VAE loading
below remaps those names onto this port's thin wrapper modules before
calling ``load_state_dict``.
"""

import os
import re

import torch

import folder_paths
import comfy.model_detection
import comfy.model_management
import comfy.sd
import comfy.model_patcher
import comfy.utils

from .model.base_model import (
    SeedVR2ModelConfig,
    _seedvr2_dit_vid_dim_from_state_dict,
    _seedvr2_raw_float8_weight_dtype,
)
from .model.config import DIT_3B_CONFIG, DIT_7B_CONFIG
from .model.vae import SeedVR2VAE


def _register_model_folders():
    models_dir = folder_paths.models_dir
    folder_paths.add_model_folder_path("diffusion_models", os.path.join(models_dir, "nivr2_dit"))
    folder_paths.add_model_folder_path("vae", os.path.join(models_dir, "nivr2_vae"))


def _detect_seedvr2_dit(state_dict: dict, unet_key_prefix: str):
    """
    Distinguishes a SeedVR2 NaDiT checkpoint (3B or 7B) by the vid_in
    projection's output width (vid_dim), which is architecture-distinctive:
    2560 for 3B, 3072 for 7B. Returns a DiTConfig, or None if not a match.
    """
    vid_dim = _seedvr2_dit_vid_dim_from_state_dict(state_dict, unet_key_prefix)
    if vid_dim is None:
        return None
    if vid_dim == DIT_3B_CONFIG.vid_dim:
        return DIT_3B_CONFIG
    if vid_dim == DIT_7B_CONFIG.vid_dim:
        return DIT_7B_CONFIG
    return None


def _register_dit_detection():
    orig_model_config_from_unet = comfy.model_detection.model_config_from_unet

    def patched_model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
        dit_config = _detect_seedvr2_dit(state_dict, unet_key_prefix)
        if dit_config is not None:
            model_config = SeedVR2ModelConfig(dit_config)
            if _seedvr2_raw_float8_weight_dtype(state_dict, unet_key_prefix) is not None:
                model_config.quant_config = {"mixed_ops": True}
            quant_config = comfy.utils.detect_layer_quantization(state_dict, unet_key_prefix)
            if quant_config:
                model_config.quant_config = quant_config
            return model_config
        return orig_model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=use_base_if_no_match, metadata=metadata)

    comfy.model_detection.model_config_from_unet = patched_model_config_from_unet


def _is_seedvr2_vae_state_dict(state_dict: dict) -> bool:
    if state_dict is None:
        return False
    return (
        "encoder.conv_in.weight" in state_dict
        and "decoder.up_blocks.0.upsamplers.0.upscale_conv.weight" in state_dict
    )


def _remap_seedvr2_vae_key(key: str) -> str:
    key = re.sub(r"^(encoder|decoder)\.conv_in\.", r"\1.conv_in.conv.", key)
    key = re.sub(r"^(encoder|decoder)\.conv_out\.", r"\1.conv_out.conv.", key)
    key = key.replace(".mid_block.resnets.0.", ".mid_block.resnet1.")
    key = key.replace(".mid_block.resnets.1.", ".mid_block.resnet2.")
    key = key.replace(".mid_block.attentions.0.group_norm.", ".mid_block.attention.norm.")
    key = key.replace(".mid_block.attentions.0.to_q.", ".mid_block.attention.to_q.")
    key = key.replace(".mid_block.attentions.0.to_k.", ".mid_block.attention.to_k.")
    key = key.replace(".mid_block.attentions.0.to_v.", ".mid_block.attention.to_v.")
    key = key.replace(".mid_block.attentions.0.to_out.0.", ".mid_block.attention.to_out.")
    key = key.replace(".downsamplers.0.conv.", ".downsampler.conv.conv.")
    key = key.replace(".upsamplers.0.conv.", ".upsampler.conv.conv.")
    key = key.replace(".upsamplers.0.upscale_conv.", ".upsampler.upscale_conv.")
    key = re.sub(r"\.(conv1|conv2|conv_shortcut)\.", r".\1.conv.", key)
    return key


def _remap_seedvr2_vae_state_dict(state_dict: dict) -> dict:
    return {_remap_seedvr2_vae_key(key): value for key, value in state_dict.items()}


def _register_vae_detection():
    orig_vae_init = comfy.sd.VAE.__init__

    def patched_vae_init(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        if _is_seedvr2_vae_state_dict(sd):
            self.memory_used_encode = lambda shape, dtype: (
                1400 * max(1, shape[2]) * shape[3] * shape[4]
            ) * comfy.model_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (
                2800 * max(1, ((shape[2] - 1) * 4) + 1) * shape[3] * shape[4] * (8 * 8)
            ) * comfy.model_management.dtype_size(dtype)
            self.process_input = lambda image: image * 2.0 - 1.0
            self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
            self.disable_offload = False
            self.downscale_ratio = (lambda a: max(0, (a - 1) // 4 + 1), 8, 8)
            self.upscale_ratio = (lambda a: max(0, (a - 1) * 4 + 1), 8, 8)
            self.downscale_index_formula = (4, 8, 8)
            self.upscale_index_formula = (4, 8, 8)
            self.latent_dim = 3
            self.latent_channels = 16
            self.output_channels = 3
            self.pad_channel_value = None
            self.not_video = False
            self.size = None
            self.extra_1d_channel = None
            self.crop_input = True
            self.audio_sample_rate = 44100

            self.first_stage_model = SeedVR2VAE(dtype=None, device=None).eval()

            if device is None:
                device = comfy.model_management.vae_device()
            self.device = device
            offload_device = comfy.model_management.vae_offload_device()
            if dtype is None:
                dtype = comfy.model_management.vae_dtype(self.device, self.working_dtypes)
            self.vae_dtype = dtype
            self.first_stage_model.to(self.vae_dtype)
            comfy.model_management.archive_model_dtypes(self.first_stage_model)
            self.output_device = comfy.model_management.intermediate_device()

            mp = comfy.model_patcher.CoreModelPatcher
            if self.disable_offload:
                mp = comfy.model_patcher.ModelPatcher
            self.patcher = mp(self.first_stage_model, load_device=self.device, offload_device=offload_device)
            mapped_sd = _remap_seedvr2_vae_state_dict(sd)
            missing, unexpected = self.first_stage_model.load_state_dict(
                mapped_sd,
                strict=False,
                assign=self.patcher.is_dynamic(),
            )
            if missing or unexpected:
                raise RuntimeError(
                    "SeedVR2 VAE checkpoint remap left unresolved keys: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            self.model_size()
            return

        orig_vae_init(self, sd, device, config, dtype, metadata)

    comfy.sd.VAE.__init__ = patched_vae_init


_register_model_folders()
_register_dit_detection()
_register_vae_detection()

from comfy_api.latest import ComfyExtension, io

from .nodes.dit_patch import LuminaNIVR2PatchDiT
from .nodes.vae_patch import LuminaNIVR2PatchVAE
from .nodes.text_embedding import LuminaNIVR2TextEmbedding
from .nodes.color_correction import LuminaNIVR2ColorCorrection


class LuminaNIVR2Extension(ComfyExtension):
    """Lumina NIVR2 ComfyUI Extension"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LuminaNIVR2PatchDiT,
            LuminaNIVR2PatchVAE,
            LuminaNIVR2TextEmbedding,
            LuminaNIVR2ColorCorrection,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return LuminaNIVR2Extension()


__all__ = [
    "LuminaNIVR2PatchDiT",
    "LuminaNIVR2PatchVAE",
    "LuminaNIVR2TextEmbedding",
    "LuminaNIVR2ColorCorrection",
    "LuminaNIVR2Extension",
    "comfy_entrypoint",
]
