"""Lumina NIVR2 Color Correction — matches upscaled output color to a reference image."""

from comfy_api.latest import io

from ..postprocess.color_fix import (
    adaptive_instance_normalization,
    hsv_saturation_histogram_match,
    lab_color_transfer,
    wavelet_adaptive_color_correction,
    wavelet_reconstruction,
)

_METHODS = {
    "lab": lab_color_transfer,
    "wavelet": wavelet_reconstruction,
    "wavelet_adaptive": wavelet_adaptive_color_correction,
    "hsv": hsv_saturation_histogram_match,
    "adain": adaptive_instance_normalization,
}


class LuminaNIVR2ColorCorrection(io.ComfyNode):
    """Color-matches an upscaled image/video to a reference (e.g. the original low-res input)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LuminaNIVR2ColorCorrection",
            display_name="Lumina NIVR2 Color Correction",
            category="Lumina NIVR2",
            description=(
                "Color-matches the content image (e.g. from native VAE Decode) to a style/reference "
                "image (e.g. the original input, resized to the output resolution). Diffusion "
                "upscaling can shift color statistics; this restores them."
            ),
            inputs=[
                io.Image.Input("content", tooltip="Image to correct (the upscaled output)."),
                io.Image.Input("style", tooltip="Reference image to match colors to (must match content's frame count and resolution)."),
                io.Combo.Input("method", options=list(_METHODS.keys()), default="lab",
                    tooltip=(
                        "lab: perceptual color matching (recommended)\n"
                        "wavelet: frequency-based, preserves fine details\n"
                        "wavelet_adaptive: wavelet base with saturation correction\n"
                        "hsv: hue-conditional saturation matching\n"
                        "adain: statistical style transfer"
                    )
                ),
            ],
            outputs=[io.Image.Output(tooltip="Color-corrected image.")]
        )

    @classmethod
    def execute(cls, content, style, method: str = "lab") -> io.NodeOutput:
        content_chw = content.permute(0, 3, 1, 2).mul(2).sub(1)
        style_chw = style.permute(0, 3, 1, 2).mul(2).sub(1)
        corrected = _METHODS[method](content_chw, style_chw)
        out = ((corrected.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).contiguous()
        return io.NodeOutput(out)
