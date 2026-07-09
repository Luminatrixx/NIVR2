"""SeedVR2 VAE Settings — SeedVR2-specific pre-sampling preparation for a native VAE."""

from comfy_api.latest import io


class LuminaNIVR2PatchVAE(io.ComfyNode):
    """
    Tunes the causal video VAE's temporal streaming chunk size -- a
    SeedVR2-VAE-specific speed/memory tradeoff with no native-node
    equivalent (native tiled VAE nodes already cover spatial tiling
    directly against the native-loaded VAE). Larger chunks stream fewer,
    bigger causal-conv passes (faster, more VRAM); smaller chunks stream
    more, smaller passes (slower, less VRAM). Must stay a multiple of 4
    (the causal temporal downsample factor).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LuminaNIVR2PatchVAE",
            display_name="SeedVR2 VAE Settings",
            category="Lumina NIVR2",
            description=(
                "Tunes the causal video VAE's temporal streaming chunk size before use with "
                "native VAE Encode/VAE Decode. Not exposed by any native node since it's specific "
                "to SeedVR2's causal-conv streaming VAE."
            ),
            inputs=[
                io.Vae.Input("vae", tooltip="VAE from native Load VAE (SeedVR2 checkpoint)."),
                io.Int.Input("chunk_size", default=4, min=4, step=4, optional=True,
                    tooltip="Causal temporal streaming chunk size in frames (multiple of 4). "
                            "Larger = faster, more VRAM. Smaller = slower, less VRAM."
                ),
            ],
            outputs=[io.Vae.Output(tooltip="VAE with the streaming chunk size applied.")]
        )

    @classmethod
    def execute(cls, vae, chunk_size: int = 4) -> io.NodeOutput:
        vae.first_stage_model.config.slicing_sample_min_size = max(4, (chunk_size // 4) * 4)
        return io.NodeOutput(vae)
