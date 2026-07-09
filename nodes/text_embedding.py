"""SeedVR2 Text Conditioning — fixed SR "quality anchor" CONDITIONING (no CLIP model)."""

from comfy_api.latest import io

import comfy.model_management

from ..model.condition import load_negative_text_embedding, load_positive_text_embedding


class LuminaNIVR2TextEmbedding(io.ComfyNode):
    """
    SeedVR2 has no user-facing text encoder -- its DiT was trained with two
    fixed positive/negative "quality anchor" embeddings, shipped as
    pos_emb.pt/neg_emb.pt. This node loads them as standard CONDITIONING,
    analogous to CLIPTextEncode but with no CLIP model, for wiring into
    SeedVR2 DiT Settings / native KSampler.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LuminaNIVR2TextEmbedding",
            display_name="SeedVR2 Text Conditioning",
            category="Lumina NIVR2",
            description=(
                "Loads SeedVR2's fixed positive/negative quality-anchor embeddings as CONDITIONING. "
                "Connect to SeedVR2 DiT Settings and/or KSampler."
            ),
            inputs=[],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        device = comfy.model_management.text_encoder_device()
        dtype = comfy.model_management.text_encoder_dtype(device)
        positive = [[load_positive_text_embedding(device, dtype), {}]]
        negative = [[load_negative_text_embedding(device, dtype), {}]]
        return io.NodeOutput(positive, negative)
