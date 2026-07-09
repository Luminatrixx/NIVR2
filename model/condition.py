"""
SeedVR2's fixed text-embedding loading.

Ported from ``seedvr2_videoupscaler/src/core/generation_utils.py``
(``load_text_embeddings``). SeedVR2 is SR-only at inference: the "text"
embeddings are fixed positive/negative quality anchors shipped as
``pos_emb.pt``/``neg_emb.pt``, not derived from user input. The original
wrapper hardcoded ``cfg.scale=1.0``, which made ``classifier_free_guidance_
dispatcher`` always skip the negative branch entirely -- so the negative
embedding was never actually used there. Here, native KSampler's ``cfg``
is a user-facing dial (not pinned), so both embeddings are loaded and
exposed via the "SeedVR2 Text Conditioning" node; at cfg=1.0 (the
recommended setting for these one-step distilled checkpoints, matching the
original recipe) ComfyUI's own ``sampling_function`` skips evaluating the
negative branch too.

The SR condition itself (the low-res latent + constant task-mask channel
concatenated onto the noisy latent) is no longer built here -- it's handled
by native concat conditioning via ``model/base_model.py``'s
``concat_keys = ("concat_image", "mask")``.
"""

import logging
import os
import shutil
import threading
import urllib.request

import torch

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_EMBEDDING_DOWNLOAD_BASE = "https://raw.githubusercontent.com/Doudoulix/Lumina_NIVR2/main/assets"
_EMBEDDING_DOWNLOAD_URLS = {
    "pos_emb.pt": (
        f"{_EMBEDDING_DOWNLOAD_BASE}/pos_emb.pt",
        "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/pos_emb.pt",
        "https://huggingface.co/AInVFX/SeedVR2_comfyUI/resolve/main/pos_emb.pt",
    ),
    "neg_emb.pt": (
        f"{_EMBEDDING_DOWNLOAD_BASE}/neg_emb.pt",
        "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/neg_emb.pt",
        "https://huggingface.co/AInVFX/SeedVR2_comfyUI/resolve/main/neg_emb.pt",
    ),
}

_embed_cache: dict = {}
_download_lock = threading.Lock()
_log = logging.getLogger(__name__)


def _ensure_embedding_file(filename: str) -> str:
    path = os.path.join(_ASSETS_DIR, filename)
    if os.path.exists(path):
        return path

    urls = _EMBEDDING_DOWNLOAD_URLS.get(filename, ())
    if not urls:
        raise FileNotFoundError(f"Missing SeedVR2 text embedding with no download source configured: {filename}")

    os.makedirs(_ASSETS_DIR, exist_ok=True)
    with _download_lock:
        if os.path.exists(path):
            return path

        errors = []
        tmp_path = f"{path}.tmp"
        for url in urls:
            try:
                _log.info("Downloading missing SeedVR2 text embedding %s from %s", filename, url)
                request = urllib.request.Request(url, headers={"User-Agent": "Lumina_NIVR2/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
                os.replace(tmp_path, path)
                _log.info("Downloaded SeedVR2 text embedding %s to %s", filename, path)
                return path
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        attempted = "\n".join(errors)
        raise FileNotFoundError(
            f"Missing required SeedVR2 text embedding '{filename}' and auto-download failed.\n{attempted}"
        )


def _load_embedding(filename: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if filename not in _embed_cache:
        _embed_cache[filename] = torch.load(_ensure_embedding_file(filename), weights_only=True)
    return _embed_cache[filename].to(device=device, dtype=dtype)


def load_positive_text_embedding(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns the fixed (L, txt_in_dim=5120) positive quality-anchor embedding."""
    return _load_embedding("pos_emb.pt", device, dtype)


def load_negative_text_embedding(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns the fixed (L, txt_in_dim=5120) negative quality-anchor embedding."""
    return _load_embedding("neg_emb.pt", device, dtype)
