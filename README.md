# Lumina NIVR2

Native ComfyUI implementation of ByteDance SeedVR2 using ComfyUI's own model stack and core graph nodes.

Instead of wrapping SeedVR2 in a monolithic custom runtime, this project registers the SeedVR2 DiT and VAE so they can be used with native `Load Diffusion Model`, `Load VAE`, `VAE Encode`, `VAE Decode`, and `KSampler` nodes. The custom part is intentionally small: this repo ships four helper nodes that cover SeedVR2-specific behavior the core nodes do not expose.

## Contents

- [What This Project Is](#what-this-project-is)
- [What This Project Is Not](#what-this-project-is-not)
- [Features](#features)
- [Current Scope](#current-scope)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model File Layout](#model-file-layout)
- [Supported Formats and Non-Supported Formats](#supported-formats-and-non-supported-formats)
- [Native Workflow Overview](#native-workflow-overview)
- [Quick Start](#quick-start)
- [Recommended Starting Settings](#recommended-starting-settings)
- [Custom Node Reference](#custom-node-reference)
- [Architecture Notes](#architecture-notes)
- [Limitations](#limitations)
- [Performance and VRAM Tips](#performance-and-vram-tips)
- [Troubleshooting](#troubleshooting)
- [Repository Layout](#repository-layout)
- [Credits](#credits)

## What This Project Is

- A from-scratch native port of the SeedVR2 NaDiT diffusion transformer and causal video VAE
- A loader/detection bridge so SeedVR2 checkpoints appear in native ComfyUI loaders
- A native graph workflow built around core ComfyUI nodes
- A small set of helper nodes for conditioning, VAE chunk tuning, and post-upscale color correction

## What This Project Is Not

- Not the original `seedvr2_videoupscaler` wrapper runtime
- Not an all-in-one "video upscaler" node
- Not a text-prompted model with a CLIP or T5 encoder
- Not a GGUF loader

If you want a native graph that feels like standard ComfyUI instead of a custom execution island, this is what the repo is for.

## Features

- Native `Load Diffusion Model` integration for SeedVR2 DiT checkpoints
- Native `Load VAE` integration for the published SeedVR2 VAE checkpoint
- Support for 3B and 7B DiT variants, including published `sharp` variants
- Support for safetensors checkpoints in plain fp16/fp32/bf16 and quantized formats ComfyUI can load through `comfy-kitchen`
- Fixed positive/negative SeedVR2 quality-anchor conditioning via a dedicated node
- SeedVR2-specific DiT preparation for native `KSampler`
- SeedVR2-specific VAE temporal chunk-size tuning for native `VAE Encode` / `VAE Decode`
- Optional per-model attention backend override for the DiT helper node
- Post-decode color correction node with multiple methods
- Automatic first-run download for the tiny `pos_emb.pt` / `neg_emb.pt` conditioning assets

## Current Scope

This repo intentionally keeps most of the workflow native.

Core ComfyUI nodes do the main work:

- `Load Diffusion Model`
- `Load VAE`
- `VAE Encode`
- `VAE Decode`
- `KSampler`
- `Image Scale`
- Optional native tiled VAE nodes
- Optional core `TorchCompileModel`

This repo adds only four custom nodes:

- `SeedVR2 DiT Settings`
- `SeedVR2 VAE Settings`
- `SeedVR2 Text Conditioning`
- `Lumina NIVR2 Color Correction`

## Requirements

- A recent ComfyUI build with native `Load Diffusion Model` / `Load VAE` support and `comfy_api.latest`
- A ComfyUI environment with:
  - `torch`
  - `einops`
  - `opencv-python`
  - `numpy`
- SeedVR2 safetensors checkpoints placed in the expected model folders

This port is designed around modern ComfyUI builds that already bundle the native infrastructure it relies on, including `comfy-kitchen` and `comfy-aimdo`.

## Installation

### Option 1: ComfyUI Manager (To be soon available)

Install the repo through ComfyUI Manager, then restart ComfyUI.

### Option 2: Manual install

Clone into your `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Doudoulix/Lumina_NIVR2.git
```

Install dependencies into the same Python environment ComfyUI uses:

```bash
python -m pip install einops opencv-python numpy
```

Then restart ComfyUI.

## Model File Layout

On import, the extension registers these folders with ComfyUI:

- `ComfyUI/models/nivr2_dit/`
- `ComfyUI/models/nivr2_vae/`

That means the native loader dropdowns will discover SeedVR2 files from those locations automatically.

### Conditioning assets

The fixed text-conditioning tensors are:

- `assets/pos_emb.pt`
- `assets/neg_emb.pt`

If they are missing, the node attempts to download them automatically on first use. If your ComfyUI environment should not download files at runtime, place them in `assets/` manually before launching.

## Supported Formats and Non-Supported Formats

Supported:

- Safetensors DiT checkpoints
- Native ComfyUI-compatible quantization metadata detected from checkpoint weights
- SeedVR2 VAE checkpoint in safetensors format

Not supported:

- GGUF DiT checkpoints such as `Q4_K_M` or `Q8_0`
- User text prompts

## Native Workflow Overview

The intended graph looks like this:

```text
Load Diffusion Model -> SeedVR2 DiT Settings -> KSampler -> VAE Decode -> Color Correction (optional)
Load VAE -----------> SeedVR2 VAE Settings ---> VAE Encode / VAE Decode
Image Scale --------> VAE Encode -------------> SeedVR2 DiT Settings (condition_latent)
SeedVR2 Text Conditioning -------------------> SeedVR2 DiT Settings / KSampler
```

The important difference from typical latent upscalers is that the low-resolution source is resized to the target output resolution before VAE encode. The DiT then treats that encoded latent as the super-resolution condition.

## Quick Start

### Image upscaling

1. Use native `Load Diffusion Model` and select a SeedVR2 DiT checkpoint from `nivr2_dit`.
2. Use native `Load VAE` and select `ema_vae_fp16.safetensors` from `nivr2_vae`.
3. Optionally pass the VAE through `SeedVR2 VAE Settings`.
4. Resize the source image to the final output resolution with native `Image Scale`.
5. Encode that resized image with native `VAE Encode`.
6. Use `SeedVR2 Text Conditioning` to get the fixed positive/negative conditioning.
7. Pass the native model, encoded condition latent, and conditioning into `SeedVR2 DiT Settings`.
8. Feed the patched model, patched conditioning, and emitted latent into native `KSampler`.
9. Decode the KSampler output with native `VAE Decode`.
10. Optionally apply `Lumina NIVR2 Color Correction`, using the resized original image as the reference.

### Video upscaling

The same graph applies to video clips represented as image batches / temporal tensors in ComfyUI.

Important constraint:

- The DiT processes one clip per call. It uses the temporal dimension for frames, not sample batching across multiple independent clips.

## Recommended Starting Settings

These are the safest defaults for the published one-step SeedVR2 release:

- `KSampler sampler_name`: `euler`
- `KSampler scheduler`: `simple`
- `KSampler steps`: `1`
- `KSampler cfg`: `1.0`
- `KSampler denoise`: `1.0`
- `SeedVR2 DiT Settings shift`: `1.0`
- `SeedVR2 DiT Settings latent_noise_scale`: `0.0`
- `SeedVR2 DiT Settings attention_backend`: `none`
- `SeedVR2 VAE Settings chunk_size`: `4`
- `Lumina NIVR2 Color Correction method`: `lab`

Good reasons to deviate:

- Increase `chunk_size` if the VAE is stable and you want better throughput
- Use 3B or fp8/int8 checkpoints when VRAM is tight
- Try `sageattn` or `flash attention 2` only if your ComfyUI install already supports them reliably

## Custom Node Reference

### SeedVR2 DiT Settings

Prepares a native SeedVR2 `MODEL` for native `KSampler`.

What it does:

- Installs SeedVR2's rectified-flow sampling behavior on the model
- Attaches the super-resolution condition latent through native concat conditioning
- Optionally perturbs the condition latent with SeedVR2-style latent noise augmentation
- Optionally overrides the attention backend for this model only

Inputs:

| Input | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | `MODEL` | - | Native `Load Diffusion Model` output |
| `condition_latent` | `LATENT` | - | Produced by native `VAE Encode` from the resized source |
| `positive` | `CONDITIONING` | - | From `SeedVR2 Text Conditioning` |
| `negative` | `CONDITIONING` | - | From `SeedVR2 Text Conditioning` |
| `shift` | `FLOAT` | `1.0` | Matches the original one-step recipe |
| `latent_noise_scale` | `FLOAT` | `0.0` | SeedVR2-style condition-latent augmentation |
| `attention_backend` | `COMBO` | `none` | `none`, `sageattn (2 and under)`, `sageattn 3`, `flash attention 2` |

Outputs:

- Patched `MODEL`
- Patched `positive` conditioning
- Patched `negative` conditioning
- Empty starting `LATENT` for `KSampler latent_image`

### SeedVR2 VAE Settings

Applies SeedVR2-specific temporal streaming chunk configuration to the VAE before native encode/decode.

Inputs:

| Input | Type | Default | Notes |
| --- | --- | --- | --- |
| `vae` | `VAE` | - | Native `Load VAE` output |
| `chunk_size` | `INT` | `4` | Must remain a multiple of 4 |

Behavior:

- Larger chunks: faster, more VRAM
- Smaller chunks: slower, less VRAM

This is separate from native spatial tiling. Native tiled VAE nodes still handle spatial tiling; this node controls temporal streaming behavior inside the causal VAE.

### SeedVR2 Text Conditioning

Loads SeedVR2's fixed positive and negative quality-anchor embeddings.

Notes:

- There is no user text prompt encoder in this repo
- The node behaves like a fixed-conditioning source
- Output wiring is similar to a text encoder node, but the embeddings are constant

Outputs:

- `positive`
- `negative`

### Lumina NIVR2 Color Correction

Matches decoded upscaled output back to a reference image or clip.

Recommended use:

- `content`: VAE-decoded upscaled result
- `style`: original input resized to the final output resolution

Methods:

| Method | Summary |
| --- | --- |
| `lab` | Perceptual LAB-space correction; recommended default |
| `wavelet` | Frequency-aware transfer that preserves fine details |
| `wavelet_adaptive` | Hybrid wavelet + saturation-aware correction |
| `hsv` | Hue-conditional saturation matching |
| `adain` | Statistical AdaIN-style transfer |

## Architecture Notes

This port keeps the model implementation native to ComfyUI conventions:

- SeedVR2 DiT is registered as a native flow model
- SeedVR2 VAE is registered through native `VAE` loading hooks
- The DiT uses native concat conditioning for its 33-channel input:
  - 16 noisy latent channels
  - 16 condition latent channels
  - 1 task-mask channel
- The VAE remains causal and chunk-streamed along the temporal axis

## Limitations

- Only one clip per KSampler call; the model is not designed for batching multiple independent clips together
- No GGUF support
- No custom prompt text encoder
- Model checkpoints are not auto-downloaded
- Attention overrides depend on your existing ComfyUI backend support

## Performance and VRAM Tips

- Start with the 3B checkpoint if you are unsure about memory headroom
- Use fp8 checkpoints when quality is acceptable and VRAM is the bottleneck
- Keep `attention_backend = none` until the rest of the pipeline is stable
- Lower `SeedVR2 VAE Settings chunk_size` to reduce temporal VAE working-set size
- Use native tiled VAE encode/decode for large spatial resolutions
- Compile only after the basic graph is working; native `TorchCompileModel` can help, but it is not required
- For color correction, `lab` is usually the best quality/simplicity tradeoff

## Troubleshooting

### SeedVR2 checkpoints do not appear in the native loader

Check all of the following:

- The repo is installed under `custom_nodes`
- ComfyUI was restarted after installation
- Files are in `models/nivr2_dit/` and `models/nivr2_vae/` or in native comfyUI folders `models/diffusion_models` and `models/vae`
- You are using safetensors checkpoints

### `SeedVR2 attention backend '...' is not available`

Set `attention_backend` back to `none` first. That keeps ComfyUI's own default attention selection and avoids requiring an explicit backend override.

### Flash Attention warnings or fallback behavior

Leave `attention_backend` at `none` unless you already know your ComfyUI + PyTorch + backend stack is stable. If you want an override, `sageattn` is often the safer first experiment.

### Out of memory during encode/decode

Try:

- `SeedVR2 VAE Settings chunk_size = 4`
- Native tiled VAE encode/decode
- Lower clip length
- Lower resolution
- 3B checkpoint instead of 7B
- int8 checkpoint instead of fp16

### The output colors look wrong or too shifted

Use `Lumina NIVR2 Color Correction` after `VAE Decode`, starting with:

- `method = lab`
- `style = original input resized to output resolution`

### Conditioning asset download fails

Place these files manually into `assets/`:

- `pos_emb.pt`
- `neg_emb.pt`

## Repository Layout

```text
Lumina_NIVR2/
|- __init__.py              # ComfyUI registration, model detection, VAE detection
|- model/                   # Native NaDiT + VAE port
|- nodes/                   # 4 shipped custom nodes
|- postprocess/             # Color-correction helpers
|- assets/                  # Fixed conditioning tensors
|- pyproject.toml
`- README.md
```

## Credits

- ByteDance Seed team for SeedVR2
- The original `seedvr2_videoupscaler` comfyUI custom implementation for the reference
- `numz/SeedVR2_comfyUI` and `AInVFX/SeedVR2_comfyUI` serving as backups for distribution of the fixed conditionnings assets

## Status

This repo is focused on a native ComfyUI workflow. If you want a smaller, more composable graph with native loaders and native sampler wiring, that is the design target here.
