"""
Image generation with SDXL + Lanczos upscaling.

Uses ``model_cpu_offload()`` to keep peak VRAM at ~4.5 GB so it fits
a 6 GB RTX 4050.  Generates at SDXL's native 1024×576 (16:9), then
upscales 2× via Pillow Lanczos for sharper Ken Burns panning.

Falls back to SD 1.5 if ``SD_MODEL`` in config points to a non-XL model.
"""
import gc
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.config import (
    SD_MODEL, IMG_W, IMG_H, STEPS, GUIDANCE,
    NEGATIVE_BASE, USE_CPU_OFFLOAD,
    UPSCALE_IMAGES, UPSCALE_FACTOR,
)

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_prompt(raw_prompt: str, genre: dict[str, Any]) -> str:
    """Prefix the raw image prompt with the genre's art style."""
    style = genre.get("image_style", "")
    mascot_name = genre.get("mascot_name")
    mascot_look = genre.get("mascot_look")
    if mascot_name and mascot_look:
        raw_prompt = raw_prompt.replace(mascot_name, mascot_look)
    parts = [p for p in (raw_prompt.strip(), style.strip()) if p]
    return ", ".join(parts)


def _negative(genre: dict[str, Any]) -> str:
    extra = genre.get("negative_prompt", "")
    parts = [p for p in (NEGATIVE_BASE, extra) if p]
    return ", ".join(parts)


def _upscale_image(path: Path, factor: int) -> Path:
    """Lanczos upscale — CPU only, zero VRAM."""
    img = Image.open(path)
    new_size = (img.width * factor, img.height * factor)
    upscaled = img.resize(new_size, Image.LANCZOS)
    out_path = path.with_name(path.stem + "_up" + path.suffix)
    upscaled.save(out_path)
    return out_path


# ── public API ──────────────────────────────────────────────────────────

def generate_images(
    scenes: list[dict],
    genre: dict[str, Any],
    outdir: Path,
    base_seed: int = 1234,
    img_w: int | None = None,
    img_h: int | None = None,
) -> list[Path]:
    """Generate one image per scene. Returns a list of saved PNG paths.

    ``img_w`` / ``img_h`` override the config defaults so callers can
    request vertical (9:16) images for YouTube Shorts.
    If ``UPSCALE_IMAGES`` is True, each image is upscaled after generation
    (SDXL unloaded first, so upscale uses zero VRAM).
    """
    import torch

    # Resolve dimensions — caller overrides take priority over config
    width  = img_w or IMG_W
    height = img_h or IMG_H

    if not torch.cuda.is_available():
        sys.exit(
            "CUDA GPU not found by PyTorch. "
            "Install the CUDA build of torch (see docs/setup_guide.md)."
        )

    _ts(f"Loading image model: {SD_MODEL} ...")

    is_sdxl = "xl" in SD_MODEL.lower()

    if is_sdxl:
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None,   # saves ~300 MB VRAM
        )

    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if USE_CPU_OFFLOAD:
        # enable_vae_slicing must be called BEFORE cpu offload on SDXL
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

    neg = _negative(genre)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre)
        path = img_dir / f"scene_{i:02d}.png"

        for retry in range(3):
            gen = torch.Generator("cuda").manual_seed(base_seed + i + retry * 1000)
            out = pipe(
                prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen,
            )
            flagged = getattr(out, "nsfw_content_detected", None)
            if not (flagged and flagged[0]):
                break
            _ts(f"  scene {i}: flagged by safety checker, reseeding")

        out.images[0].save(path)
        raw_paths.append(path)
        _ts(f"Image {i + 1}/{len(scenes)} done")

    # Free VRAM completely before upscaling on CPU
    del pipe
    free_gpu()

    # Upscale (CPU-only, uses Pillow — no VRAM)
    if UPSCALE_IMAGES and UPSCALE_FACTOR > 1:
        _ts(f"Upscaling images {UPSCALE_FACTOR}x (Lanczos, CPU)...")
        final_paths = []
        for i, raw in enumerate(raw_paths):
            try:
                up = _upscale_image(raw, UPSCALE_FACTOR)
                final_paths.append(up)
            except Exception as exc:
                _ts(f"  Upscale failed for scene {i} ({exc}), using original")
                final_paths.append(raw)
            _ts(f"  Upscaled {i + 1}/{len(raw_paths)}")
        return final_paths

    return raw_paths
