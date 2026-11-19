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


def _build_prompt(raw_prompt: str, genre: dict[str, Any], scene: dict | None = None) -> str:
    """Build the full image prompt for one scene.

    Structure (always in this order so SD 1.5 weights them correctly):
      1. STYLE block   — overall visual style (from genre["image_style"])
      2. CHARACTER block — consistent narrator/character description
         Sourced from: scene["continuity_character"] > genre["character_block"]
      3. SCENE block   — environment + pose + camera from the LLM (raw_prompt)
      4. LoRA trigger  — if a LoRA is loaded (genre["lora_trigger"])

    The style and character blocks are injected automatically every scene
    so the character stays visually consistent without relying on the LLM
    to repeat the description.
    """
    style     = genre.get("image_style", "").strip()
    trigger   = genre.get("lora_trigger", "").strip()

    # Character block: scene-level continuity override > genre-level default
    character = ""
    if scene and scene.get("continuity_character"):
        character = scene["continuity_character"].strip()
    if not character:
        character = genre.get("character_block", "").strip()

    # Mascot substitution (for other genres that use a named mascot)
    mascot_name = genre.get("mascot_name")
    mascot_look = genre.get("mascot_look")
    if mascot_name and mascot_look:
        raw_prompt = raw_prompt.replace(mascot_name, mascot_look)

    # Build ordered parts — skip empty strings
    parts = [p for p in (style, character, raw_prompt.strip(), trigger) if p]
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
    locked_indices: set[int] | None = None,
) -> list[Path]:
    """Generate one image per scene. Returns a list of saved PNG paths.

    ``locked_indices``: set of scene indices to SKIP — their existing
    images are reused. Enables selective regeneration of individual scenes.
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
            disable_mmap=True,     # avoids paging file OOM on Windows
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

    # ── Optional LoRA from genre template ──────────────────────────────
    # Set lora_hf_repo + lora_weight_name in a genre JSON to automatically
    # load and fuse a style LoRA (e.g. stickman, anime, watercolor).
    lora_repo   = genre.get("lora_hf_repo", "").strip()
    lora_weight = genre.get("lora_weight_name", "").strip()
    lora_scale  = float(genre.get("lora_scale", 0.85))
    if lora_repo and lora_weight:
        _ts(f"Loading style LoRA: {lora_repo}/{lora_weight} (scale={lora_scale})")
        try:
            pipe.load_lora_weights(lora_repo, weight_name=lora_weight)
            pipe.fuse_lora(lora_scale=lora_scale)
            _ts("LoRA fused successfully")
        except Exception as e:
            _ts(f"LoRA load failed ({e}) — continuing without LoRA")

    neg = _negative(genre)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre, scene=scene)
        path = img_dir / f"scene_{i:02d}.png"

        # Skip locked scenes — reuse existing image
        if locked_indices and i in locked_indices and path.exists():
            raw_paths.append(path)
            _ts(f"Image {i + 1}/{len(scenes)} locked — reusing existing")
            continue

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
