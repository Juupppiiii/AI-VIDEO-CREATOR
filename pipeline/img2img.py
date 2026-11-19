"""
Image-to-image generation — uses reference frames as style input.

Instead of generating from text alone (txt2img), this takes a reference
image and transforms it based on the prompt. The output inherits the
color palette, composition, and style from the reference.

The strength parameter controls how much it changes:
  0.2 = almost identical to reference
  0.5 = same style, different content
  0.8 = uses reference loosely
"""
import gc
import time
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.config import (
    SD_MODEL, STEPS, GUIDANCE, NEGATIVE_BASE, USE_CPU_OFFLOAD,
)
from pipeline.images import _build_prompt, _negative, _upscale_image, free_gpu

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def extract_reference_frames(video: Path, outdir: Path, count: int = 8) -> list[Path]:
    """Extract evenly-spaced frames from a reference video."""
    import subprocess

    ref_dir = outdir / "reference_frames"
    ref_dir.mkdir(parents=True, exist_ok=True)

    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    dur = float(p.stdout.strip())
    interval = dur / (count + 1)

    frames = []
    for i in range(1, count + 1):
        t = interval * i
        out = ref_dir / f"ref_{i:02d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video),
            "-vframes", "1", str(out),
        ], capture_output=True)
        if out.exists():
            frames.append(out)

    _ts(f"Extracted {len(frames)} reference frames")
    return frames


def generate_images_img2img(
    scenes: list[dict],
    genre: dict[str, Any],
    reference_frames: list[Path],
    outdir: Path,
    base_seed: int = 1234,
    img_w: int = 432,
    img_h: int = 768,
    strength: float = 0.55,
) -> list[Path]:
    """Generate images using img2img with reference frames as style input.

    Each scene uses the closest reference frame as the starting image.
    Strength controls how much SD changes from the reference.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found.")

    _ts(f"Loading SD img2img pipeline (strength={strength})...")

    from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16, safety_checker=None, disable_mmap=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if USE_CPU_OFFLOAD:
        if hasattr(pipe, "enable_vae_slicing"): pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"): pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    neg = _negative(genre)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre, scene=scene)
        path = img_dir / f"scene_{i:02d}.png"

        # Pick reference frame (cycle through available frames)
        ref_idx = i % len(reference_frames) if reference_frames else 0
        ref_img = Image.open(reference_frames[ref_idx]).convert("RGB")
        ref_img = ref_img.resize((img_w, img_h), Image.LANCZOS)

        gen = torch.Generator(device="cpu").manual_seed(base_seed + i)

        out = pipe(
            prompt=prompt,
            image=ref_img,
            negative_prompt=neg,
            strength=strength,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        )
        out.images[0].save(path)
        paths.append(path)
        _ts(f"Image {i + 1}/{len(scenes)} done (ref: {reference_frames[ref_idx].name})")

    del pipe
    free_gpu()
    return paths
