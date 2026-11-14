"""
Animated clip generation using AnimateDiff-Lightning on SD 1.5.

Each scene prompt produces a 16-frame MP4 clip (~0.64 s at 25 fps).
The clips are then stitched together by the assembler exactly like
still-image clips — but they already contain motion, so Ken Burns
is skipped.

VRAM budget (RTX 4050 5 GB):
  SD 1.5 UNet      ~2.0 GB fp16
  Motion adapter   ~0.3 GB
  VAE decode       ~0.5 GB
  Activations      ~1.0 GB
  ─────────────────────────────
  Peak             ~3.8 GB  ✓

First run downloads:
  AnimateDiff-Lightning 4-step checkpoint  ~180 MB
  (SD 1.5 base already cached from image generation)
"""
import gc
import time
from pathlib import Path
from typing import Any

from pipeline.config import (
    ANIMATEDIFF_STEPS, ANIMATEDIFF_FRAMES,
    ANIMATEDIFF_BASE, ANIMATEDIFF_REPO, ANIMATEDIFF_CKPT,
    FPS, GUIDANCE, NEGATIVE_BASE,
)

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_prompt(raw: str, genre: dict[str, Any]) -> str:
    style   = genre.get("image_style", "")
    mascot  = genre.get("mascot_name")
    look    = genre.get("mascot_look")
    if mascot and look:
        raw = raw.replace(mascot, look)
    # keep prompt under 75 tokens — AnimateDiff degrades above that
    combined = f"{raw.strip()}, {style.strip()}" if style else raw.strip()
    words = combined.split()
    if len(words) > 70:
        combined = " ".join(words[:70])
    return combined


def _negative(genre: dict[str, Any]) -> str:
    extra = genre.get("negative_prompt", "")
    parts = [p for p in (NEGATIVE_BASE, extra) if p]
    return ", ".join(parts)


# ── public API ──────────────────────────────────────────────────────────

def generate_animated_clips(
    scenes:    list[dict],
    genre:     dict[str, Any],
    outdir:    Path,
    base_seed: int = 1234,
    img_w:     int = 512,
    img_h:     int = 512,
) -> list[Path]:
    """Generate one animated MP4 clip per scene using AnimateDiff-Lightning.

    Returns a list of paths to the rendered MP4 files.
    The clips have no audio — the assembler adds narration later.
    """
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
    from diffusers.utils import export_to_video
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for AnimateDiff. "
            "Install the CUDA build of torch (see docs/setup_guide.md)."
        )

    # SD 1.5 + AnimateDiff work best at multiples of 8, max 512 on 5 GB
    # Clamp to safe sizes so we don't OOM
    safe_w = min(img_w, 512)
    safe_h = min(img_h, 512)
    if safe_w != img_w or safe_h != img_h:
        _ts(f"AnimateDiff: clamped size {img_w}x{img_h} → {safe_w}x{safe_h} (VRAM limit)")

    _ts("Loading AnimateDiff-Lightning motion adapter...")
    adapter = MotionAdapter().to("cuda", torch.float16)
    ckpt_path = hf_hub_download(ANIMATEDIFF_REPO, ANIMATEDIFF_CKPT)
    adapter.load_state_dict(load_file(ckpt_path, device="cuda"))

    _ts(f"Loading base model: {ANIMATEDIFF_BASE} ...")
    pipe = AnimateDiffPipeline.from_pretrained(
        ANIMATEDIFF_BASE,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    # Lightning scheduler — trailing timestep spacing required
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear",
    )
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    clips_dir = outdir / "anim_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    neg    = _negative(genre)
    paths: list[Path] = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre)
        out    = clips_dir / f"scene_{i:02d}.mp4"

        generator = torch.Generator("cuda").manual_seed(base_seed + i)
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_frames=ANIMATEDIFF_FRAMES,
            num_inference_steps=ANIMATEDIFF_STEPS,
            guidance_scale=1.0,          # Lightning works best at cfg=1 for 4-step
            width=safe_w,
            height=safe_h,
            generator=generator,
        )

        # export_to_video writes an MP4 at the given fps
        export_to_video(result.frames[0], str(out), fps=FPS)
        paths.append(out)
        _ts(f"Animated clip {i + 1}/{len(scenes)} done")

    del pipe, adapter
    _free_gpu()
    return paths
