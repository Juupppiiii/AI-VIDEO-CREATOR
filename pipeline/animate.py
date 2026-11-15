"""
Animated clip generation using AnimateDiff-Lightning on SD 1.5.

Each scene prompt produces a 16-frame MP4 clip (~0.64 s at 25 fps).
The clips are stitched together by the assembler — Ken Burns is skipped
because the clips already contain motion.

VRAM budget (RTX 4050 6 GB):
  SD 1.5 UNet      ~2.0 GB fp16  (sequential offload)
  Motion adapter   ~0.3 GB
  VAE decode       ~0.5 GB
  ─────────────────────────────
  Peak             ~2.8 GB  ✓

First run downloads ~180 MB (Lightning adapter checkpoint).
SD 1.5 base is already cached from image generation.
"""
import gc
import queue
import threading
import time
from pathlib import Path
from typing import Any

from pipeline.config import (
    ANIMATEDIFF_STEPS, ANIMATEDIFF_FRAMES,
    ANIMATEDIFF_BASE, ANIMATEDIFF_REPO, ANIMATEDIFF_CKPT,
    FPS, NEGATIVE_BASE,
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
    """Build image prompt — prepend trigger, append style, cap at 70 words."""
    style   = genre.get("image_style", "").strip()
    trigger = genre.get("lora_trigger", "").strip()
    mascot  = genre.get("mascot_name")
    look    = genre.get("mascot_look")
    if mascot and look:
        raw = raw.replace(mascot, look)
    parts = [p for p in (trigger, raw.strip(), style) if p]
    combined = ", ".join(parts)
    words = combined.split()
    if len(words) > 70:
        combined = " ".join(words[:70])
    return combined


def _negative(genre: dict[str, Any]) -> str:
    extra = genre.get("negative_prompt", "")
    parts = [p for p in (NEGATIVE_BASE, extra) if p]
    return ", ".join(parts)


def _load_pipe_threaded(result_box: dict, error_box: dict):
    """
    Load the AnimateDiff pipeline in a background thread so the caller
    can send heartbeat messages to Gradio while waiting.
    result_box["pipe"] is set when done; error_box["err"] on failure.
    """
    try:
        import torch
        from diffusers import (
            AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # 1. Download / load motion adapter to CPU
        ckpt_path = hf_hub_download(ANIMATEDIFF_REPO, ANIMATEDIFF_CKPT)
        adapter = MotionAdapter()
        adapter.load_state_dict(load_file(ckpt_path, device="cpu"))

        # 2. Load SD 1.5 base + adapter (CPU, disable_mmap avoids page-file OOM)
        pipe = AnimateDiffPipeline.from_pretrained(
            ANIMATEDIFF_BASE,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
            disable_mmap=True,
        )

        # 3. Lightning scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )

        # 4. Sequential CPU offload — each sub-model moves to GPU only
        #    when needed, keeping peak VRAM ~2.8 GB and peak RAM ~3 GB.
        pipe.enable_sequential_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        result_box["pipe"] = pipe

    except Exception as exc:
        import traceback
        error_box["err"] = traceback.format_exc()


# ── public API ──────────────────────────────────────────────────────────

def generate_animated_clips(
    scenes:    list[dict],
    genre:     dict[str, Any],
    outdir:    Path,
    base_seed: int = 1234,
    img_w:     int = 512,
    img_h:     int = 512,
) -> list[Path]:
    """
    Generate one animated MP4 clip per scene using AnimateDiff-Lightning.

    Model loading runs in a side thread and sends periodic heartbeat log
    messages so Gradio's WebSocket stays alive during the long load.

    Returns a list of MP4 paths (no audio — assembler adds narration).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. Install the CUDA build of torch."
        )

    # Clamp to 512 max — AnimateDiff + SD 1.5 is stable up to 512×512 on 6 GB
    safe_w = min(img_w, 512)
    safe_h = min(img_h, 512)
    if safe_w != img_w or safe_h != img_h:
        _ts(f"AnimateDiff: clamped {img_w}x{img_h} → {safe_w}x{safe_h} (VRAM limit)")

    # ── Load model in background thread, heartbeat from main thread ────
    result_box: dict = {}
    error_box:  dict = {}

    loader = threading.Thread(
        target=_load_pipe_threaded,
        args=(result_box, error_box),
        daemon=True,
    )

    _ts("Loading AnimateDiff-Lightning (this takes ~60s, please wait)...")
    loader.start()

    # Send a heartbeat every 5 seconds so the WebSocket stays alive
    t0 = time.time()
    while loader.is_alive():
        time.sleep(5)
        elapsed = int(time.time() - t0)
        _ts(f"  Still loading model... ({elapsed}s elapsed, do not close the browser)")

    loader.join()

    if "err" in error_box:
        raise RuntimeError(f"AnimateDiff model load failed:\n{error_box['err']}")

    pipe = result_box["pipe"]
    _ts("Model loaded. Generating animated clips...")

    # ── Generation loop ────────────────────────────────────────────────
    clips_dir = outdir / "anim_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    from diffusers.utils import export_to_video

    neg   = _negative(genre)
    paths: list[Path] = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre)
        out    = clips_dir / f"scene_{i:02d}.mp4"

        generator = torch.Generator(device="cpu").manual_seed(base_seed + i)
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            num_frames=ANIMATEDIFF_FRAMES,
            num_inference_steps=ANIMATEDIFF_STEPS,
            guidance_scale=1.0,   # Lightning 4-step works best at cfg=1
            width=safe_w,
            height=safe_h,
            generator=generator,
        )

        export_to_video(result.frames[0], str(out), fps=FPS)
        paths.append(out)
        _ts(f"Animated clip {i + 1}/{len(scenes)} done")

    del pipe
    _free_gpu()
    return paths
