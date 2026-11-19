"""
Animated clip generation using AnimateDiff-Lightning on SD 1.5.

Produces 16-frame looping MP4 clips per scene (~0.64 s at 25 fps).
Assembler loops them to match narration duration.

Key memory fixes for RTX 4050 6 GB / 7.7 GB RAM:
  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — eliminates the
     "tried to allocate 864 MiB" fragmentation OOM in VAE decode.
  2. enable_vae_tiling() — decodes VAE output in tiles instead of one
     giant tensor, keeps peak VRAM under 2 GB during decode.
  3. enable_sequential_cpu_offload() — sub-models move to GPU one at a
     time; peak VRAM ~2 GB, peak RAM ~3 GB.
  4. Model loaded in a background thread so the Gradio WebSocket gets
     heartbeat messages and never drops during the ~2-min load.

First run downloads ~180 MB (Lightning adapter).
SD 1.5 base is already cached from image generation.
"""
import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

# ── CRITICAL: set before torch is imported anywhere in this process ───
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

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
            torch.cuda.synchronize()
    except Exception:
        pass


def _build_prompt(raw: str, genre: dict[str, Any]) -> str:
    """Prepend trigger word + style; cap at 70 words (AnimateDiff limit)."""
    style   = genre.get("image_style", "").strip()
    trigger = genre.get("lora_trigger", "").strip()
    mascot  = genre.get("mascot_name")
    look    = genre.get("mascot_look")
    if mascot and look:
        raw = raw.replace(mascot, look)
    parts    = [p for p in (trigger, raw.strip(), style) if p]
    combined = ", ".join(parts)
    words    = combined.split()
    return " ".join(words[:70]) if len(words) > 70 else combined


def _negative(genre: dict[str, Any]) -> str:
    extra = genre.get("negative_prompt", "")
    return ", ".join(p for p in (NEGATIVE_BASE, extra) if p)


# ── Background model loader ────────────────────────────────────────────

def _load_pipe(result: dict, error: dict) -> None:
    """
    Load AnimateDiff-Lightning pipeline onto CPU first, then apply
    sequential offload.  Runs in a daemon thread so the caller can
    send heartbeat messages while waiting.
    """
    try:
        import torch
        from diffusers import (
            AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # 1. Motion adapter — load to CPU to avoid pagefile OOM
        ckpt = hf_hub_download(ANIMATEDIFF_REPO, ANIMATEDIFF_CKPT)
        adapter = MotionAdapter()
        adapter.load_state_dict(load_file(ckpt, device="cpu"))

        # 2. Base pipeline on CPU (disable_mmap avoids Windows page-file OOM)
        pipe = AnimateDiffPipeline.from_pretrained(
            ANIMATEDIFF_BASE,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            safety_checker=None,
            disable_mmap=True,
        )

        # 3. Lightning scheduler (required for 4-step distilled model)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            beta_schedule="linear",
        )

        # 4. Memory optimisations — ORDER MATTERS
        #    Some methods may not exist in all diffusers versions
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        pipe.enable_sequential_cpu_offload()  # ~2 GB peak VRAM

        result["pipe"] = pipe

    except Exception:
        import traceback
        error["tb"] = traceback.format_exc()


# ── Public API ─────────────────────────────────────────────────────────

def generate_animated_clips(
    scenes:    list[dict],
    genre:     dict[str, Any],
    outdir:    Path,
    base_seed: int  = 1234,
    img_w:     int  = 512,
    img_h:     int  = 512,
) -> list[Path]:
    """
    Generate one 16-frame animated MP4 per scene using AnimateDiff-Lightning.

    Returns MP4 paths (no audio — assembler muxes narration in).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Install the CUDA build of torch.")

    # Clamp — AnimateDiff + SD 1.5 is VRAM-stable up to 512×512 on 6 GB
    w = min(img_w, 512)
    h = min(img_h, 512)
    if w != img_w or h != img_h:
        _ts(f"AnimateDiff: clamped {img_w}×{img_h} → {w}×{h} (VRAM budget)")

    # ── Load pipeline in background thread ────────────────────────────
    result: dict = {}
    error:  dict = {}
    t = threading.Thread(target=_load_pipe, args=(result, error), daemon=True)
    _ts("Loading AnimateDiff-Lightning… (first time ~2 min, please wait)")
    t.start()

    t0 = time.time()
    while t.is_alive():
        time.sleep(5)
        _ts(f"  Loading model… {int(time.time()-t0)}s elapsed")
    t.join()

    if "tb" in error:
        raise RuntimeError(f"AnimateDiff failed to load:\n{error['tb']}")

    pipe = result["pipe"]
    _ts("Model ready — generating clips…")

    # ── Generation ────────────────────────────────────────────────────
    from diffusers.utils import export_to_video

    clips_dir = outdir / "anim_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    neg   = _negative(genre)
    paths: list[Path] = []

    for i, scene in enumerate(scenes):
        prompt = _build_prompt(scene["image_prompt"], genre)
        out    = clips_dir / f"scene_{i:02d}.mp4"
        gen    = torch.Generator(device="cpu").manual_seed(base_seed + i)

        try:
            video = pipe(
                prompt            = prompt,
                negative_prompt   = neg,
                num_frames        = ANIMATEDIFF_FRAMES,
                num_inference_steps = ANIMATEDIFF_STEPS,
                guidance_scale    = 1.0,
                width             = w,
                height            = h,
                generator         = gen,
            ).frames[0]
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                _ts(f"  Scene {i}: OOM — clearing VRAM and retrying...")
                _free_gpu()
                gen = torch.Generator(device="cpu").manual_seed(base_seed + i + 500)
                video = pipe(
                    prompt=prompt, negative_prompt=neg,
                    num_frames=ANIMATEDIFF_FRAMES, num_inference_steps=ANIMATEDIFF_STEPS,
                    guidance_scale=1.0, width=min(w, 384), height=min(h, 384),
                    generator=gen,
                ).frames[0]
            else:
                raise

        export_to_video(video, str(out), fps=FPS)
        paths.append(out)
        _ts(f"Clip {i+1}/{len(scenes)} done")

    del pipe
    _free_gpu()
    return paths
