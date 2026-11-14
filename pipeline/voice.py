"""
Voice generation with Kokoro TTS.

Supports multiple speakers per scene via the genre's ``voice_map``.
Each speaker is mapped to a Kokoro voice id.  Loads one voice pipeline
instance and switches voices between scenes to keep RAM low.
"""
import gc
import time
from pathlib import Path
from typing import Any

from pipeline.config import VOICE_MAP, DEFAULT_VOICE, DEFAULT_VOICE_SPEED

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


def _resolve_voice(speaker: str, genre: dict[str, Any]) -> str:
    """Return the Kokoro voice id for a speaker label."""
    # Genre-level override → global map → default
    genre_map = genre.get("voice_map", {})
    return genre_map.get(speaker) or VOICE_MAP.get(speaker) or DEFAULT_VOICE


# ── public API ──────────────────────────────────────────────────────────

def generate_voice(
    scenes: list[dict],
    genre: dict[str, Any],
    outdir: Path,
    speed: float | None = None,
) -> tuple[list[Path], list[float]]:
    """Synthesise narration for each scene.

    Returns ``(audio_paths, durations_in_seconds)``.
    """
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    speed = speed or genre.get("voice_speed", DEFAULT_VOICE_SPEED)

    _ts("Loading Kokoro TTS...")
    tts = KPipeline(lang_code="a")   # American English

    audio_dir = outdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    durations: list[float] = []

    for i, scene in enumerate(scenes):
        voice = _resolve_voice(scene.get("speaker", "narrator"), genre)
        text = scene["narration"]

        chunks = []
        for _, _, audio in tts(text, voice=voice, speed=speed):
            chunks.append(
                audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            )

        if not chunks:
            _ts(f"  scene {i}: TTS returned no audio, inserting 0.5s silence")
            chunks = [np.zeros(int(24000 * 0.5), dtype=np.float32)]

        wav = np.concatenate(chunks)
        path = audio_dir / f"scene_{i:02d}.wav"
        sf.write(path, wav, 24000)
        paths.append(path)
        durations.append(len(wav) / 24000)
        _ts(f"Voice {i + 1}/{len(scenes)} done ({durations[-1]:.1f}s, voice={voice})")

    del tts
    free_gpu()

    return paths, durations
