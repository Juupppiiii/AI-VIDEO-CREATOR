"""
Sound effects system.

Generates basic SFX programmatically using numpy (no external files needed).
Builds a single timeline-aligned SFX WAV track that the assembler mixes in.

Users can drop better WAV/MP3 files into ``sfx/`` to override the generated
defaults in ``data/sfx/``.
"""
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import SFX_DIR, DATA_DIR, SFX_VOLUME, SCENE_PAD

SAMPLE_RATE = 44100
_GENERATED_DIR = DATA_DIR / "sfx"

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Individual SFX generators (numpy only, no scipy) ───────────────────

def _envelope(length: int, attack: float = 0.05, release: float = 0.3) -> np.ndarray:
    """Simple attack-release envelope."""
    env = np.ones(length, dtype=np.float32)
    att = int(attack * SAMPLE_RATE)
    rel = int(release * SAMPLE_RATE)
    if att > 0 and att < length:
        env[:att] = np.linspace(0, 1, att)
    if rel > 0 and rel < length:
        env[-rel:] = np.linspace(1, 0, rel)
    return env


def _gen_whoosh(duration: float = 0.4) -> np.ndarray:
    """Quick whoosh/swish for transitions."""
    n = int(SAMPLE_RATE * duration)
    noise = np.random.randn(n).astype(np.float32)
    # Exponential decay envelope with quick attack
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 0.5  # bell-shaped
    return noise * env * 0.4


def _gen_impact(duration: float = 0.5) -> np.ndarray:
    """Short punchy impact for dramatic reveals."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    # Low thud + high transient
    low = np.sin(2 * np.pi * 60 * t) * np.exp(-5 * t)
    high = np.sin(2 * np.pi * 400 * t) * np.exp(-20 * t)
    noise = np.random.randn(n).astype(np.float32) * np.exp(-15 * t) * 0.3
    return (low * 0.6 + high * 0.3 + noise).astype(np.float32)


def _gen_rumble(duration: float = 3.0) -> np.ndarray:
    """Low frequency rumble for tension/suspense."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    env = _envelope(n, attack=0.5, release=0.8)
    # Layered low sines with slow modulation
    sig = (
        np.sin(2 * np.pi * 35 * t) * 0.4
        + np.sin(2 * np.pi * 55 * t) * 0.3
        + np.sin(2 * np.pi * 25 * t + np.sin(2 * np.pi * 0.5 * t) * 2) * 0.3
    )
    return (sig * env * 0.35).astype(np.float32)


def _gen_heartbeat(duration: float = 2.0) -> np.ndarray:
    """Lub-dub heartbeat pattern for drama/horror."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    beat_dur = int(SAMPLE_RATE * 0.12)
    gap = int(SAMPLE_RATE * 0.08)

    def _one_beat(freq=60):
        t = np.linspace(0, beat_dur / SAMPLE_RATE, beat_dur, dtype=np.float32)
        return np.sin(2 * np.pi * freq * t) * np.exp(-15 * t) * 0.5

    # Two lub-dubs in the duration
    bpm_interval = int(SAMPLE_RATE * 0.7)  # ~85 bpm
    for start in range(0, n - bpm_interval, bpm_interval):
        lub = _one_beat(55)
        dub = _one_beat(45)
        end1 = min(start + beat_dur, n)
        out[start:end1] += lub[: end1 - start]
        start2 = start + beat_dur + gap
        end2 = min(start2 + beat_dur, n)
        if start2 < n:
            out[start2:end2] += dub[: end2 - start2]

    return out


def _gen_chime(duration: float = 0.8) -> np.ndarray:
    """Bright ascending chime for positive/kids moments."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    freqs = [523, 659, 784]  # C5, E5, G5 — major chord
    spacing = int(SAMPLE_RATE * 0.15)
    tone_dur = int(SAMPLE_RATE * 0.4)
    for i, freq in enumerate(freqs):
        start = i * spacing
        end = min(start + tone_dur, n)
        t = np.linspace(0, (end - start) / SAMPLE_RATE, end - start, dtype=np.float32)
        tone = np.sin(2 * np.pi * freq * t) * np.exp(-4 * t) * 0.3
        out[start:end] += tone
    return out


def _gen_drone(duration: float = 4.0) -> np.ndarray:
    """Slow evolving drone for mystery/eerie scenes."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    env = _envelope(n, attack=1.0, release=1.0)
    sig = (
        np.sin(2 * np.pi * 80 * t + np.sin(2 * np.pi * 0.3 * t) * 3) * 0.3
        + np.sin(2 * np.pi * 120 * t + np.sin(2 * np.pi * 0.2 * t) * 2) * 0.2
        + np.sin(2 * np.pi * 160 * t) * 0.1
    )
    return (sig * env * 0.3).astype(np.float32)


_GENERATORS = {
    "whoosh": _gen_whoosh,
    "impact": _gen_impact,
    "rumble": _gen_rumble,
    "heartbeat": _gen_heartbeat,
    "chime": _gen_chime,
    "drone": _gen_drone,
}


# ── Mood → SFX mapping ─────────────────────────────────────────────────

MOOD_SFX: dict[str, str | None] = {
    "tense": "rumble",
    "dread": "heartbeat",
    "eerie": "drone",
    "terrifying": "heartbeat",
    "unsettling": "drone",
    "hopeless": "rumble",
    "shocking": "impact",
    "revealing": "impact",
    "triumphant": "impact",
    "emotional": None,
    "heartbreaking": None,
    "heavy": "rumble",
    "struggling": None,
    "hopeful": "chime",
    "determined": None,
    "peaceful": None,
    "warm": "chime",
    "cheerful": "chime",
    "curious": None,
    "excited": "whoosh",
    "suspicious": "drone",
    "intriguing": "drone",
    "mysterious": "drone",
    "satisfying": "chime",
    "impressed": "chime",
    "suspenseful": "rumble",
    "fearful": "heartbeat",
    "neutral": None,
}


# ── File management ─────────────────────────────────────────────────────

def ensure_defaults():
    """Generate default SFX WAV files if they don't exist."""
    import soundfile as sf

    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for name, gen_fn in _GENERATORS.items():
        path = _GENERATED_DIR / f"{name}.wav"
        if not path.exists():
            wav = gen_fn()
            sf.write(path, wav, SAMPLE_RATE)

    SFX_DIR.mkdir(parents=True, exist_ok=True)


def get_sfx_path(sfx_type: str) -> Path | None:
    """Find an SFX file: user-provided (sfx/) overrides generated (data/sfx/)."""
    # User override
    for ext in (".wav", ".mp3"):
        user_path = SFX_DIR / f"{sfx_type}{ext}"
        if user_path.exists():
            return user_path
    # Generated default
    gen_path = _GENERATED_DIR / f"{sfx_type}.wav"
    if gen_path.exists():
        return gen_path
    return None


# ── Timeline builder ────────────────────────────────────────────────────

def build_sfx_track(
    scenes: list[dict],
    durations: list[float],
    transition_dur: float,
    total_video_duration: float,
    time_offset: float = 0.0,
) -> Path | None:
    """Build a single WAV file with SFX placed at each scene's start.

    *time_offset* shifts all SFX forward (e.g. to account for an intro card).
    Returns the path to the combined SFX track, or None if no SFX apply.
    """
    import soundfile as sf

    ensure_defaults()

    n_samples = int(total_video_duration * SAMPLE_RATE) + SAMPLE_RATE  # +1s buffer
    track = np.zeros(n_samples, dtype=np.float32)
    any_sfx = False

    # Calculate scene start times on the final timeline
    t0 = time_offset
    for i, (scene, dur) in enumerate(zip(scenes, durations)):
        mood = scene.get("mood", "neutral")
        sfx_type = MOOD_SFX.get(mood)

        if sfx_type:
            sfx_path = get_sfx_path(sfx_type)
            if sfx_path:
                data, sr = sf.read(sfx_path, dtype="float32")

                # Convert stereo to mono if needed
                if data.ndim == 2:
                    data = data.mean(axis=1)

                if sr != SAMPLE_RATE:
                    # Simple resample by linear interpolation
                    indices = np.linspace(0, len(data) - 1,
                                          int(len(data) * SAMPLE_RATE / sr))
                    data = np.interp(indices, np.arange(len(data)), data)

                start_sample = int(t0 * SAMPLE_RATE)
                end_sample = min(start_sample + len(data), n_samples)
                length = end_sample - start_sample
                if length > 0 and start_sample >= 0:
                    track[start_sample:end_sample] += data[:length] * SFX_VOLUME
                    any_sfx = True

        # Advance timeline using config SCENE_PAD, not a hardcoded value
        clip_dur = dur + SCENE_PAD
        if i < len(durations) - 1:
            t0 += clip_dur - transition_dur
        else:
            t0 += clip_dur

    if not any_sfx:
        return None

    # Normalize to prevent clipping
    peak = np.abs(track).max()
    if peak > 0.95:
        track = track * (0.95 / peak)

    out_path = _GENERATED_DIR / "_combined_sfx.wav"
    sf.write(out_path, track, SAMPLE_RATE)
    _ts(f"SFX track generated ({total_video_duration:.1f}s)")
    return out_path
