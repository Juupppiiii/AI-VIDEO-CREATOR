"""
Style cloning — analyze a reference video to extract editing patterns.

Inspired by AI Content Studio's style engine: uses FFmpeg scene detection
to measure average shot duration, cut frequency, and pacing rhythm.
Produces a reusable style profile that the assembler can apply.

No OpenCV dependency — uses FFmpeg's scene detection filter instead.
"""
import json
import re
import subprocess
import time
from pathlib import Path

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _run(cmd, timeout=120):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:5]))}")
    return p


def _get_duration(video: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    p = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    return float(p.stdout.strip())


def detect_cuts(video: Path, threshold: float = 0.3) -> list[float]:
    """Detect scene cuts using FFmpeg's scene change filter.

    Returns a list of timestamps (seconds) where cuts occur.
    """
    _ts(f"Detecting scene cuts in {video.name} (threshold={threshold})...")

    p = subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=300,
    )

    # Parse timestamps from showinfo output
    cuts = []
    for match in re.finditer(r"pts_time:(\d+\.?\d*)", p.stderr):
        cuts.append(float(match.group(1)))

    _ts(f"  Found {len(cuts)} scene cuts")
    return cuts


def analyze_style(video: Path) -> dict:
    """Analyze a reference video and extract its editing style profile.

    Returns a dict with:
    - avg_shot_duration: average seconds per shot
    - cut_frequency: cuts per minute
    - total_duration: video length
    - total_cuts: number of detected scene changes
    - pacing: "fast" (<3s avg), "medium" (3-6s), "slow" (>6s)
    - recommended_scene_pad: suggested SCENE_PAD based on pacing
    - recommended_transition_dur: suggested transition duration
    """
    _ts(f"Analyzing editing style of: {video.name}")

    duration = _get_duration(video)
    cuts = detect_cuts(video)
    total_cuts = len(cuts)

    # Calculate shot durations
    if total_cuts > 0:
        boundaries = [0.0] + cuts + [duration]
        shot_durations = [
            boundaries[i+1] - boundaries[i]
            for i in range(len(boundaries) - 1)
        ]
        avg_shot = sum(shot_durations) / len(shot_durations)
        min_shot = min(shot_durations)
        max_shot = max(shot_durations)
    else:
        avg_shot = duration
        min_shot = duration
        max_shot = duration
        shot_durations = [duration]

    cuts_per_min = (total_cuts / duration * 60) if duration > 0 else 0

    # Classify pacing
    if avg_shot < 3:
        pacing = "fast"
        rec_pad = 0.3
        rec_trans = 0.3
    elif avg_shot < 6:
        pacing = "medium"
        rec_pad = 0.5
        rec_trans = 0.5
    else:
        pacing = "slow"
        rec_pad = 0.8
        rec_trans = 0.7

    profile = {
        "avg_shot_duration": round(avg_shot, 2),
        "min_shot_duration": round(min_shot, 2),
        "max_shot_duration": round(max_shot, 2),
        "cut_frequency_per_min": round(cuts_per_min, 1),
        "total_duration": round(duration, 1),
        "total_cuts": total_cuts,
        "total_shots": total_cuts + 1,
        "pacing": pacing,
        "recommended_scene_pad": rec_pad,
        "recommended_transition_dur": rec_trans,
    }

    _ts(f"  Style: {pacing} pacing, {avg_shot:.1f}s avg shot, "
        f"{cuts_per_min:.0f} cuts/min")

    return profile


def save_style_profile(profile: dict, outdir: Path, name: str = "style") -> Path:
    """Save a style profile to JSON."""
    path = outdir / f"{name}_profile.json"
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    _ts(f"Style profile saved: {path}")
    return path


def clone_style(reference_video: Path, outdir: Path) -> dict:
    """Full style cloning pipeline: analyze → save → return profile."""
    profile = analyze_style(reference_video)
    save_style_profile(profile, outdir)
    return profile
