"""
Audio analysis for multimodal clipping.

Detects volume peaks (applause, laughter, punchlines, music hits)
in the audio track using FFmpeg's volumedetect and astats filters.
These peaks are fused with transcript timestamps for better clip selection.
"""
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
    return p


def detect_volume_peaks(
    video: Path,
    window_seconds: float = 2.0,
    top_n: int = 20,
) -> list[dict]:
    """Detect the loudest moments in a video's audio track.

    Splits audio into windows and measures RMS volume for each.
    Returns the top N loudest windows sorted by timestamp.

    Each result: {time_s, rms_db, is_peak}
    """
    _ts("Analyzing audio volume peaks...")

    # Get total duration
    p = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ])
    if p.returncode != 0:
        _ts("  Could not probe audio duration")
        return []

    total_dur = float(p.stdout.strip())
    n_windows = int(total_dur / window_seconds)
    if n_windows < 2:
        return []

    # Measure RMS for each window using FFmpeg astats
    volumes = []
    for i in range(n_windows):
        t = i * window_seconds
        p = _run([
            "ffmpeg", "-ss", f"{t:.1f}",
            "-i", str(video),
            "-t", f"{window_seconds}",
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-",
        ], timeout=30)

        # Parse RMS from stderr
        rms = -60.0  # default (silence)
        for match in re.finditer(r"lavfi\.astats\.Overall\.RMS_level=(-?\d+\.?\d*)", p.stderr):
            rms = float(match.group(1))

        volumes.append({"time_s": t, "rms_db": rms})

    if not volumes:
        return []

    # Find the average RMS
    valid_rms = [v["rms_db"] for v in volumes if v["rms_db"] > -59]
    if not valid_rms:
        return []

    avg_rms = sum(valid_rms) / len(valid_rms)

    # Mark peaks (significantly above average)
    for v in volumes:
        v["is_peak"] = v["rms_db"] > avg_rms + 3  # 3dB above average

    # Sort by loudness, take top N
    peaks = sorted(volumes, key=lambda v: v["rms_db"], reverse=True)[:top_n]
    # Re-sort by time for chronological order
    peaks.sort(key=lambda v: v["time_s"])

    _ts(f"  Found {sum(1 for p in peaks if p['is_peak'])} volume peaks "
        f"(avg RMS: {avg_rms:.1f} dB)")

    return peaks


def detect_scene_cuts(video: Path, threshold: float = 0.3) -> list[float]:
    """Detect visual scene cuts using FFmpeg's scene filter.

    Returns timestamps where significant visual changes occur.
    """
    _ts("Detecting visual scene cuts...")

    p = subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=300,
    )

    cuts = []
    for match in re.finditer(r"pts_time:(\d+\.?\d*)", p.stderr):
        cuts.append(float(match.group(1)))

    _ts(f"  Found {len(cuts)} scene cuts")
    return cuts


def fuse_hotspots(
    transcript_highlights: list[dict],
    volume_peaks: list[dict],
    scene_cuts: list[float],
    merge_window: float = 5.0,
) -> list[dict]:
    """Fuse multimodal signals into ranked hotspot timestamps.

    Combines transcript-based highlights with audio peaks and scene cuts.
    Moments where multiple signals converge get higher scores.

    Returns sorted list of {time_s, score, sources} dicts.
    """
    _ts("Fusing multimodal hotspots...")

    hotspots = {}  # time_bucket -> {score, sources}

    # Add transcript highlights (highest weight)
    for h in transcript_highlights:
        t = h.get("start", 0)
        bucket = int(t / merge_window) * merge_window
        if bucket not in hotspots:
            hotspots[bucket] = {"time_s": t, "score": 0, "sources": []}
        hotspots[bucket]["score"] += 3
        hotspots[bucket]["sources"].append("transcript")
        if not hotspots[bucket].get("caption"):
            hotspots[bucket]["caption"] = h.get("caption", "")

    # Add volume peaks (medium weight)
    for v in volume_peaks:
        if not v.get("is_peak"):
            continue
        t = v["time_s"]
        bucket = int(t / merge_window) * merge_window
        if bucket not in hotspots:
            hotspots[bucket] = {"time_s": t, "score": 0, "sources": [], "caption": ""}
        hotspots[bucket]["score"] += 2
        hotspots[bucket]["sources"].append("audio_peak")

    # Add scene cuts (lower weight — context changes)
    for t in scene_cuts:
        bucket = int(t / merge_window) * merge_window
        if bucket not in hotspots:
            hotspots[bucket] = {"time_s": t, "score": 0, "sources": [], "caption": ""}
        hotspots[bucket]["score"] += 1
        hotspots[bucket]["sources"].append("scene_cut")

    # Sort by score (highest first), then by time
    ranked = sorted(hotspots.values(), key=lambda h: (-h["score"], h["time_s"]))

    _ts(f"  {len(ranked)} hotspots (top score: {ranked[0]['score'] if ranked else 0})")
    return ranked
