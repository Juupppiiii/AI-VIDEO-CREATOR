"""
Output quality assurance.

Runs quick checks on the final video to catch common problems
before the user watches a broken file.

Inspired by the cartoon-audio-short kit's QA pipeline.
"""
import json
import subprocess
import time
from pathlib import Path

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _probe(video: Path) -> dict | None:
    """Run ffprobe and return parsed JSON, or None on failure."""
    try:
        p = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(video),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if p.returncode == 0:
            return json.loads(p.stdout)
    except Exception:
        pass
    return None


def check_video(video: Path | None, outdir: Path) -> list[str]:
    """Check the final video for common issues.

    Returns a list of issue descriptions (empty = all good).
    """
    issues = []

    if video is None:
        issues.append("No video file produced.")
        return issues

    if not video.exists():
        issues.append(f"Video file missing: {video}")
        return issues

    # File size check
    size_mb = video.stat().st_size / (1024 * 1024)
    if size_mb < 0.05:
        issues.append(f"Video too small ({size_mb:.2f} MB) — likely empty or corrupt.")

    # Probe with ffprobe
    info = _probe(video)
    if info is None:
        issues.append("ffprobe could not read the video — file may be corrupt.")
        return issues

    # Duration check
    fmt = info.get("format", {})
    duration = float(fmt.get("duration", 0))
    if duration < 1.0:
        issues.append(f"Video too short ({duration:.1f}s) — assembly likely failed.")
    elif duration > 600:
        issues.append(f"Video unusually long ({duration:.0f}s) — check scene count.")

    # Stream checks
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video:
        issues.append("No video stream found in the output file.")
    if not has_audio:
        issues.append("No audio stream found — voice/music may have failed.")

    # Video stream details
    for s in streams:
        if s.get("codec_type") == "video":
            w = int(s.get("width", 0))
            h = int(s.get("height", 0))
            if w == 0 or h == 0:
                issues.append("Video has zero dimensions.")
            # Check for reasonable framerate
            fps_str = s.get("avg_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0
                if fps < 10:
                    issues.append(f"Low framerate ({fps:.1f} fps) — render may be incomplete.")
            except (ValueError, ZeroDivisionError):
                pass

    # Audio silence check — look at overall audio bitrate
    for s in streams:
        if s.get("codec_type") == "audio":
            bit_rate = int(s.get("bit_rate", 0))
            if 0 < bit_rate < 8000:
                issues.append("Audio bitrate very low — may be near-silent.")

    # Write QA report
    report = {
        "video": str(video),
        "size_mb": round(size_mb, 2),
        "duration_s": round(duration, 1),
        "has_video": has_video,
        "has_audio": has_audio,
        "issues": issues,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    qa_path = outdir / "qa_report.json"
    qa_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return issues
