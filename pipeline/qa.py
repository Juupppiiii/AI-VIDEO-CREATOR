"""
Output quality assurance + scored script QA.

Runs quick checks on the final video to catch common problems
before the user watches a broken file.

Also provides scored script evaluation that triggers auto-rewrite
when quality is below threshold (inspired by AI Content Studio's
self-healing QA retry loop).
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


# ── Scored Script QA ────────────────────────────────────────────────────

def score_script(
    script: dict,
    model: str | None = None,
    threshold: int = 70,
) -> tuple[int, str, dict | None]:
    """Score a script 0-100 and return improvement suggestions.

    Returns (score, feedback, rewritten_script_or_None).
    If score < threshold, the LLM also returns a rewritten version.
    """
    from pipeline.config import OLLAMA_URL, OLLAMA_MODEL
    import requests

    model = model or OLLAMA_MODEL
    _ts(f"Scoring script quality (threshold: {threshold})...")

    scenes_text = "\n".join(
        f"Scene {i+1}: {s.get('narration', '')}"
        for i, s in enumerate(script.get("scenes", []))
    )

    system = (
        "You are a video script quality assessor. Score scripts 0-100 "
        "on hook strength, pacing, factual accuracy, and narrative arc. "
        "Reply with valid JSON only."
    )
    user = f"""Score this video script:

Title: {script.get("title", "")}
Scenes:
{scenes_text}

Evaluate on these criteria (each 0-25 points):
1. HOOK (0-25): Does scene 1 grab attention in 3 seconds?
2. PACING (0-25): Does each scene earn the next 15 seconds?
3. ACCURACY (0-25): Are claims plausible and not obviously wrong?
4. ARC (0-25): Is there a clear setup, escalation, and payoff?

Return JSON:
{{
  "score": 75,
  "hook_score": 20,
  "pacing_score": 18,
  "accuracy_score": 22,
  "arc_score": 15,
  "feedback": "specific feedback on what's weak",
  "rewrite_needed": true
}}

If the total score is below {threshold}, also include a "rewritten_scenes" array
with improved narration for each scene (same format as the input)."""

    try:
        from pipeline.script import _safe_format, _call_ollama
        data = _call_ollama(system, user, model)

        score = int(data.get("score", 50))
        feedback = data.get("feedback", "")
        _ts(f"Script score: {score}/100 (hook={data.get('hook_score')}, "
            f"pacing={data.get('pacing_score')}, accuracy={data.get('accuracy_score')}, "
            f"arc={data.get('arc_score')})")
        _ts(f"Feedback: {feedback[:200]}")

        rewritten = None
        if score < threshold and data.get("rewritten_scenes"):
            _ts(f"Score below {threshold} — applying auto-rewrite")
            rewritten = dict(script)
            new_scenes = data["rewritten_scenes"]
            if isinstance(new_scenes, list) and len(new_scenes) == len(script.get("scenes", [])):
                for i, ns in enumerate(new_scenes):
                    if isinstance(ns, dict) and ns.get("narration"):
                        rewritten["scenes"][i]["narration"] = ns["narration"]
                        if ns.get("image_prompt"):
                            rewritten["scenes"][i]["image_prompt"] = ns["image_prompt"]

        return score, feedback, rewritten

    except Exception as exc:
        _ts(f"Script scoring failed: {exc}")
        return 50, "Scoring unavailable", None
