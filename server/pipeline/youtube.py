"""
YouTube highlight clipper.

Downloads a video via yt-dlp, extracts subtitles, uses the local LLM
to identify the most engaging moments, then clips and assembles them
into a highlight reel with transitions and captions.

Requires ``yt-dlp`` in PATH — install with ``pip install yt-dlp``.
"""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pipeline.config import (
    OLLAMA_URL, OLLAMA_MODEL, OUT_ROOT, OUT_W, OUT_H, FPS,
    TRANSITION_DUR, MUSIC_DIR, MUSIC_VOLUME,
)

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _run(cmd, cwd=None, timeout=600):
    p = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd[:4]))} ...\n"
            + (p.stderr or "")[-1500:]
        )
    return p


def check_ytdlp() -> bool:
    """Return True if yt-dlp is available."""
    return shutil.which("yt-dlp") is not None


# ── Download ────────────────────────────────────────────────────────────

def download_video(url: str, outdir: Path) -> tuple[Path, Path | None]:
    """Download video + auto-subtitles. Returns (video_path, srt_path_or_None)."""
    _ts(f"Downloading video: {url}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Download video as mp4
    _run([
        "yt-dlp",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
        "--merge-output-format", "mp4",
        "--write-auto-sub", "--sub-lang", "en",
        "--convert-subs", "srt",
        "--no-playlist",
        "-o", str(outdir / "source.%(ext)s"),
        url,
    ], timeout=300)

    # Find the downloaded files
    video = None
    srt = None
    for f in outdir.iterdir():
        if f.name.startswith("source") and f.suffix == ".mp4":
            video = f
        elif f.suffix == ".srt" and "source" in f.name:
            srt = f

    if video is None:
        raise FileNotFoundError("yt-dlp did not produce an mp4 file.")
    _ts(f"Downloaded: {video.name}" + (f" + {srt.name}" if srt else " (no subtitles)"))
    return video, srt


# ── Subtitle parsing ────────────────────────────────────────────────────

def _srt_time_to_seconds(ts: str) -> float:
    """Parse '00:01:23,456' → 83.456"""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_srt(path: Path) -> list[dict]:
    """Parse SRT into [{start, end, text}, ...]"""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        entries.append({
            "start": _srt_time_to_seconds(time_match.group(1)),
            "end": _srt_time_to_seconds(time_match.group(2)),
            "text": " ".join(lines[2:]).strip(),
        })
    return entries


def transcript_to_text(entries: list[dict], chunk_seconds: float = 30) -> str:
    """Collapse subtitle entries into timestamped chunks for the LLM."""
    if not entries:
        return ""
    chunks = []
    current_start = entries[0]["start"]
    current_texts = []
    for e in entries:
        if e["start"] - current_start > chunk_seconds and current_texts:
            mins = int(current_start // 60)
            secs = int(current_start % 60)
            chunks.append(f"[{mins:02d}:{secs:02d}] {' '.join(current_texts)}")
            current_start = e["start"]
            current_texts = []
        current_texts.append(e["text"])
    if current_texts:
        mins = int(current_start // 60)
        secs = int(current_start % 60)
        chunks.append(f"[{mins:02d}:{secs:02d}] {' '.join(current_texts)}")
    return "\n".join(chunks)


# ── LLM highlight identification ────────────────────────────────────────

def identify_highlights(
    transcript: str,
    n_clips: int = 5,
    clip_duration: int = 15,
    model: str | None = None,
) -> list[dict]:
    """Ask the LLM to pick the most engaging moments from a transcript."""
    import requests

    model = model or OLLAMA_MODEL
    _ts(f"Asking {model} to identify {n_clips} highlights...")

    system = (
        "You analyze video transcripts and identify the most engaging, "
        "dramatic, funny, or surprising moments. You always reply with "
        "valid JSON only."
    )
    user = f"""Here is a timestamped transcript of a video:

{transcript[:8000]}

Identify the {n_clips} most engaging, shareable, or dramatic moments.
For each moment, pick a start timestamp that captures the lead-up
(a few seconds before the key moment).

Each clip should be about {clip_duration} seconds long.

Return JSON in this exact format:
{{
  "title": "catchy compilation title under 70 characters",
  "highlights": [
    {{
      "start_time": "MM:SS",
      "caption": "short 1-sentence description of what happens"
    }}
  ]
}}"""

    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": 0,
        "options": {"temperature": 0.5},
    }, timeout=900)
    r.raise_for_status()

    data = json.loads(r.json()["message"]["content"])
    highlights = data.get("highlights", [])

    # Parse timestamps
    result = []
    for h in highlights:
        ts = h.get("start_time", "0:00")
        parts = ts.replace(".", ":").split(":")
        try:
            if len(parts) == 2:
                secs = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) >= 3:
                secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                secs = int(parts[0])
        except ValueError:
            secs = 0
        result.append({
            "start": max(0, secs - 2),  # start 2s before the moment
            "duration": clip_duration,
            "caption": h.get("caption", ""),
        })

    _ts(f"Found {len(result)} highlights")
    return result, data.get("title", "Highlights")


# ── Clipping and assembly ───────────────────────────────────────────────

def clip_highlights(
    source: Path,
    highlights: list[dict],
    outdir: Path,
) -> list[Path]:
    """Extract individual clips from the source video."""
    clips_dir = outdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    paths = []

    for i, h in enumerate(highlights):
        out = clips_dir / f"clip_{i:02d}.mp4"
        _run([
            "ffmpeg", "-y",
            "-ss", str(h["start"]),
            "-i", str(source),
            "-t", str(h["duration"]),
            "-vf", f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                   f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-ar", "44100", "-ac", "2",
            str(out),
        ])
        paths.append(out)
        _ts(f"Clip {i + 1}/{len(highlights)} extracted ({h['start']}s, {h['duration']}s)")

    return paths


def _srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def assemble_clips(
    clips: list[Path],
    highlights: list[dict],
    title: str,
    outdir: Path,
) -> Path:
    """Concatenate clips with transitions and caption overlay."""
    if not clips:
        raise ValueError("No clips to assemble.")

    # Write captions SRT
    srt_path = outdir / "captions.srt"
    srt_lines = []
    t = 0.0
    for i, (clip, h) in enumerate(zip(clips, highlights)):
        dur = h["duration"]
        caption = h.get("caption", "")
        if caption:
            srt_lines += [
                str(i + 1),
                f"{_srt_time(t + 0.5)} --> {_srt_time(t + dur - 0.5)}",
                caption,
                "",
            ]
        t += dur
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    # Concat with ffmpeg
    concat_file = outdir / "list.txt"
    concat_file.write_text(
        "\n".join(f"file '{c.as_posix()}'" for c in clips),
        encoding="utf-8",
    )

    merged = outdir / "merged.mp4"
    _run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy", str(merged),
    ])

    # Add captions
    srt_posix = srt_path.resolve().as_posix().replace(":", "\\:")
    style = (
        "FontName=Arial,FontSize=22,Bold=1,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BackColour=&H80000000,BorderStyle=3,"
        "Outline=2,Shadow=0,Alignment=2,MarginV=35"
    )

    final = outdir / "final.mp4"
    _run([
        "ffmpeg", "-y", "-i", str(merged),
        "-vf", f"subtitles='{srt_posix}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "copy",
        str(final),
    ])

    # Metadata
    meta = {
        "title": title,
        "clips": len(clips),
        "highlights": [h["caption"] for h in highlights],
        "containsSyntheticMedia": False,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    _ts(f"Highlight reel: {final}")
    return final


# ── public API ──────────────────────────────────────────────────────────

def process_youtube(
    url: str,
    n_clips: int = 5,
    clip_duration: int = 15,
    model: str | None = None,
) -> Path:
    """Full pipeline: URL → download → highlights → clip → assemble."""
    slug = re.sub(r"[^a-z0-9]+", "-", url.lower())[-30:].strip("-") or "yt"
    outdir = OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-yt-{slug}"
    outdir.mkdir(parents=True, exist_ok=True)

    video, srt = download_video(url, outdir)

    if srt:
        entries = parse_srt(srt)
        transcript = transcript_to_text(entries)
    else:
        _ts("No subtitles found — using video title for highlight detection")
        # Fallback: get video info for the LLM
        p = _run(["yt-dlp", "--print", "title", "--no-playlist", url])
        transcript = f"Video title: {p.stdout.strip()}\n(No transcript available)"

    highlights, title = identify_highlights(transcript, n_clips, clip_duration, model)

    if not highlights:
        _ts("LLM found no highlights. Clipping evenly spaced segments instead.")
        # Fallback: evenly space clips across the video
        p = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ])
        total_dur = float(p.stdout.strip())
        spacing = total_dur / (n_clips + 1)
        highlights = [
            {"start": int(spacing * (i + 1)), "duration": clip_duration, "caption": ""}
            for i in range(n_clips)
        ]
        title = "Highlights"

    clips = clip_highlights(video, highlights, outdir)
    final = assemble_clips(clips, highlights, title, outdir)
    return final, outdir
