"""
B-Roll auto-injection — fetches contextual stock footage from Pexels.

Covers talking-head pauses and amplifies viewer retention by inserting
relevant visual cutaways between scenes or during scene transitions.

Requires a free Pexels API key (env var PEXELS_API_KEY or passed directly).
Falls back gracefully if no API key or no results found.
"""
import os
import time
from pathlib import Path
from typing import Any

import requests

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


PEXELS_API = "https://api.pexels.com/videos/search"


def search_broll(
    query: str,
    api_key: str | None = None,
    orientation: str = "portrait",
    n: int = 3,
) -> list[dict]:
    """Search Pexels for stock video clips matching a query.

    Returns list of {url, width, height, duration} dicts.
    """
    key = api_key or os.environ.get("PEXELS_API_KEY", "")
    if not key:
        _ts("B-Roll: no Pexels API key — skipping")
        return []

    _ts(f"B-Roll: searching Pexels for '{query[:40]}'...")
    try:
        r = requests.get(
            PEXELS_API,
            params={"query": query, "orientation": orientation, "per_page": n},
            headers={"Authorization": key},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        clips = []
        for v in data.get("videos", [])[:n]:
            # Pick the smallest HD file
            files = sorted(v.get("video_files", []), key=lambda f: f.get("width", 9999))
            for f in files:
                if f.get("width", 0) >= 720:
                    clips.append({
                        "url": f["link"],
                        "width": f.get("width", 0),
                        "height": f.get("height", 0),
                        "duration": v.get("duration", 0),
                        "query": query,
                    })
                    break
        _ts(f"B-Roll: found {len(clips)} clips")
        return clips
    except Exception as exc:
        _ts(f"B-Roll: Pexels search failed: {exc}")
        return []


def download_broll(clips: list[dict], outdir: Path) -> list[Path]:
    """Download B-Roll clips to disk. Returns list of local paths."""
    broll_dir = outdir / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, clip in enumerate(clips):
        out = broll_dir / f"broll_{i:02d}.mp4"
        try:
            _ts(f"B-Roll: downloading clip {i+1}/{len(clips)}...")
            r = requests.get(clip["url"], timeout=30, stream=True)
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            paths.append(out)
        except Exception as exc:
            _ts(f"B-Roll: download failed for clip {i}: {exc}")

    return paths


def get_broll_for_scenes(
    scenes: list[dict],
    outdir: Path,
    api_key: str | None = None,
    max_clips: int = 3,
) -> list[Path]:
    """Fetch B-Roll clips relevant to the scene content.

    Picks keywords from scene narrations and searches Pexels.
    Returns downloaded clip paths (may be empty if no API key or results).
    """
    # Build search queries from scene content
    queries = set()
    for s in scenes:
        narration = s.get("narration", "")
        mood = s.get("mood", "")
        # Extract key nouns/concepts (simple heuristic)
        words = narration.split()
        if len(words) > 3:
            queries.add(" ".join(words[1:4]))  # first few content words
        if mood:
            queries.add(mood)

    if not queries:
        return []

    all_clips = []
    for q in list(queries)[:3]:  # limit to 3 searches
        clips = search_broll(q, api_key=api_key, n=1)
        all_clips.extend(clips)
        if len(all_clips) >= max_clips:
            break

    if not all_clips:
        return []

    return download_broll(all_clips[:max_clips], outdir)
