"""
Upload package generator — titles, description, hashtags, thumbnail prompt.

Inspired by the Master Prompt's Stage 5: produces a copy-ready upload
package for YouTube after the video is rendered.
"""
import json
import time
from pathlib import Path
from typing import Any

from pipeline.config import OLLAMA_URL, OLLAMA_MODEL
from pipeline.script import _safe_format, _call_ollama

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


_PACKAGE_SYSTEM = (
    "You generate YouTube upload packages: titles, descriptions, and hashtags "
    "for short-form animated videos. You always reply with valid JSON only."
)

_PACKAGE_USER = """Here is a completed video script. Generate its upload package.

Title: {title}
Genre: {genre}
Script summary:
{summary}

Produce:
1. FIVE title options — curiosity-driven, under 70 characters each. Mark one as recommended.
2. A copy-ready description with:
   - A strong opening explaining the premise without spoiling the ending
   - A natural invitation to comment on the story's dilemma
   - A brief subscribe line for more animated stories
   - A short disclosure: "An AI-assisted animation created for entertainment."
   - Three relevant hashtags
3. A thumbnail prompt describing a single striking image from the story — the most dramatic or curious moment, with the main character visible and a clear focal point. Include the visual style (flat 2D character in 3D environment).

Reply with JSON:
{
  "titles": [
    {"text": "title option 1", "recommended": true},
    {"text": "title option 2", "recommended": false},
    {"text": "title option 3", "recommended": false},
    {"text": "title option 4", "recommended": false},
    {"text": "title option 5", "recommended": false}
  ],
  "description": "full copy-ready description text",
  "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"],
  "thumbnail_prompt": "detailed image generation prompt for thumbnail"
}"""


def generate_upload_package(
    script: dict,
    genre: dict[str, Any],
    outdir: Path,
    model: str | None = None,
) -> dict:
    """Generate a YouTube upload package from a completed script.

    Returns dict with titles, description, hashtags, thumbnail_prompt.
    Also saves to ``upload_package.json`` in the output directory.
    """
    model = model or OLLAMA_MODEL
    _ts("Generating upload package (titles, description, thumbnail prompt)...")

    # Build a summary from the script
    scenes = script.get("scenes", [])
    summary_lines = []
    for i, s in enumerate(scenes, 1):
        narration = s.get("narration", "")[:100]
        summary_lines.append(f"Scene {i}: {narration}")
    summary = "\n".join(summary_lines)

    user_msg = _safe_format(
        _PACKAGE_USER,
        title=script.get("title", "Untitled"),
        genre=genre.get("name", ""),
        summary=summary,
    )

    try:
        package = _call_ollama(_PACKAGE_SYSTEM, user_msg, model)

        # Validate structure
        if "titles" not in package or not isinstance(package["titles"], list):
            package["titles"] = [{"text": script.get("title", ""), "recommended": True}]
        if "description" not in package:
            package["description"] = script.get("description", "")
        if "hashtags" not in package:
            package["hashtags"] = []
        if "thumbnail_prompt" not in package:
            package["thumbnail_prompt"] = ""

        # Save
        out_path = outdir / "upload_package.json"
        out_path.write_text(
            json.dumps(package, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _ts(f"Upload package saved: {out_path}")

        # Log the titles
        for t in package.get("titles", []):
            marker = " [recommended]" if t.get("recommended") else ""
            _ts(f"  Title: {t.get('text', '')}{marker}")

        return package

    except Exception as exc:
        _ts(f"Upload package generation failed: {exc}")
        return {
            "titles": [{"text": script.get("title", ""), "recommended": True}],
            "description": script.get("description", ""),
            "hashtags": [],
            "thumbnail_prompt": "",
        }
