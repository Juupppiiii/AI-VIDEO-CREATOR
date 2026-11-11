"""
Load and manage genre templates.

Each genre is a JSON file in ``genres/templates/`` that defines the LLM prompts,
image style, voice settings, and music mood for a particular video category.
"""
import json
from pathlib import Path
from typing import Any

from pipeline.config import GENRES_DIR

_cache: dict[str, dict[str, Any]] = {}


def _templates_dir() -> Path:
    """Return the templates directory, creating it if missing."""
    GENRES_DIR.mkdir(parents=True, exist_ok=True)
    return GENRES_DIR


def list_genres() -> list[dict[str, str]]:
    """Return ``[{"id": ..., "name": ..., "description": ...}, ...]``."""
    genres = []
    for path in sorted(_templates_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            genres.append({
                "id": path.stem,
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return genres


def load_genre(genre_id: str) -> dict[str, Any]:
    """Load a genre template by id (filename without ``.json``)."""
    if genre_id in _cache:
        return _cache[genre_id]
    path = _templates_dir() / f"{genre_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Genre template not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    _cache[genre_id] = data
    return data


def genre_ids() -> list[str]:
    """Return a plain list of genre id strings."""
    return [g["id"] for g in list_genres()]
