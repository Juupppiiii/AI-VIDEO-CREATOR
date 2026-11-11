"""
Trend fetcher — pulls trending topics from public sources when online
and caches them locally for offline use.

Sources:
  - Google Trends RSS (no API key needed)

The cache lives at ``data/trends/latest.json``.
"""
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.config import DATA_DIR

CACHE_PATH = DATA_DIR / "trends" / "latest.json"
GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=US"

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Google Trends RSS ───────────────────────────────────────────────────

def _fetch_google_trends() -> list[dict[str, str]]:
    """Parse Google Trends daily RSS. Returns list of {title, traffic, link}."""
    import requests

    try:
        r = requests.get(GOOGLE_TRENDS_RSS, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        _ts(f"Google Trends fetch failed: {exc}")
        return []

    items = []
    try:
        root = ET.fromstring(r.content)
        ns = {"ht": "https://trends.google.com/trending/rss"}
        for item in root.iter("item"):
            title_el = item.find("title")
            traffic_el = item.find("ht:approx_traffic", ns)
            link_el = item.find("link")
            if title_el is not None and title_el.text:
                items.append({
                    "title": title_el.text.strip(),
                    "traffic": (traffic_el.text.strip()
                                if traffic_el is not None and traffic_el.text
                                else ""),
                    "link": (link_el.text.strip()
                             if link_el is not None and link_el.text
                             else ""),
                })
    except ET.ParseError as exc:
        _ts(f"Failed to parse Google Trends XML: {exc}")

    return items


# ── Suggest topics for a genre ──────────────────────────────────────────

_GENRE_HINTS: dict[str, list[str]] = {
    "kids_educational": [
        "Why do {trend} matter?",
        "Fun facts about {trend}",
        "Learn about {trend} with Sunny!",
    ],
    "drama_story": [
        "A {trend} secret that ruined everything",
        "The {trend} betrayal nobody expected",
        "She discovered the truth about {trend}",
    ],
    "trivia_quiz": [
        "Can you answer these {trend} questions?",
        "Only geniuses know these {trend} facts",
        "{trend}: what most people get wrong",
    ],
    "horror_story": [
        "The {trend} that should never have been found",
        "Something is wrong with {trend}",
        "Don't look up {trend} at 3 AM",
    ],
    "motivational": [
        "How {trend} changed one person's life forever",
        "They said {trend} was impossible. They were wrong.",
        "The {trend} story that will inspire you",
    ],
    "mystery": [
        "The unsolved {trend} mystery",
        "The {trend} case nobody can explain",
        "Who is behind the {trend} disappearance?",
    ],
}


def suggest_topics(genre_id: str, limit: int = 5) -> list[str]:
    """Return topic suggestions based on cached trends + genre hints."""
    trends = get_cached_trends()
    if not trends:
        return []

    hints = _GENRE_HINTS.get(genre_id, ["{trend}"])
    suggestions = []
    for trend in trends[:limit]:
        keyword = trend["title"]
        hint = hints[len(suggestions) % len(hints)]
        suggestions.append(hint.format(trend=keyword))

    return suggestions


# ── public API ──────────────────────────────────────────────────────────

def update_trends() -> dict[str, Any]:
    """Fetch trends from all sources and save to cache. Returns the cache."""
    _ts("Fetching trending topics...")

    google = _fetch_google_trends()
    _ts(f"  Google Trends: {len(google)} items")

    data = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "sources": {
            "google_trends": google,
        },
        "all": google,  # flat merged list for easy access
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    _ts(f"Trends cached to {CACHE_PATH}")
    return data


def get_cached_trends() -> list[dict[str, str]]:
    """Return the flat list of cached trends, or [] if no cache exists."""
    if not CACHE_PATH.exists():
        return []
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("all", [])
    except (json.JSONDecodeError, KeyError):
        return []


def cache_age_hours() -> float | None:
    """Return how many hours old the cache is, or None if missing."""
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        from datetime import timezone
        delta = datetime.now(timezone.utc) - ts
        return delta.total_seconds() / 3600
    except Exception:
        return None
