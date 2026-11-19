"""
Research module — gathers factual context before script generation.

Inspired by AI Content Studio: searches the web and news for the topic,
then produces a research brief the LLM uses to write a better script.

Sources:
  - Google search (via requests, no API key needed)
  - News headlines (via Google News RSS, no API key needed)
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from pipeline.config import OLLAMA_URL, OLLAMA_MODEL

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Web search (no API key) ────────────────────────────────────────────

def _search_web(query: str, n: int = 5) -> list[dict]:
    """Simple web search via DuckDuckGo Lite (no API key)."""
    _ts(f"Searching web: {query[:60]}...")
    try:
        r = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        # Extract result snippets from the HTML
        results = []
        for match in re.finditer(
            r'class="result-snippet">(.*?)</td>', r.text, re.DOTALL
        ):
            text = re.sub(r"<.*?>", "", match.group(1)).strip()
            if text and len(text) > 30:
                results.append({"text": text[:300]})
                if len(results) >= n:
                    break
        _ts(f"  Found {len(results)} search results")
        return results
    except Exception as exc:
        _ts(f"  Web search failed: {exc}")
        return []


def _search_news(query: str, n: int = 5) -> list[dict]:
    """Fetch news headlines via Google News RSS (no API key)."""
    _ts(f"Searching news: {query[:60]}...")
    try:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en"
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)
        items = []
        for item in root.iter("item"):
            title = item.find("title")
            pub = item.find("pubDate")
            if title is not None and title.text:
                items.append({
                    "headline": title.text.strip(),
                    "date": pub.text.strip() if pub is not None and pub.text else "",
                })
                if len(items) >= n:
                    break
        _ts(f"  Found {len(items)} news headlines")
        return items
    except Exception as exc:
        _ts(f"  News search failed: {exc}")
        return []


# ── LLM research synthesis ──────────────────────────────────────────────

def _synthesize(topic: str, web_results: list, news: list, model: str) -> str:
    """Ask the LLM to produce a research brief from gathered data."""
    _ts("Synthesizing research brief...")

    context_parts = []
    if web_results:
        context_parts.append("WEB SEARCH RESULTS:")
        for i, r in enumerate(web_results, 1):
            context_parts.append(f"  {i}. {r['text']}")
    if news:
        context_parts.append("\nRECENT NEWS:")
        for i, n in enumerate(news, 1):
            context_parts.append(f"  {i}. {n['headline']} ({n.get('date', '')})")

    if not context_parts:
        return ""

    context = "\n".join(context_parts)

    system = (
        "You are a research assistant. Summarize the key facts, "
        "interesting angles, and surprising details about a topic. "
        "Be concise and factual. Reply in plain text, not JSON."
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Here is what I found online:\n\n{context}\n\n"
        "Write a short research brief (150-250 words) covering:\n"
        "- The most interesting/surprising facts\n"
        "- Common misconceptions to avoid\n"
        "- Good story angles for a video\n"
        "- Any recent developments"
    )

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0.3},
        }, timeout=120)
        r.raise_for_status()
        brief = r.json()["message"]["content"].strip()
        _ts(f"Research brief: {len(brief)} chars")
        return brief
    except Exception as exc:
        _ts(f"Research synthesis failed: {exc}")
        return ""


# ── public API ──────────────────────────────────────────────────────────

def research_topic(
    topic: str,
    model: str | None = None,
) -> str:
    """Research a topic and return a brief for the script generator.

    Searches web + news, then asks the LLM to synthesize a research
    brief. Returns the brief as a string (empty if research fails —
    the pipeline continues without it).
    """
    model = model or OLLAMA_MODEL
    _ts(f"Researching topic: {topic}")

    web = _search_web(topic)
    news = _search_news(topic)

    if not web and not news:
        _ts("No research data found — script will use LLM knowledge only")
        return ""

    brief = _synthesize(topic, web, news, model)
    return brief
