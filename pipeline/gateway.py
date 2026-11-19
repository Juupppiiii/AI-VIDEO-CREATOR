"""
Multi-model LLM gateway with automatic failover.

Supports: Ollama (local, default), Google Gemini, OpenAI-compatible APIs.
Falls back to the next provider if the primary fails.

Usage:
    from pipeline.gateway import llm_chat
    result = llm_chat(system, user)  # uses default provider
    result = llm_chat(system, user, provider="gemini", api_key="...")
"""
import json
import os
import time
from typing import Any

import requests

from pipeline.config import OLLAMA_URL, OLLAMA_MODEL

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Provider implementations ───────────────────────────────────────────

def _call_ollama(
    system: str, user: str, model: str, json_mode: bool = True,
) -> str:
    """Call local Ollama. Returns raw text response."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0.7},
    }
    if json_mode:
        body["format"] = "json"

    r = requests.post(OLLAMA_URL, json=body, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"]


def _call_gemini(
    system: str, user: str, model: str, api_key: str, json_mode: bool = True,
) -> str:
    """Call Google Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    body = {
        "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
        "generationConfig": {
            "temperature": 0.7,
        },
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    r = requests.post(
        url, params={"key": api_key}, json=body, timeout=120,
    )
    r.raise_for_status()
    candidates = r.json().get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    return candidates[0]["content"]["parts"][0]["text"]


def _call_openai(
    system: str, user: str, model: str, api_key: str,
    base_url: str = "https://api.openai.com/v1",
    json_mode: bool = True,
) -> str:
    """Call OpenAI or compatible API (together.ai, groq, etc.)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Unified gateway ────────────────────────────────────────────────────

PROVIDERS = {
    "ollama": _call_ollama,
    "gemini": _call_gemini,
    "openai": _call_openai,
}


def llm_chat(
    system: str,
    user: str,
    *,
    provider: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    json_mode: bool = True,
    fallback_providers: list[str] | None = None,
) -> str:
    """Send a chat message through the LLM gateway.

    Tries the primary provider, then falls back to alternatives.
    Returns the raw text response.
    """
    providers_to_try = [provider] + (fallback_providers or [])

    # Default models per provider
    default_models = {
        "ollama": model or OLLAMA_MODEL,
        "gemini": model or "gemini-2.0-flash",
        "openai": model or "gpt-4o-mini",
    }

    last_error = None
    for prov in providers_to_try:
        m = default_models.get(prov, model or OLLAMA_MODEL)
        _ts(f"LLM gateway: trying {prov} ({m})...")

        try:
            if prov == "ollama":
                return _call_ollama(system, user, m, json_mode)
            elif prov == "gemini":
                key = api_key or os.environ.get("GEMINI_API_KEY", "")
                if not key:
                    raise ValueError("No Gemini API key")
                return _call_gemini(system, user, m, key, json_mode)
            elif prov == "openai":
                key = api_key or os.environ.get("OPENAI_API_KEY", "")
                if not key:
                    raise ValueError("No OpenAI API key")
                url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                return _call_openai(system, user, m, key, url, json_mode)
            else:
                raise ValueError(f"Unknown provider: {prov}")
        except Exception as exc:
            _ts(f"  {prov} failed: {exc}")
            last_error = exc
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def llm_chat_json(system: str, user: str, **kwargs) -> dict:
    """Convenience: call llm_chat and parse JSON response."""
    raw = llm_chat(system, user, json_mode=True, **kwargs)
    return json.loads(raw)
