"""
Script generation and refinement via Ollama.

1. Generate: reads a genre template, formats the prompt, calls the LLM.
2. Refine (optional): second LLM pass that critiques hook strength,
   pacing, and cliffhangers — then produces an improved version.
"""
import json
import sys
import time
from typing import Any

import requests

from pipeline.config import OLLAMA_URL, OLLAMA_MODEL, REFINE_SCRIPT

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Safe template formatting ────────────────────────────────────────────

def _safe_format(template: str, **kwargs) -> str:
    """Format a template that contains literal braces (e.g. JSON examples).

    Escapes all ``{`` / ``}`` first, then re-opens only the known
    placeholder names so ``.format()`` fills them in.  This lets genre
    templates include JSON examples without doubling every brace.
    """
    # Escape every brace
    escaped = template.replace("{", "{{").replace("}", "}}")
    # Un-escape the known placeholders: {{key}} → {key}
    for key in kwargs:
        escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
    return escaped.format(**kwargs)


# ── LLM call ───────────────────────────────────────────────────────────

def _call_ollama(system: str, user: str, model: str) -> dict:
    """Single Ollama chat call. Returns parsed JSON dict."""
    r = requests.post(OLLAMA_URL, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": 0,
        "options": {"temperature": 0.7},
    }, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# ── Refinement prompt ──────────────────────────────────────────────────

_REFINE_SYSTEM = (
    "You are a script editor for viral short-form video content. "
    "You improve scripts to maximize viewer retention. "
    "You always reply with valid JSON only."
)

_REFINE_USER = """Here is a video script draft. Evaluate and improve it:

1. HOOK TEST: Does scene 1 make someone stop scrolling in 3 seconds? If not, rewrite it with a stronger cold open. The first sentence is everything.
2. PACING: Is each scene earning the next 15 seconds of watch time? If a scene feels flat or redundant, sharpen it.
3. MICRO-CLIFFHANGERS: Does every scene (except the last) end with a reason to keep watching? Add one if missing.
4. TWIST/PAYOFF: Is there a surprising moment? Does the ending satisfy and make the viewer want to share?
5. NARRATION: Tighten wordy sentences. Use present tense for immediacy. Cut filler.

Current script:
{script_json}

Return the IMPROVED script in the EXACT same JSON format with the same keys.
Keep the same number of scenes. Improve the narration text and image prompts.
Do not change speaker assignments or add/remove scenes."""


def refine_script(script: dict, model: str) -> dict:
    """Second LLM pass: critique and improve the script."""
    _ts("Refining script (second LLM pass)...")
    script_json = json.dumps(script, indent=2, ensure_ascii=False)
    user_msg = _safe_format(_REFINE_USER, script_json=script_json)

    try:
        refined = _call_ollama(_REFINE_SYSTEM, user_msg, model)
        # Validate structure
        scenes = refined.get("scenes", [])
        if not isinstance(scenes, list) or len(scenes) < 3:
            _ts("Refinement produced invalid structure, keeping original.")
            return script
        for s in scenes:
            if not s.get("narration", "").strip():
                _ts("Refinement has empty narration, keeping original.")
                return script
            s.setdefault("speaker", "narrator")
            s.setdefault("mood", "neutral")
            s.setdefault("image_prompt", "")
        refined.setdefault("title", script.get("title", ""))
        refined.setdefault("description", script.get("description", ""))
        refined.setdefault("tags", script.get("tags", []))
        refined["genre"] = script.get("genre", "unknown")
        _ts(f"Script refined — title: {refined.get('title', '?')}")
        return refined
    except Exception as exc:
        _ts(f"Refinement failed ({exc}), keeping original script.")
        return script


# ── public API ──────────────────────────────────────────────────────────

def generate_script(
    topic: str,
    genre: dict[str, Any],
    *,
    n_scenes: int | None = None,
    age: str | None = None,
    model: str | None = None,
    refine: bool | None = None,
) -> dict[str, Any]:
    """Generate a video script for *topic* using the given *genre* template.

    When ``refine`` is True (or defaults to the config setting), a second
    LLM pass critiques and improves the draft for better viewer retention.

    Returns the parsed JSON dict with ``title``, ``description``, ``tags``,
    ``scenes`` (each scene has ``narration``, ``image_prompt``, ``speaker``,
    ``mood``).
    """
    model = model or OLLAMA_MODEL
    n_scenes = n_scenes or genre.get("target_scenes", 7)
    age = age or genre.get("default_age", "general")
    do_refine = refine if refine is not None else REFINE_SCRIPT

    system = genre["script_system_prompt"]
    user = _safe_format(
        genre["script_user_template"],
        topic=topic,
        scenes=n_scenes,
        age=age,
    )

    data = None
    for attempt in range(1, 4):
        _ts(f"Asking {model} for a script (attempt {attempt})...")
        try:
            data = _call_ollama(system, user, model)
            scenes = data["scenes"]
            assert isinstance(scenes, list) and len(scenes) >= 3
            for s in scenes:
                assert s.get("narration", "").strip()
                assert s.get("image_prompt", "").strip()
                s.setdefault("speaker", "narrator")
                s.setdefault("mood", "neutral")
            data.setdefault("title", topic)
            data.setdefault("description", "")
            data.setdefault("tags", [])
            data["genre"] = genre.get("name", "unknown")
            _ts(f"Draft ready — {len(scenes)} scenes, title: {data['title']}")
            break
        except requests.RequestException as exc:
            _ts(f"Ollama request failed: {exc}")
            if attempt == 3:
                sys.exit(
                    "Could not reach Ollama. Is `ollama serve` running "
                    f"and the model `{model}` pulled?"
                )
        except Exception as exc:
            _ts(f"Bad script format ({exc}), retrying...")
            if attempt == 3:
                sys.exit("Could not get a valid script after 3 attempts.")

    if data is None:
        sys.exit("Could not get a valid script after 3 attempts.")

    # Optional refinement pass
    if do_refine:
        data = refine_script(data, model)

    return data
