#!/usr/bin/env python3
"""
Gradio web UI for the multi-genre video production pipeline.
Clean, minimal design inspired by Apple and Ultron dashboard.

Run:  python server/app/main.py   (opens at http://127.0.0.1:7860)
"""
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────
_APP = Path(__file__).resolve().parent
_SERVER = _APP.parent
_ROOT = _SERVER.parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
os.chdir(_ROOT)

import gradio as gr

from pipeline import config
from pipeline.script import generate_script, set_logger as set_script_log
from pipeline.images import generate_images, set_logger as set_img_log
from pipeline.voice import generate_voice, set_logger as set_voice_log
from pipeline.assembler import assemble, set_logger as set_asm_log
from genres.loader import load_genre, list_genres, genre_ids
from trends.fetcher import update_trends, suggest_topics, cache_age_hours
import requests as _requests

# ── constants ───────────────────────────────────────────────────────────
RESOLUTIONS = {"720p (faster)": (1280, 720), "1080p (slower)": (1920, 1080)}
HEADERS = ["Narration", "Image prompt", "Speaker", "Mood"]

GENRE_MAP = {
    "drama_story":      "Drama",
    "horror_story":      "Horror",
    "kids_educational":  "Kids",
    "motivational":      "Motivational",
    "mystery":           "Mystery",
    "trivia_quiz":       "Trivia",
}
LABEL_TO_ID = {v: k for k, v in GENRE_MAP.items()}


# ── clean design CSS (Inter font, ultron-inspired tokens) ───────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #ffffff;
    --surface: #fafafa;
    --border: #e5e7eb;
    --border-strong: #d1d5db;
    --text: #111827;
    --text-2: #6b7280;
    --text-3: #9ca3af;
    --accent: #2563eb;
    --accent-hover: #1d4ed8;
    --accent-subtle: #eff6ff;
    --success: #15803d;
    --success-subtle: #f0fdf4;
    --danger: #b91c1c;
    --danger-subtle: #fef2f2;
    --radius: 10px;
    --radius-lg: 14px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04);
    --font: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
}

/* Base */
.gradio-container {
    font-family: var(--font) !important;
    background: var(--surface) !important;
    max-width: 920px !important;
    margin: 0 auto !important;
    -webkit-font-smoothing: antialiased !important;
}
.block, .form { border: none !important; }

/* ── Header ── */
.app-header { padding: 32px 4px 20px; }
.app-header h1 {
    font-size: 26px !important; font-weight: 700 !important;
    color: var(--text) !important; letter-spacing: -0.03em !important;
    margin: 0 !important;
}
.app-header p {
    font-size: 14px !important; color: var(--text-2) !important;
    margin: 4px 0 0 !important;
}

/* ── Panels (ultron-style cards) ── */
.panel {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow) !important;
    padding: 22px !important;
    margin-bottom: 14px !important;
}
.panel-title {
    font-size: 11px !important; font-weight: 600 !important;
    color: var(--text-3) !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; margin-bottom: 14px !important;
}

/* ── Genre selector ── */
.genre-select .wrap {
    gap: 6px !important; flex-wrap: wrap !important;
}
.genre-select label {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: var(--text) !important;
    cursor: pointer !important;
    transition: all 150ms ease !important;
    min-width: 88px !important;
    text-align: center !important;
}
.genre-select label:hover {
    border-color: var(--accent) !important;
    background: var(--accent-subtle) !important;
}
.genre-select label.selected {
    border-color: var(--accent) !important;
    background: var(--accent-subtle) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
.genre-select .label-wrap { display: none !important; }

/* ── Inputs ── */
.panel input[type="text"], .panel textarea {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 9px 12px !important;
    font-family: var(--font) !important;
    font-size: 14px !important;
    background: var(--bg) !important;
    transition: border-color 150ms !important;
}
.panel input:focus, .panel textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* ── Buttons ── */
.btn-primary button {
    background: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 14px !important; font-weight: 600 !important;
    font-family: var(--font) !important;
    width: 100% !important;
    transition: background 150ms !important;
}
.btn-primary button:hover { background: var(--accent-hover) !important; }
.btn-ghost button {
    background: transparent !important; color: var(--accent) !important;
    border: none !important; padding: 6px 12px !important;
    font-size: 13px !important; font-weight: 600 !important;
    font-family: var(--font) !important;
}
.btn-ghost button:hover { background: var(--accent-subtle) !important; border-radius: 6px !important; }

/* ── Status badge ── */
.status-badge .prose {
    display: inline-block !important;
    background: var(--success-subtle) !important;
    color: var(--success) !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    font-size: 13px !important; font-weight: 500 !important;
}

/* ── Suggestions box ── */
.suggestions-box {
    background: var(--accent-subtle) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
    color: var(--text) !important;
    margin-bottom: 8px !important;
}
.suggestions-box .prose { font-size: 13px !important; line-height: 1.7 !important; }
.suggestions-label {
    font-size: 11px !important; font-weight: 600 !important;
    color: var(--accent) !important; text-transform: uppercase !important;
    letter-spacing: 0.04em !important; margin-bottom: 6px !important;
}

/* ── Log console ── */
.console textarea {
    background: #111827 !important; color: #9ca3af !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace !important;
    font-size: 12px !important; padding: 14px !important;
    line-height: 1.6 !important;
}

/* ── Table ── */
.panel table { font-size: 13px !important; border-radius: 8px !important; overflow: hidden !important; }
.panel table th {
    background: var(--surface) !important;
    font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.04em !important;
    color: var(--text-3) !important;
}

/* ── Sliders / checkboxes ── */
input[type="range"] { accent-color: var(--accent) !important; }
input[type="checkbox"] { accent-color: var(--accent) !important; }

/* ── Video ── */
.panel video { border-radius: 8px !important; }

/* ── Accordion ── */
.panel .accordion { border: none !important; background: transparent !important; }

/* ── Footer ── */
.app-footer { text-align: center; padding: 20px 0; color: var(--text-3); font-size: 12px; }

/* ── Tabs ── */
.tabs > .tab-nav > button {
    font-family: var(--font) !important;
    font-size: 14px !important; font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important;
}
.tabs > .tab-nav > button.selected {
    font-weight: 600 !important; color: var(--accent) !important;
}
"""


# ── helpers ─────────────────────────────────────────────────────────────

def _genre_choices():
    return [GENRE_MAP.get(gid, gid) for gid in genre_ids()]

def _to_id(label):
    return LABEL_TO_ID.get(label, label)

def _slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"

def _check_ollama():
    """Quick check if Ollama is reachable."""
    try:
        r = _requests.get(
            config.OLLAMA_URL.replace("/api/chat", ""), timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Trends ──────────────────────────────────────────────────────────────

def refresh_trends():
    try:
        update_trends()
        return "Trends refreshed."
    except Exception as exc:
        return f"Could not fetch trends: {exc}"

def get_trend_suggestions(genre_label):
    gid = _to_id(genre_label) if genre_label else "drama_story"
    suggestions = suggest_topics(gid, limit=5)
    if not suggestions:
        age = cache_age_hours()
        if age is None:
            return "No trend data cached yet. Click **Refresh** (needs internet)."
        return f"Trend cache is {age:.0f}h old. Click **Refresh** to update."
    return "\n".join(f"- {s}" for s in suggestions)


# ── Step 1: Write script ───────────────────────────────────────────────

def write_script(genre_label, topic, age, n_scenes, model, do_refine):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Enter a topic or premise first.")

    if not _check_ollama():
        raise gr.Error(
            "Cannot reach Ollama. Make sure 'ollama serve' is running "
            "and the model is pulled."
        )

    gid = _to_id(genre_label or "Drama")
    try:
        genre = load_genre(gid)
    except FileNotFoundError:
        raise gr.Error(f"Genre not found: {gid}. Available: {', '.join(genre_ids())}")

    config.OLLAMA_MODEL = model or config.OLLAMA_MODEL
    outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{gid}-{_slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        script = generate_script(
            topic, genre,
            n_scenes=int(n_scenes or genre.get("target_scenes", 7)),
            age=age if age != "general" else None,
            refine=bool(do_refine),
        )
    except SystemExit as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Script generation failed: {exc}")

    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    rows = [
        [s["narration"], s["image_prompt"], s.get("speaker", "narrator"), s.get("mood", "")]
        for s in script["scenes"]
    ]

    note = " (refined)" if do_refine else ""
    return (
        script.get("title", topic),
        script.get("description", ""),
        ", ".join(script.get("tags", [])),
        rows,
        str(outdir),
        f"Script ready{note} — {len(rows)} scenes",
    )


# ── Step 2: Render video ──────────────────────────────────────────────

def render_video(genre_label, outdir, title, desc, tags, table, speed, res, seed):
    if not outdir:
        raise gr.Error("Generate a script first.")

    gid = _to_id(genre_label or "Drama")
    try:
        genre = load_genre(gid)
    except FileNotFoundError:
        raise gr.Error(f"Genre not found: {gid}")

    rows = table.values.tolist() if hasattr(table, "values") else table
    scenes = []
    for row in rows:
        narr = (row[0] or "").strip()
        if not narr:
            continue
        scenes.append({
            "narration": narr,
            "image_prompt": (row[1] or "").strip() or narr,
            "speaker": (row[2] or "").strip() or "narrator",
            "mood": (row[3] or "").strip() or "neutral",
        })
    if not scenes:
        raise gr.Error("No scenes in the script table.")

    outdir = Path(outdir)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    script = {"title": title, "description": desc, "tags": tag_list, "scenes": scenes}
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    config.OUT_W, config.OUT_H = RESOLUTIONS.get(res, (1280, 720))

    q: queue.Queue = queue.Queue()
    import pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    for mod in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        mod.set_logger(tee)
    try:
        from pipeline.sfx import set_logger as sfx_log; sfx_log(tee)
    except Exception: pass
    try:
        from pipeline.thumbnail import set_logger as thumb_log; thumb_log(tee)
    except Exception: pass

    result = {}

    def work():
        try:
            imgs = generate_images(scenes, genre, outdir, int(seed))
            auds, durs = generate_voice(scenes, genre, outdir, float(speed))
            result["final"] = assemble(scenes, imgs, auds, durs, genre, outdir, title=title)

            if config.GENERATE_THUMBNAIL and imgs:
                try:
                    from pipeline.thumbnail import generate_thumbnail
                    best = imgs[min(1, len(imgs) - 1)]
                    raw_name = best.stem.removesuffix("_up") + best.suffix
                    raw = best.with_name(raw_name)
                    if raw.exists() and raw != best: best = raw
                    generate_thumbnail(title, best, gid, outdir)
                except Exception as e:
                    q.put(f"Thumbnail skipped: {e}")

            meta = {
                "title": title,
                "description": (desc or "").strip() + "\n\nCreated with AI Video Creator.",
                "tags": tag_list,
                "genre": genre.get("name", ""),
                "madeForKids": genre.get("made_for_kids", False),
                "containsSyntheticMedia": True,
            }
            (outdir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            q.put(f"Done — {outdir / 'final.mp4'}")
        except BaseException:
            result["error"] = traceback.format_exc()
            q.put("ERROR:\n" + result["error"])
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Starting render pipeline..."]
    yield "\n".join(lines), None
    try:
        while True:
            try:
                msg = q.get(timeout=300)
            except queue.Empty:
                lines.append("Timed out."); break
            if msg is None: break
            lines.append(msg)
            yield "\n".join(lines), None
    finally:
        for mod in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
            mod.set_logger(print)

    yield "\n".join(lines), str(result["final"]) if "final" in result else None


# ── YouTube clipper ────────────────────────────────────────────────────

def clip_youtube(url, n_clips, clip_dur, model):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube URL first.")

    from pipeline.youtube import check_ytdlp, process_youtube, set_logger
    if not check_ytdlp():
        raise gr.Error("yt-dlp not found. Install with: pip install yt-dlp")
    if not _check_ollama():
        raise gr.Error("Ollama not reachable. Run 'ollama serve' first.")

    q: queue.Queue = queue.Queue()

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    set_logger(tee)
    result = {}

    def work():
        try:
            final, outdir = process_youtube(
                url, n_clips=int(n_clips), clip_duration=int(clip_dur),
                model=model or None,
            )
            result["final"] = final
            q.put(f"Done — {final}")
        except BaseException:
            result["error"] = traceback.format_exc()
            q.put("ERROR:\n" + result["error"])
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Starting YouTube clip pipeline..."]
    yield "\n".join(lines), None
    try:
        while True:
            try:
                msg = q.get(timeout=300)
            except queue.Empty:
                lines.append("Timed out."); break
            if msg is None: break
            lines.append(msg)
            yield "\n".join(lines), None
    finally:
        pass

    yield "\n".join(lines), str(result["final"]) if "final" in result else None


# ── Genre change handler ──────────────────────────────────────────────

def on_genre_change(genre_label):
    gid = _to_id(genre_label or "Drama")
    try:
        genre = load_genre(gid)
    except FileNotFoundError:
        return gr.update(), gr.update(), gr.update(), gr.update()

    return (
        gr.update(value=genre.get("voice_speed", 0.9)),
        gr.update(value=genre.get("target_scenes", 7)),
        gr.update(
            visible="age_options" in genre,
            choices=genre.get("age_options", ["general"]),
            value=genre.get("default_age", "general"),
        ),
        get_trend_suggestions(genre_label),
    )


# ════════════════════════════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AI Video Creator", css=CSS) as demo:
    outdir_state = gr.State("")

    gr.HTML(
        '<div class="app-header">'
        '<h1>AI Video Creator</h1>'
        '<p>Generate videos from a topic or clip highlights from YouTube. '
        'Everything runs locally on your GPU.</p>'
        '</div>'
    )

    with gr.Tabs():

        # ═══════════════ TAB 1: Create Video ═══════════════
        with gr.TabItem("Create Video"):

            # ── Genre & Topic ──
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="panel-title">Genre & Topic</div>')

                genre_radio = gr.Radio(
                    choices=_genre_choices(),
                    value="Drama",
                    label="Genre",
                    elem_classes="genre-select",
                )

                # Trend suggestions (always visible)
                gr.HTML('<div class="suggestions-label">Suggested topics</div>')
                trends_box = gr.Markdown(
                    get_trend_suggestions("Drama"),
                    elem_classes="suggestions-box",
                )
                btn_trends = gr.Button(
                    "Refresh trends", elem_classes="btn-ghost", size="sm",
                )

                topic = gr.Textbox(
                    label="Topic / Premise",
                    placeholder="A mother discovers her son has been lying for years...",
                    lines=2,
                )
                age = gr.Dropdown(
                    ["general"], value="general", label="Age group", visible=False,
                )

                with gr.Row():
                    n_scenes = gr.Slider(3, 12, value=8, step=1, label="Scenes")
                    model = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL, label="Model",
                        allow_custom_value=True,
                    )

                do_refine = gr.Checkbox(
                    value=True, label="Refine script",
                    info="Second LLM pass for stronger hooks and pacing",
                )
                btn_script = gr.Button("Write Script", variant="primary", elem_classes="btn-primary")

            # ── Script review ──
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="panel-title">Script Review</div>')
                status = gr.Markdown("", elem_classes="status-badge")
                vid_title = gr.Textbox(label="Title", interactive=True)
                with gr.Row():
                    vid_desc = gr.Textbox(label="Description", lines=2, interactive=True, scale=2)
                    vid_tags = gr.Textbox(label="Tags", interactive=True, scale=1)
                table = gr.Dataframe(
                    headers=HEADERS, datatype=["str"] * 4,
                    col_count=(4, "fixed"), row_count=(8, "dynamic"),
                    interactive=True, wrap=True, label="Scenes",
                )
                with gr.Row():
                    speed = gr.Slider(0.7, 1.2, value=0.85, step=0.05, label="Voice speed")
                    res = gr.Dropdown(list(RESOLUTIONS), value="720p (faster)", label="Resolution")
                    seed = gr.Number(value=1234, precision=0, label="Image seed")
                btn_render = gr.Button("Render Video", variant="primary", elem_classes="btn-primary")

            # ── Output ──
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="panel-title">Output</div>')
                log_create = gr.Textbox(label="Progress", lines=10, max_lines=14, interactive=False, elem_classes="console")
                video_create = gr.Video(label="Video")

        # ═══════════════ TAB 2: YouTube Clipper ═══════════════
        with gr.TabItem("YouTube Clipper"):
            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="panel-title">Clip Highlights from YouTube</div>')
                gr.HTML(
                    '<p style="color:var(--text-2);font-size:13px;margin:0 0 14px">'
                    'Paste a YouTube URL. The system downloads the video, analyzes the '
                    'transcript with the local LLM, picks the best moments, and assembles '
                    'a highlight reel with captions.</p>'
                )
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                with gr.Row():
                    yt_clips = gr.Slider(2, 10, value=5, step=1, label="Number of clips")
                    yt_dur = gr.Slider(5, 30, value=15, step=5, label="Clip duration (sec)")
                    yt_model = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL, label="Model",
                        allow_custom_value=True,
                    )
                btn_clip = gr.Button("Clip Highlights", variant="primary", elem_classes="btn-primary")

            with gr.Group(elem_classes="panel"):
                gr.HTML('<div class="panel-title">Output</div>')
                log_yt = gr.Textbox(label="Progress", lines=10, max_lines=14, interactive=False, elem_classes="console")
                video_yt = gr.Video(label="Highlight Reel")

    gr.HTML('<div class="app-footer">Output includes thumbnail.png, metadata.json, and scene assets.</div>')

    # ── Wiring ──
    genre_radio.change(on_genre_change, [genre_radio], [speed, n_scenes, age, trends_box])
    btn_trends.click(refresh_trends, outputs=[trends_box])

    btn_script.click(
        write_script,
        [genre_radio, topic, age, n_scenes, model, do_refine],
        [vid_title, vid_desc, vid_tags, table, outdir_state, status],
    )
    btn_render.click(
        render_video,
        [genre_radio, outdir_state, vid_title, vid_desc, vid_tags, table, speed, res, seed],
        [log_create, video_create],
    )
    btn_clip.click(
        clip_youtube,
        [yt_url, yt_clips, yt_dur, yt_model],
        [log_yt, video_yt],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True, allowed_paths=[str(config.OUT_ROOT.resolve())])
