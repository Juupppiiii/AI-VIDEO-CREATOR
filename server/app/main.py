#!/usr/bin/env python3
"""
Gradio web UI for the multi-genre video production pipeline.
iOS-inspired design with clean cards, soft shadows, and intuitive flow.

Run:  python server/app/main.py   (opens at http://127.0.0.1:7860)
"""
import json
import os
import queue
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
from trends.fetcher import (
    update_trends, get_cached_trends, suggest_topics, cache_age_hours,
)

# ── constants ───────────────────────────────────────────────────────────
RESOLUTIONS = {"720p (faster)": (1280, 720), "1080p (slower)": (1920, 1080)}
HEADERS = ["Narration", "Image prompt", "Speaker", "Mood"]

# Genre display with emoji icons
GENRE_LABELS = {
    "drama_story":      "🎭  Drama",
    "kids_educational":  "📚  Kids",
    "trivia_quiz":       "❓  Trivia",
    "horror_story":      "👻  Horror",
    "motivational":      "💪  Motivational",
    "mystery":           "🔍  Mystery",
}


# ── iOS-inspired CSS ────────────────────────────────────────────────────
IOS_CSS = """
/* ═══ iOS Design System ═══ */

:root {
    --ios-blue: #007AFF;
    --ios-blue-hover: #0066D6;
    --ios-green: #34C759;
    --ios-red: #FF3B30;
    --ios-bg: #F2F2F7;
    --ios-card: #FFFFFF;
    --ios-text: #1C1C1E;
    --ios-secondary: #8E8E93;
    --ios-tertiary: #AEAEB2;
    --ios-separator: rgba(60, 60, 67, 0.1);
    --ios-radius: 16px;
    --ios-radius-sm: 12px;
    --ios-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
    --ios-shadow-hover: 0 2px 8px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06);
    --ios-font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                "Helvetica Neue", Arial, sans-serif;
}

/* Base container */
.gradio-container {
    font-family: var(--ios-font) !important;
    background: var(--ios-bg) !important;
    max-width: 900px !important;
    margin: 0 auto !important;
}

/* Remove default Gradio borders everywhere */
.block, .form, .wrap {
    border: none !important;
}

/* ═══ Header ═══ */
.ios-header {
    text-align: center;
    padding: 28px 16px 12px;
}

.ios-header h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--ios-text) !important;
    letter-spacing: -0.02em !important;
    margin: 0 0 4px 0 !important;
}

.ios-header p {
    font-size: 15px !important;
    color: var(--ios-secondary) !important;
    margin: 0 !important;
    font-weight: 400 !important;
}

/* ═══ Cards ═══ */
.ios-card {
    background: var(--ios-card) !important;
    border-radius: var(--ios-radius) !important;
    box-shadow: var(--ios-shadow) !important;
    border: none !important;
    padding: 20px 20px 16px !important;
    margin-bottom: 12px !important;
}

.ios-card .label-wrap {
    margin-bottom: 4px !important;
}

/* Section titles inside cards */
.ios-section-title {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--ios-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 0 4px !important;
    margin-bottom: 8px !important;
}

/* ═══ Genre Radio Buttons (segmented control look) ═══ */
.ios-genre-picker .wrap {
    gap: 8px !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
}

.ios-genre-picker label {
    background: var(--ios-card) !important;
    border: 2px solid transparent !important;
    border-radius: var(--ios-radius-sm) !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: var(--ios-text) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    min-width: 110px !important;
    text-align: center !important;
}

.ios-genre-picker label:hover {
    box-shadow: var(--ios-shadow-hover) !important;
    transform: translateY(-1px) !important;
}

.ios-genre-picker label.selected {
    border-color: var(--ios-blue) !important;
    background: rgba(0, 122, 255, 0.06) !important;
    color: var(--ios-blue) !important;
    font-weight: 600 !important;
}

.ios-genre-picker .label-wrap {
    display: none !important;
}

/* ═══ Inputs ═══ */
.ios-card input[type="text"],
.ios-card textarea,
.ios-card .wrap input {
    border: 1.5px solid var(--ios-separator) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-family: var(--ios-font) !important;
    font-size: 15px !important;
    background: var(--ios-bg) !important;
    transition: border-color 0.2s ease !important;
}

.ios-card input[type="text"]:focus,
.ios-card textarea:focus {
    border-color: var(--ios-blue) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.12) !important;
}

/* ═══ Buttons ═══ */
.ios-btn-primary button {
    background: var(--ios-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--ios-radius-sm) !important;
    padding: 12px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    font-family: var(--ios-font) !important;
    letter-spacing: -0.01em !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}

.ios-btn-primary button:hover {
    background: var(--ios-blue-hover) !important;
    transform: scale(0.98) !important;
}

.ios-btn-primary button:active {
    transform: scale(0.96) !important;
}

.ios-btn-secondary button {
    background: rgba(0, 122, 255, 0.08) !important;
    color: var(--ios-blue) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: var(--ios-font) !important;
    transition: all 0.15s ease !important;
}

.ios-btn-secondary button:hover {
    background: rgba(0, 122, 255, 0.14) !important;
}

/* ═══ Sliders ═══ */
input[type="range"] {
    accent-color: var(--ios-blue) !important;
}

/* ═══ Dropdowns ═══ */
.ios-card select, .ios-card .wrap select {
    border-radius: 10px !important;
    border: 1.5px solid var(--ios-separator) !important;
    padding: 8px 12px !important;
    background: var(--ios-bg) !important;
    font-family: var(--ios-font) !important;
}

/* ═══ Checkbox ═══ */
.ios-card input[type="checkbox"] {
    accent-color: var(--ios-blue) !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 5px !important;
}

/* ═══ Status pill ═══ */
.ios-status .prose {
    background: rgba(52, 199, 89, 0.1) !important;
    color: #248A3D !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: center !important;
    display: inline-block !important;
}

/* ═══ Progress log ═══ */
.ios-log textarea {
    background: #1C1C1E !important;
    color: #98989D !important;
    border-radius: var(--ios-radius-sm) !important;
    border: none !important;
    font-family: "SF Mono", "Menlo", "Monaco", "Courier New", monospace !important;
    font-size: 12px !important;
    padding: 14px !important;
    line-height: 1.6 !important;
}

/* ═══ Video player ═══ */
.ios-card video {
    border-radius: var(--ios-radius-sm) !important;
}

/* ═══ Dataframe / Table ═══ */
.ios-card table {
    border-radius: var(--ios-radius-sm) !important;
    overflow: hidden !important;
    font-size: 13px !important;
}

.ios-card table th {
    background: var(--ios-bg) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    color: var(--ios-secondary) !important;
}

/* ═══ Accordion ═══ */
.ios-card .accordion {
    border: none !important;
    background: transparent !important;
}

/* ═══ Step numbers ═══ */
.ios-step-label {
    display: inline-block;
    background: var(--ios-blue);
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    text-align: center;
    line-height: 24px;
    font-size: 13px;
    font-weight: 700;
    margin-right: 8px;
}

/* ═══ Footer ═══ */
.ios-footer {
    text-align: center;
    padding: 16px;
    font-size: 13px;
    color: var(--ios-tertiary);
}

/* ═══ Responsive ═══ */
@media (max-width: 768px) {
    .gradio-container {
        padding: 0 8px !important;
    }
    .ios-genre-picker label {
        min-width: 90px !important;
        font-size: 13px !important;
    }
}
"""


# ── helpers ─────────────────────────────────────────────────────────────

def _genre_radio_choices() -> list[str]:
    """Genre choices as 'id' values with emoji labels for Radio."""
    ids = genre_ids()
    return [GENRE_LABELS.get(gid, gid) for gid in ids]


def _genre_id_from_label(label: str) -> str:
    """Reverse-map an emoji label back to a genre id."""
    for gid, glabel in GENRE_LABELS.items():
        if glabel == label:
            return gid
    return label  # fallback: the label IS the id


def _slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"


# ── Step 0: trend suggestions ──────────────────────────────────────────

def refresh_trends():
    try:
        update_trends()
        return "Trends updated."
    except Exception as exc:
        return f"Could not fetch trends: {exc}"


def get_suggestions(genre_label):
    genre_id = _genre_id_from_label(genre_label)
    suggestions = suggest_topics(genre_id, limit=5)
    if not suggestions:
        return "No cached trends. Click refresh first."
    return "\n".join(f"• {s}" for s in suggestions)


# ── Step 1: script generation ──────────────────────────────────────────

def write_script(genre_label, topic, age, n_scenes, model, do_refine):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Please type a topic first.")

    genre_id = _genre_id_from_label(genre_label)
    try:
        genre = load_genre(genre_id)
    except FileNotFoundError:
        raise gr.Error(f"Unknown genre: {genre_id}")

    config.OLLAMA_MODEL = model
    outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{genre_id}-{_slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        script = generate_script(
            topic, genre, n_scenes=int(n_scenes), age=age, refine=do_refine,
        )
    except SystemExit as exc:
        raise gr.Error(str(exc))
    except Exception as exc:
        raise gr.Error(f"Script generation failed: {exc}")

    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = [
        [s["narration"], s["image_prompt"], s.get("speaker", "narrator"), s.get("mood", "")]
        for s in script["scenes"]
    ]

    refined_note = " (refined)" if do_refine else ""
    status_msg = f"Script ready{refined_note} — {len(rows)} scenes. Review and edit below, then render."

    return (
        script.get("title", topic),
        script.get("description", ""),
        ", ".join(script.get("tags", [])),
        rows,
        str(outdir),
        status_msg,
    )


# ── Step 2: render ─────────────────────────────────────────────────────

def make_video(genre_label, outdir, title, description, tags, table, speed, res, seed):
    if not outdir:
        raise gr.Error("Generate a script first.")

    genre_id = _genre_id_from_label(genre_label)
    try:
        genre = load_genre(genre_id)
    except FileNotFoundError:
        raise gr.Error(f"Unknown genre: {genre_id}")

    rows = table.values.tolist() if hasattr(table, "values") else table
    scenes = []
    for row in rows:
        narration = (row[0] or "").strip()
        prompt = (row[1] or "").strip() or narration
        speaker = (row[2] or "").strip() or "narrator"
        mood = (row[3] or "").strip() or "neutral"
        if narration:
            scenes.append({
                "narration": narration,
                "image_prompt": prompt,
                "speaker": speaker,
                "mood": mood,
            })
    if len(scenes) < 1:
        raise gr.Error("The script table is empty.")

    outdir = Path(outdir)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    script = {"title": title, "description": description, "tags": tag_list, "scenes": scenes}
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config.OUT_W, config.OUT_H = RESOLUTIONS[res]

    q: queue.Queue = queue.Queue()
    import pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    for mod in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        mod.set_logger(tee)

    result: dict = {}

    def work():
        try:
            try:
                from pipeline.sfx import set_logger as set_sfx_log
                set_sfx_log(tee)
            except Exception:
                pass
            try:
                from pipeline.thumbnail import set_logger as set_thumb_log
                set_thumb_log(tee)
            except Exception:
                pass

            imgs = generate_images(scenes, genre, outdir, int(seed))
            auds, durs = generate_voice(scenes, genre, outdir, float(speed))
            result["final"] = assemble(
                scenes, imgs, auds, durs, genre, outdir, title=title,
            )

            if config.GENERATE_THUMBNAIL and imgs:
                try:
                    from pipeline.thumbnail import generate_thumbnail
                    best_img = imgs[min(1, len(imgs) - 1)]
                    raw_name = best_img.stem.removesuffix("_up") + best_img.suffix
                    raw = best_img.with_name(raw_name)
                    if raw.exists() and raw != best_img:
                        best_img = raw
                    generate_thumbnail(title, best_img, genre_id, outdir)
                except Exception as exc:
                    q.put(f"Thumbnail skipped: {exc}")

            meta = {
                "title": title,
                "description": (description or "").strip()
                    + "\n\nThis video was created with the help of AI tools.",
                "tags": tag_list,
                "genre": genre.get("name", ""),
                "madeForKids": genre.get("made_for_kids", False),
                "containsSyntheticMedia": True,
            }
            (outdir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            q.put(f"DONE -> {outdir / 'final.mp4'}")
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
                lines.append("Timed out waiting for the render thread.")
                break
            if msg is None:
                break
            lines.append(msg)
            yield "\n".join(lines), None
    finally:
        for mod in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
            mod.set_logger(print)

    video = str(result["final"]) if "final" in result else None
    yield "\n".join(lines), video


# ── Genre change handler ───────────────────────────────────────────────

def on_genre_change(genre_label):
    genre_id = _genre_id_from_label(genre_label)
    try:
        genre = load_genre(genre_id)
    except FileNotFoundError:
        return gr.update(), gr.update(), gr.update(), gr.update()

    speed = genre.get("voice_speed", 0.9)
    scenes = genre.get("target_scenes", 7)
    has_age = "age_options" in genre
    age_choices = genre.get("age_options", ["general"])
    default_age = genre.get("default_age", "general")

    return (
        gr.update(value=speed),
        gr.update(value=scenes),
        gr.update(visible=has_age, choices=age_choices, value=default_age),
        gr.update(value=get_suggestions(genre_label)),
    )


# ── Gradio UI (iOS-inspired) ──────────────────────────────────────────

with gr.Blocks(title="AI Video Creator", css=IOS_CSS) as demo:
    outdir_state = gr.State("")

    # ── Header ──
    gr.HTML("""
        <div class="ios-header">
            <h1>AI Video Creator</h1>
            <p>Choose a genre, enter a topic, and create a video — all locally on your GPU.</p>
        </div>
    """)

    # ══════════════ STEP 1: Genre & Topic ══════════════
    with gr.Group(elem_classes="ios-card"):
        gr.HTML('<div class="ios-section-title"><span class="ios-step-label">1</span> Genre & Topic</div>')

        genre_radio = gr.Radio(
            choices=_genre_radio_choices(),
            value=GENRE_LABELS.get("drama_story", "drama_story"),
            label="Genre",
            elem_classes="ios-genre-picker",
        )

        with gr.Accordion("💡 Trending topic ideas", open=False):
            with gr.Row():
                trends_box = gr.Markdown("No cached trends. Click refresh.")
                btn_trends = gr.Button("Refresh", elem_classes="ios-btn-secondary", scale=0)
            btn_trends.click(refresh_trends, outputs=trends_box)

        topic = gr.Textbox(
            label="Topic / Premise",
            placeholder="A mother discovers her son has been lying for years...",
            lines=2,
            max_lines=4,
        )

        age = gr.Dropdown(
            ["general"], value="general", label="Age group", visible=False,
        )

        with gr.Row():
            n_scenes = gr.Slider(3, 12, value=8, step=1, label="Scenes")
            model = gr.Dropdown(
                ["qwen2.5:7b", "llama3.2:3b"],
                value=config.OLLAMA_MODEL,
                label="Model",
                allow_custom_value=True,
                scale=1,
            )

        do_refine = gr.Checkbox(
            value=True, label="Refine script",
            info="Second LLM pass for better hooks and pacing",
        )

        gr.HTML('<div style="height: 4px;"></div>')
        btn_script = gr.Button(
            "Write Script",
            variant="primary",
            elem_classes="ios-btn-primary",
        )

    # ══════════════ STEP 2: Review & Edit ══════════════
    with gr.Group(elem_classes="ios-card"):
        gr.HTML('<div class="ios-section-title"><span class="ios-step-label">2</span> Review & Edit Script</div>')

        status = gr.Markdown("", elem_classes="ios-status")

        vid_title = gr.Textbox(label="Title", interactive=True)

        with gr.Row():
            vid_desc = gr.Textbox(label="Description", lines=2, interactive=True, scale=2)
            vid_tags = gr.Textbox(label="Tags", interactive=True, scale=1,
                                  placeholder="drama, story, twist")

        table = gr.Dataframe(
            headers=HEADERS,
            datatype=["str", "str", "str", "str"],
            col_count=(4, "fixed"),
            row_count=(8, "dynamic"),
            interactive=True,
            wrap=True,
            label="Scenes",
        )

        with gr.Row():
            speed = gr.Slider(0.7, 1.2, value=0.85, step=0.05, label="Voice speed")
            res = gr.Dropdown(
                list(RESOLUTIONS), value="720p (faster)", label="Resolution",
            )
            seed = gr.Number(value=1234, precision=0, label="Image seed")

        gr.HTML('<div style="height: 4px;"></div>')
        btn_make = gr.Button(
            "Make Video",
            variant="primary",
            elem_classes="ios-btn-primary",
        )

    # ══════════════ STEP 3: Result ══════════════
    with gr.Group(elem_classes="ios-card"):
        gr.HTML('<div class="ios-section-title"><span class="ios-step-label">3</span> Result</div>')

        log_box = gr.Textbox(
            label="Progress",
            lines=10, max_lines=14,
            interactive=False,
            elem_classes="ios-log",
        )
        video = gr.Video(label="Finished video")

    # ── Footer ──
    gr.HTML("""
        <div class="ios-footer">
            Each video folder includes: thumbnail.png · metadata.json · scene images & audio
        </div>
    """)

    # ── Event wiring ──
    genre_radio.change(
        on_genre_change, [genre_radio], [speed, n_scenes, age, trends_box],
    )
    genre_radio.change(get_suggestions, [genre_radio], trends_box)

    btn_script.click(
        write_script,
        [genre_radio, topic, age, n_scenes, model, do_refine],
        [vid_title, vid_desc, vid_tags, table, outdir_state, status],
    )
    btn_make.click(
        make_video,
        [genre_radio, outdir_state, vid_title, vid_desc, vid_tags, table, speed, res, seed],
        [log_box, video],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True, allowed_paths=[str(config.OUT_ROOT.resolve())])
