#!/usr/bin/env python3
"""
Gradio web UI for the multi-genre video production pipeline.

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
HEADERS = [
    "Narration (what the voice says)",
    "Image prompt (what to draw)",
    "Speaker",
    "Mood",
]


# ── helpers ─────────────────────────────────────────────────────────────

def _genre_choices() -> list[str]:
    return genre_ids()


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


def get_suggestions(genre_id):
    suggestions = suggest_topics(genre_id, limit=6)
    if not suggestions:
        return "No cached trends. Click 'Refresh trends' first (needs internet)."
    return "\n".join(f"• {s}" for s in suggestions)


# ── Step 1: script generation ──────────────────────────────────────────

def write_script(genre_id, topic, age, n_scenes, model, do_refine):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Please type a topic first.")
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
    return (
        script.get("title", topic),
        script.get("description", ""),
        ", ".join(script.get("tags", [])),
        rows,
        str(outdir),
        f"Script ready{refined_note} ({len(rows)} scenes). "
        f"Review it, then click 'Make the video'.",
    )


# ── Step 2: render ─────────────────────────────────────────────────────

def make_video(genre_id, outdir, title, description, tags, table, speed, res, seed):
    if not outdir:
        raise gr.Error("Write the script first (step 1).")

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

    # Live log
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
            # Wire SFX and thumbnail loggers to the tee as well
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
                scenes, imgs, auds, durs, genre, outdir,
                title=title,
            )

            # Thumbnail
            if config.GENERATE_THUMBNAIL and imgs:
                try:
                    from pipeline.thumbnail import generate_thumbnail
                    best_img = imgs[min(1, len(imgs) - 1)]
                    # Use the raw (non-upscaled) version for thumbnail
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

    lines = ["Starting... (first run downloads the image model — can take a few minutes)"]
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

def on_genre_change(genre_id):
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
        gr.update(value=get_suggestions(genre_id)),
    )


# ── Gradio UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI Video Creator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI Video Creator\n"
        "Pick a genre → enter a topic → generate → review → render.\n"
        "All processing runs locally on your GPU."
    )
    outdir_state = gr.State("")

    with gr.Row():
        # ── Left column: controls ──
        with gr.Column(scale=1):
            gr.Markdown("### Genre & Topic")
            genre_dd = gr.Dropdown(
                choices=_genre_choices(),
                value="drama_story",
                label="Genre",
                info="Choose the style of video to create",
            )
            with gr.Accordion("Trend suggestions (needs internet)", open=False):
                btn_trends = gr.Button("Refresh trends", size="sm")
                trends_box = gr.Markdown("")
                btn_trends.click(refresh_trends, outputs=trends_box)
                genre_dd.change(get_suggestions, [genre_dd], trends_box)

            topic = gr.Textbox(
                label="Topic / Premise",
                placeholder="e.g. A mother discovers her son has been lying for years",
                lines=2,
            )
            age = gr.Dropdown(
                ["general"], value="general", label="Age group",
                visible=False,
            )
            n_scenes = gr.Slider(3, 12, value=8, step=1, label="Scenes")
            model = gr.Dropdown(
                ["qwen2.5:7b", "llama3.2:3b"],
                value=config.OLLAMA_MODEL,
                label="Ollama model",
                allow_custom_value=True,
            )
            do_refine = gr.Checkbox(
                value=True, label="Refine script (second LLM pass)",
                info="Improves hooks and pacing — adds ~30-60s to script generation",
            )
            btn_script = gr.Button("1. Write script", variant="primary")

            gr.Markdown("### Render settings")
            speed = gr.Slider(0.7, 1.2, value=0.85, step=0.05, label="Voice speed")
            res = gr.Dropdown(
                list(RESOLUTIONS), value="720p (faster)", label="Resolution"
            )
            seed = gr.Number(
                value=1234, precision=0,
                label="Image seed (change for different pictures)"
            )

        # ── Right column: script + output ──
        with gr.Column(scale=2):
            gr.Markdown("### Script (review & edit)")
            status = gr.Markdown("")
            vid_title = gr.Textbox(label="Video title", interactive=True)
            vid_desc = gr.Textbox(label="Description", lines=3, interactive=True)
            vid_tags = gr.Textbox(label="Tags (comma separated)", interactive=True)
            table = gr.Dataframe(
                headers=HEADERS,
                datatype=["str", "str", "str", "str"],
                col_count=(4, "fixed"),
                row_count=(8, "dynamic"),
                interactive=True,
                wrap=True,
                label="Scenes",
            )
            btn_make = gr.Button("2. Make the video", variant="primary")

    gr.Markdown("### Result")
    log_box = gr.Textbox(label="Progress log", lines=12, max_lines=16, interactive=False)
    video = gr.Video(label="Finished video")
    gr.Markdown(
        "Each video folder also contains: `thumbnail.png`, `metadata.json`, "
        "and the individual scene images/audio for further editing."
    )

    # ── Event wiring ──
    genre_dd.change(on_genre_change, [genre_dd], [speed, n_scenes, age, trends_box])

    btn_script.click(
        write_script,
        [genre_dd, topic, age, n_scenes, model, do_refine],
        [vid_title, vid_desc, vid_tags, table, outdir_state, status],
    )
    btn_make.click(
        make_video,
        [genre_dd, outdir_state, vid_title, vid_desc, vid_tags, table, speed, res, seed],
        [log_box, video],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True, allowed_paths=[str(config.OUT_ROOT.resolve())])
