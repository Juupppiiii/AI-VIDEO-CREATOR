#!/usr/bin/env python3
"""
Gradio web UI — clean, minimal design adapted from Ultron dashboard.
Includes progress bars and an Open Folder button.

Run:  python server/app/main.py   (opens at http://127.0.0.1:7860)
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

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
from genres.loader import load_genre, genre_ids
from trends.fetcher import update_trends, suggest_topics, cache_age_hours
import requests as _requests

# ── constants ───────────────────────────────────────────────────────────
RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}
HEADERS = ["Narration", "Image prompt", "Speaker", "Mood"]
GENRE_MAP = {
    "drama_story": "Drama", "horror_story": "Horror",
    "kids_educational": "Kids", "motivational": "Motivational",
    "mystery": "Mystery", "trivia_quiz": "Trivia",
}
LABEL_TO_ID = {v: k for k, v in GENRE_MAP.items()}

# ── Gradio theme (Ultron design tokens) ────────────────────────────────
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#f9fafb", body_text_color="#111827",
    body_text_color_subdued="#6b7280",
    block_background_fill="#ffffff", block_border_width="1px",
    block_border_color="#e5e7eb", container_radius="12px",
    block_shadow="0 1px 3px 0 rgba(0,0,0,0.04)",
    block_title_text_color="#111827", block_title_text_size="14px",
    block_title_text_weight="600",
    block_label_text_color="#6b7280", block_label_text_size="13px",
    block_label_text_weight="500",
    input_background_fill="#ffffff", input_border_color="#e5e7eb",
    input_border_width="1px", input_radius="8px", input_text_size="14px",
    input_shadow="none", input_padding="10px 12px",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="white", button_primary_border_color="transparent",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#eff6ff",
    button_secondary_text_color="#2563eb", button_secondary_border_color="#d1d5db",
    button_border_width="1px",
    button_primary_shadow="none", button_secondary_shadow="none",
    button_large_radius="8px", button_large_text_size="14px",
    button_large_text_weight="600",
    button_small_radius="6px", button_small_text_size="13px",
    layout_gap="16px",
    section_header_text_size="11px", section_header_text_weight="600",
)

# ── Supplemental CSS ───────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 940px !important; margin: 0 auto !important; }
.panel-eyebrow { font-size: 11px; font-weight: 600; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 12px 2px; }
.genre-pills .label-wrap { display: none !important; }
.genre-pills .wrap { gap: 8px !important; }
.callout-box { border: 1px solid #bfdbfe; border-left: 3px solid #2563eb; border-radius: 8px; background: #eff6ff; padding: 14px 16px; margin-bottom: 12px; }
.callout-box p { margin: 0; font-size: 13px; color: #374151; line-height: 1.7; }
.callout-label { font-size: 11px; font-weight: 600; color: #2563eb; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.console-log textarea { background: #111827 !important; color: #d1d5db !important; border-radius: 8px !important; border: none !important; font-size: 12px !important; line-height: 1.65 !important; padding: 16px !important; }
.ok-badge span { display: inline-block; background: #f0fdf4; color: #15803d; border-radius: 999px; padding: 5px 14px; font-size: 13px; font-weight: 500; }
.app-footer { text-align: center; padding: 24px 0 8px; font-size: 12px; color: #9ca3af; }

/* Progress bar */
.pbar-wrap { margin: 8px 0 4px; }
.pbar-track { background: #f3f4f6; border-radius: 6px; height: 8px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
.pbar-fill-ok { background: #2563eb; }
.pbar-fill-done { background: #15803d; }
.pbar-fill-err { background: #b91c1c; }
.pbar-label { font-size: 12px; color: #6b7280; margin-top: 4px; display: flex; justify-content: space-between; }

/* Open folder button */
.open-folder-btn button { font-size: 13px !important; }
"""

# ── helpers ─────────────────────────────────────────────────────────────
def _choices():
    return [GENRE_MAP.get(g, g) for g in genre_ids()]

def _to_id(label):
    return LABEL_TO_ID.get(label, label)

def _slugify(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:40] or "video"

def _ollama_ok():
    try:
        return _requests.get(config.OLLAMA_URL.replace("/api/chat", ""), timeout=3).ok
    except Exception:
        return False


# ── progress bar ────────────────────────────────────────────────────────
def _pbar_html(frac: float, stage: str = "") -> str:
    """Render an HTML progress bar. frac in 0.0-1.0, negative = error."""
    if frac < 0:
        return (
            '<div class="pbar-wrap"><div class="pbar-track">'
            '<div class="pbar-fill pbar-fill-err" style="width:100%"></div>'
            '</div><div class="pbar-label"><span>Error</span></div></div>'
        )
    pct = min(int(frac * 100), 100)
    cls = "pbar-fill-done" if pct >= 100 else "pbar-fill-ok"
    return (
        f'<div class="pbar-wrap"><div class="pbar-track">'
        f'<div class="pbar-fill {cls}" style="width:{pct}%"></div>'
        f'</div><div class="pbar-label">'
        f'<span>{stage}</span><span>{pct}%</span></div></div>'
    )


def _estimate_render_progress(lines: list[str]) -> tuple[float, str]:
    """Parse log lines to estimate render progress and current stage."""
    if not lines:
        return 0.0, ""
    text = "\n".join(lines[-8:])

    if "ERROR" in text:
        return -1.0, "Error"
    if "Done" in text:
        return 1.0, "Complete"

    # Walk backwards for the most recent stage marker
    for line in reversed(lines):
        if m := re.search(r"Image (\d+)/(\d+) done", line):
            return 0.05 + 0.30 * int(m[1]) / int(m[2]), f"Generating images ({m[1]}/{m[2]})"
        if m := re.search(r"Upscaled (\d+)/(\d+)", line):
            return 0.35 + 0.05 * int(m[1]) / int(m[2]), f"Upscaling ({m[1]}/{m[2]})"
        if m := re.search(r"Voice (\d+)/(\d+) done", line):
            return 0.45 + 0.15 * int(m[1]) / int(m[2]), f"Generating voice ({m[1]}/{m[2]})"
        if m := re.search(r"Clip (\d+)/(\d+) rendered", line):
            return 0.65 + 0.12 * int(m[1]) / int(m[2]), f"Rendering clips ({m[1]}/{m[2]})"

    if any("Chaining clips" in l or "transitions" in l for l in lines[-4:]):
        return 0.82, "Adding transitions"
    if any("Intro card" in l for l in lines[-4:]):
        return 0.78, "Creating intro/outro"
    if any("SFX" in l for l in lines[-4:]):
        return 0.88, "Generating sound effects"
    if any("Mixing" in l or "music" in l for l in lines[-4:]):
        return 0.92, "Mixing audio"
    if any("Thumbnail" in l for l in lines[-4:]):
        return 0.96, "Creating thumbnail"
    if any("Final video" in l for l in lines[-4:]):
        return 0.98, "Finalizing"
    if any("Loading image" in l or "Loading Stable" in l for l in lines[-4:]):
        return 0.03, "Loading image model"
    if any("Loading Kokoro" in l for l in lines[-4:]):
        return 0.42, "Loading voice model"
    if any("Starting" in l for l in lines[-4:]):
        return 0.01, "Starting"

    return 0.01, "Working"


def _estimate_yt_progress(lines: list[str]) -> tuple[float, str]:
    """Estimate YouTube clipper progress."""
    if not lines:
        return 0.0, ""
    text = "\n".join(lines[-5:])

    if "ERROR" in text:
        return -1.0, "Error"
    if "Done" in text:
        return 1.0, "Complete"

    for line in reversed(lines):
        if m := re.search(r"Clip (\d+)/(\d+) extracted", line):
            return 0.50 + 0.35 * int(m[1]) / int(m[2]), f"Clipping ({m[1]}/{m[2]})"

    if any("Downloading" in l for l in lines[-4:]):
        return 0.10, "Downloading video"
    if any("Downloaded" in l for l in lines[-4:]):
        return 0.30, "Downloaded"
    if any("highlight" in l.lower() for l in lines[-4:]):
        return 0.40, "Identifying highlights"
    if any("Highlight reel" in l or "assembl" in l.lower() for l in lines[-4:]):
        return 0.90, "Assembling highlight reel"

    return 0.05, "Working"


# ── open folder ─────────────────────────────────────────────────────────
def open_output_folder(outdir: str):
    """Open the output folder in Windows File Explorer."""
    if not outdir:
        raise gr.Error("No output folder yet. Generate a video first.")
    p = Path(outdir)
    target = p if p.is_dir() else p.parent
    if not target.is_dir():
        target = config.OUT_ROOT
        target.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(target))
    except AttributeError:
        # Non-Windows fallback
        subprocess.Popen(["xdg-open", str(target)])
    return f"Opened: {target}"


# ── trends ──────────────────────────────────────────────────────────────
def refresh_trends():
    try:
        update_trends()
        return _trend_text("Drama")
    except Exception as e:
        return f"Could not fetch trends: {e}"

def _trend_text(label):
    gid = _to_id(label) if label else "drama_story"
    items = suggest_topics(gid, limit=5)
    if not items:
        age = cache_age_hours()
        return "No trend data yet — click **Refresh** (needs internet)." if age is None else f"Cache is {age:.0f}h old. Click **Refresh** to update."
    return "\n".join(f"- {s}" for s in items)


# ── write script (with gr.Progress) ────────────────────────────────────
def write_script(genre_label, topic, age, n_scenes, model, do_refine, progress=gr.Progress()):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Enter a topic first.")

    progress(0.05, desc="Checking Ollama connection...")
    if not _ollama_ok():
        raise gr.Error("Ollama is not reachable. Run `ollama serve` first.")

    gid = _to_id(genre_label or "Drama")
    try:
        genre = load_genre(gid)
    except FileNotFoundError:
        raise gr.Error(f"Genre '{gid}' not found.")

    config.OLLAMA_MODEL = model or config.OLLAMA_MODEL
    outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{gid}-{_slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)

    progress(0.15, desc="Generating script with LLM...")
    try:
        script = generate_script(
            topic, genre,
            n_scenes=int(n_scenes or genre.get("target_scenes", 7)),
            age=age if age != "general" else None,
            refine=bool(do_refine),
        )
    except SystemExit as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Script generation failed: {e}")

    progress(0.90, desc="Saving script...")
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    rows = [
        [s["narration"], s["image_prompt"], s.get("speaker", "narrator"), s.get("mood", "")]
        for s in script["scenes"]
    ]
    tag = " (refined)" if do_refine else ""

    progress(1.0, desc="Done")
    return (
        script.get("title", topic),
        script.get("description", ""),
        ", ".join(script.get("tags", [])),
        rows,
        str(outdir),
        f"Ready{tag} — {len(rows)} scenes",
    )


# ── render video (with progress bar) ───────────────────────────────────
def render_video(genre_label, outdir, title, desc, tags, table, speed, res, seed):
    if not outdir:
        raise gr.Error("Generate a script first.")
    gid = _to_id(genre_label or "Drama")
    try:
        genre = load_genre(gid)
    except FileNotFoundError:
        raise gr.Error(f"Genre '{gid}' not found.")

    rows = table.values.tolist() if hasattr(table, "values") else table
    scenes = [
        {
            "narration": (r[0] or "").strip(),
            "image_prompt": (r[1] or "").strip() or (r[0] or "").strip(),
            "speaker": (r[2] or "").strip() or "narrator",
            "mood": (r[3] or "").strip() or "neutral",
        }
        for r in rows if (r[0] or "").strip()
    ]
    if not scenes:
        raise gr.Error("Script table is empty.")

    outdir = Path(outdir)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    (outdir / "script.json").write_text(
        json.dumps({"title": title, "description": desc, "tags": tag_list, "scenes": scenes},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config.OUT_W, config.OUT_H = RESOLUTIONS.get(res, (1280, 720))

    q = queue.Queue()
    import pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        m.set_logger(tee)
    try:
        from pipeline.sfx import set_logger as sl1; sl1(tee)
    except Exception:
        pass
    try:
        from pipeline.thumbnail import set_logger as sl2; sl2(tee)
    except Exception:
        pass

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
                    raw_n = best.stem.removesuffix("_up") + best.suffix
                    raw = best.with_name(raw_n)
                    if raw.exists() and raw != best:
                        best = raw
                    generate_thumbnail(title, best, gid, outdir)
                except Exception as e:
                    q.put(f"Thumbnail skipped: {e}")
            (outdir / "metadata.json").write_text(
                json.dumps({
                    "title": title,
                    "description": (desc or "").strip(),
                    "tags": tag_list,
                    "genre": genre.get("name", ""),
                    "madeForKids": genre.get("made_for_kids", False),
                    "containsSyntheticMedia": True,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            q.put(f"Done — {outdir / 'final.mp4'}")
        except BaseException:
            q.put("ERROR:\n" + traceback.format_exc())
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Starting render..."]
    yield _pbar_html(0.01, "Starting"), "\n".join(lines), None
    while True:
        try:
            msg = q.get(timeout=2)   # short timeout — keeps WebSocket alive
        except queue.Empty:
            # heartbeat: re-yield current state so the connection doesn't drop
            frac, stage = _estimate_render_progress(lines)
            yield _pbar_html(frac, stage), "\n".join(lines), None
            # check if the thread is still alive; if not and queue empty, bail
            continue
        if msg is None:
            break
        lines.append(msg)
        frac, stage = _estimate_render_progress(lines)
        yield _pbar_html(frac, stage), "\n".join(lines), None

    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        m.set_logger(print)

    frac, stage = _estimate_render_progress(lines)
    yield (
        _pbar_html(frac, stage),
        "\n".join(lines),
        str(result["final"]) if "final" in result else None,
    )


# ── youtube clipper (with progress bar) ─────────────────────────────────
def clip_youtube(url, n_clips, clip_dur, model):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube URL first.")
    from pipeline.youtube import check_ytdlp, process_youtube, set_logger
    if not check_ytdlp():
        raise gr.Error("yt-dlp not found. Install: pip install yt-dlp")
    if not _ollama_ok():
        raise gr.Error("Ollama not reachable.")

    q = queue.Queue()

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    set_logger(tee)
    result = {}

    def work():
        try:
            final, _ = process_youtube(
                url, n_clips=int(n_clips), clip_duration=int(clip_dur),
                model=model or None,
            )
            result["final"] = final
            q.put(f"Done — {final}")
        except BaseException:
            q.put("ERROR:\n" + traceback.format_exc())
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Downloading and analyzing..."]
    yield _pbar_html(0.05, "Starting"), "\n".join(lines), None
    while True:
        try:
            msg = q.get(timeout=2)   # short timeout — keeps WebSocket alive
        except queue.Empty:
            frac, stage = _estimate_yt_progress(lines)
            yield _pbar_html(frac, stage), "\n".join(lines), None
            continue
        if msg is None:
            break
        lines.append(msg)
        frac, stage = _estimate_yt_progress(lines)
        yield _pbar_html(frac, stage), "\n".join(lines), None

    frac, stage = _estimate_yt_progress(lines)
    yield (
        _pbar_html(frac, stage),
        "\n".join(lines),
        str(result["final"]) if "final" in result else None,
    )


# ── genre change ────────────────────────────────────────────────────────
def on_genre_change(label):
    gid = _to_id(label or "Drama")
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
        _trend_text(label),
    )


# ════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AI Video Creator") as demo:
    outdir_state = gr.State("")

    gr.Markdown(
        "# AI Video Creator\n"
        "Generate AI videos from a topic or clip highlights from YouTube. "
        "Runs locally on your GPU."
    )

    with gr.Tabs():

        # ─── Tab 1: Create ──────────────────────────────────
        with gr.TabItem("Create Video"):

            with gr.Group():
                gr.HTML('<p class="panel-eyebrow">Genre & Topic</p>')
                genre = gr.Radio(
                    choices=_choices(), value="Drama",
                    label="Pick a genre", elem_classes="genre-pills",
                )
                gr.HTML(
                    '<div class="callout-box">'
                    '<p class="callout-label">Suggested topics</p>'
                    '</div>'
                )
                trends = gr.Markdown(_trend_text("Drama"))
                btn_trends = gr.Button("Refresh trends", variant="secondary", size="sm")

                topic = gr.Textbox(
                    label="Topic / Premise",
                    placeholder="A mother discovers her son has been lying for years...",
                    lines=2,
                )
                age = gr.Dropdown(
                    ["general"], value="general", label="Age group", visible=False,
                )
                with gr.Row():
                    scenes_sl = gr.Slider(3, 12, value=8, step=1, label="Scenes")
                    model_dd = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL, label="Model",
                        allow_custom_value=True,
                    )
                refine_cb = gr.Checkbox(
                    value=True,
                    label="Refine script (second LLM pass for better hooks)",
                )
                btn_write = gr.Button("Write Script", variant="primary")

            with gr.Group():
                gr.HTML('<p class="panel-eyebrow">Script Review</p>')
                status = gr.Markdown("", elem_classes="ok-badge")
                title_tb = gr.Textbox(label="Title", interactive=True)
                with gr.Row():
                    desc_tb = gr.Textbox(
                        label="Description", lines=2, interactive=True, scale=2,
                    )
                    tags_tb = gr.Textbox(label="Tags", interactive=True, scale=1)
                scene_table = gr.Dataframe(
                    headers=HEADERS, datatype=["str"] * 4,
                    column_count=(4, "fixed"), row_count=(8, "dynamic"),
                    interactive=True, wrap=True, label="Scenes",
                )
                with gr.Row():
                    speed_sl = gr.Slider(0.7, 1.2, value=0.85, step=0.05, label="Voice speed")
                    res_dd = gr.Dropdown(
                        list(RESOLUTIONS), value="720p", label="Resolution",
                    )
                    seed_num = gr.Number(value=1234, precision=0, label="Seed")
                btn_render = gr.Button("Render Video", variant="primary")

            with gr.Group():
                gr.HTML('<p class="panel-eyebrow">Output</p>')
                progress_bar1 = gr.HTML(_pbar_html(0, "Waiting"))
                log1 = gr.Textbox(
                    label="Log", lines=10, max_lines=14,
                    interactive=False, elem_classes="console-log",
                )
                vid1 = gr.Video(label="Video")
                btn_open1 = gr.Button(
                    "Open output folder", variant="secondary", size="sm",
                    elem_classes="open-folder-btn",
                )

        # ─── Tab 2: YouTube ─────────────────────────────────
        with gr.TabItem("YouTube Clipper"):
            with gr.Group():
                gr.HTML('<p class="panel-eyebrow">Clip highlights from YouTube</p>')
                gr.Markdown(
                    "Paste a YouTube URL. The system downloads the video, "
                    "analyzes the transcript, and assembles a highlight reel."
                )
                yt_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                with gr.Row():
                    yt_n = gr.Slider(2, 10, value=5, step=1, label="Clips")
                    yt_dur = gr.Slider(5, 30, value=15, step=5, label="Clip length (sec)")
                    yt_model = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL, label="Model",
                        allow_custom_value=True,
                    )
                btn_clip = gr.Button("Clip Highlights", variant="primary")

            with gr.Group():
                gr.HTML('<p class="panel-eyebrow">Output</p>')
                progress_bar2 = gr.HTML(_pbar_html(0, "Waiting"))
                log2 = gr.Textbox(
                    label="Log", lines=10, max_lines=14,
                    interactive=False, elem_classes="console-log",
                )
                vid2 = gr.Video(label="Highlight Reel")
                btn_open2 = gr.Button(
                    "Open output folder", variant="secondary", size="sm",
                    elem_classes="open-folder-btn",
                )

    gr.HTML(
        '<p class="app-footer">'
        'Each output folder includes thumbnail.png, metadata.json, and scene assets.'
        '</p>'
    )

    # ── wiring ──────────────────────────────────────────────
    genre.change(on_genre_change, [genre], [speed_sl, scenes_sl, age, trends])
    btn_trends.click(refresh_trends, outputs=[trends])

    btn_write.click(
        write_script,
        [genre, topic, age, scenes_sl, model_dd, refine_cb],
        [title_tb, desc_tb, tags_tb, scene_table, outdir_state, status],
    )
    btn_render.click(
        render_video,
        [genre, outdir_state, title_tb, desc_tb, tags_tb,
         scene_table, speed_sl, res_dd, seed_num],
        [progress_bar1, log1, vid1],
    )
    btn_clip.click(
        clip_youtube,
        [yt_url, yt_n, yt_dur, yt_model],
        [progress_bar2, log2, vid2],
    )
    btn_open1.click(open_output_folder, [outdir_state], [status])
    btn_open2.click(open_output_folder, [outdir_state], [status])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=1)
    demo.launch(inbrowser=True, theme=THEME, css=CSS, allowed_paths=[str(config.OUT_ROOT.resolve())])
