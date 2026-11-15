#!/usr/bin/env python3
"""
AI Comic Documentary Video Creator
────────────────────────────────────────────────────────────────────────
Focused on one niche: short-form comic-book documentary Shorts (9:16).

Run:  python app/main.py
Opens at http://127.0.0.1:7860
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

# ── path setup ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import gradio as gr

from pipeline import config
from pipeline.script   import generate_script,  set_logger as set_script_log
from pipeline.images   import generate_images,   set_logger as set_img_log
from pipeline.voice    import generate_voice,    set_logger as set_voice_log
from pipeline.assembler import assemble,         set_logger as set_asm_log
from genres.loader     import load_genre
from trends.fetcher    import update_trends, suggest_topics, cache_age_hours
import requests as _requests

# ── Fixed niche ───────────────────────────────────────────────────────────
GENRE_ID   = "comic_documentary"
# Output is always 9:16 portrait (YouTube Shorts)
OUT_W, OUT_H = 1080, 1920
IMG_W, IMG_H =  432,  768   # SD 1.5 native 9:16 (multiples of 64)

SCENE_HEADERS = ["Narration", "Image prompt", "Mood"]

# ── Voices ────────────────────────────────────────────────────────────────
VOICES = {
    "Adam (male, documentary)"  : "am_adam",
    "Michael (male, deep)"      : "am_michael",
    "Heart (female, warm)"      : "af_heart",
    "Bella (female, smooth)"    : "af_bella",
    "Nicole (female, calm)"     : "af_nicole",
    "Sarah (female, bright)"    : "af_sarah",
}


# ── Helpers ───────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:36] or "video"

def _ollama_ok() -> bool:
    try:
        r = _requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Progress bar HTML ─────────────────────────────────────────────────────
def _pbar(frac: float, stage: str) -> str:
    pct = max(0, min(100, int(frac * 100)))
    color = "#2563eb" if pct < 100 else "#16a34a"
    return (
        f'<div style="margin:6px 0 2px;font-size:12px;color:#6b7280">{stage}</div>'
        f'<div style="background:#e5e7eb;border-radius:999px;height:8px;overflow:hidden">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        f'border-radius:999px;transition:width .4s ease"></div></div>'
        f'<div style="font-size:11px;color:#9ca3af;margin-top:2px">{pct}%</div>'
    )

def _estimate_progress(lines: list[str]) -> tuple[float, str]:
    text = "\n".join(lines).lower()
    if "done —"       in text: return 1.00, "Complete ✓"
    if "thumbnail"    in text: return 0.97, "Generating thumbnail"
    if "mixing"       in text: return 0.92, "Mixing audio"
    if "chaining"     in text: return 0.86, "Chaining transitions"
    if m := re.search(r"clip (\d+)/(\d+)", text):
        n, t = int(m.group(1)), int(m.group(2))
        return 0.55 + 0.28 * (n / t), f"Rendering clip {n}/{t}"
    if m := re.search(r"voice (\d+)/(\d+)", text):
        n, t = int(m.group(1)), int(m.group(2))
        return 0.40 + 0.12 * (n / t), f"Voice {n}/{t}"
    if m := re.search(r"(?:upscaled?|image) (\d+)/(\d+)", text):
        n, t = int(m.group(1)), int(m.group(2))
        return 0.12 + 0.26 * (n / t), f"Image {n}/{t}"
    if "loading"      in text: return 0.08, "Loading model"
    if "starting"     in text: return 0.02, "Starting"
    return 0.01, "Working…"


# ── Trend suggestions ─────────────────────────────────────────────────────
def _trends_md() -> str:
    items = suggest_topics(GENRE_ID, limit=6)
    if not items:
        age = cache_age_hours()
        if age is None:
            return "_No trend data yet — click **Refresh** (needs internet)._"
        return f"_Cache is {age:.0f}h old. Click **Refresh** to update._"
    return "\n".join(f"• {s}" for s in items)

def refresh_trends() -> str:
    try:
        update_trends()
    except Exception:
        pass
    return _trends_md()


# ── 1. Write Script ───────────────────────────────────────────────────────
def write_script(
    topic: str,
    custom_prompt: str,
    character: str,
    n_scenes: int,
    voice_label: str,
    model: str,
    do_refine: bool,
    progress=gr.Progress(),
):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Enter a topic first.")

    progress(0.05, desc="Checking Ollama…")
    if not _ollama_ok():
        raise gr.Error("Ollama is not reachable. Run `ollama serve` first.")

    try:
        genre = load_genre(GENRE_ID)
    except FileNotFoundError:
        raise gr.Error(f"Genre file '{GENRE_ID}.json' not found in genres/templates/.")

    # Inject custom style and/or character block into the genre template
    if custom_prompt and custom_prompt.strip():
        genre = dict(genre)
        genre["image_style"] = custom_prompt.strip()
    if character and character.strip():
        genre = dict(genre)
        genre["character_block"] = character.strip()

    config.OLLAMA_MODEL = model or config.OLLAMA_MODEL

    outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)

    progress(0.15, desc="Writing script with LLM…")
    try:
        script = generate_script(
            topic, genre,
            n_scenes=int(n_scenes),
            age=None,
            refine=bool(do_refine),
        )
    except SystemExit as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Script generation failed: {e}")

    progress(0.95, desc="Saving…")
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = [
        [s["narration"], s["image_prompt"], s.get("mood", "curious")]
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
        f"✓ {len(rows)} scenes ready{tag}",
    )


# ── 2. Render Video ───────────────────────────────────────────────────────
def render_video(
    outdir_str:    str,
    title:         str,
    desc:          str,
    tags:          str,
    table,
    custom_prompt: str,
    character:     str,
    voice_label:   str,
    speed:         float,
    seed:          int,
    use_anim:      bool,
    # caption settings
    cap_font:      str,
    cap_size:      int,
    cap_color:     str,
    cap_bold:      bool,
    cap_style:     str,
    cap_pos:       str,
):
    if not outdir_str:
        raise gr.Error("Write the script first (Step 1).")

    try:
        genre = load_genre(GENRE_ID)
    except FileNotFoundError:
        raise gr.Error(f"Genre file '{GENRE_ID}.json' not found.")

    if custom_prompt and custom_prompt.strip():
        genre = dict(genre)
        genre["image_style"] = custom_prompt.strip()
    if character and character.strip():
        genre = dict(genre)
        genre["character_block"] = character.strip()

    rows   = table.values.tolist() if hasattr(table, "values") else (table or [])
    scenes = [
        {
            "narration":    (r[0] or "").strip(),
            "image_prompt": (r[1] or "").strip() or (r[0] or "").strip(),
            "speaker":      "narrator",
            "mood":         (r[2] or "").strip() or "curious",
        }
        for r in rows if (r[0] or "").strip()
    ]
    if not scenes:
        raise gr.Error("Scene table is empty. Write the script first.")

    outdir   = Path(outdir_str)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    voice_id = VOICES.get(voice_label, "am_adam")

    # Override voice in genre so generate_voice() picks it up
    genre = dict(genre)
    genre["voice_map"] = {"narrator": voice_id}

    (outdir / "script.json").write_text(
        json.dumps({"title": title, "description": desc,
                    "tags": tag_list, "scenes": scenes},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    q = queue.Queue()
    import pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        m.set_logger(tee)
    for mod_name in ("pipeline.sfx", "pipeline.thumbnail", "pipeline.animate"):
        try:
            mod = __import__(mod_name, fromlist=["set_logger"])
            mod.set_logger(tee)
        except Exception:
            pass

    result: dict = {}

    def work():
        try:
            if use_anim:
                from pipeline.animate import generate_animated_clips, set_logger as _sl
                _sl(tee)
                imgs = generate_animated_clips(
                    scenes, genre, outdir, int(seed),
                    img_w=IMG_W, img_h=IMG_H,
                )
            else:
                imgs = generate_images(
                    scenes, genre, outdir, int(seed),
                    img_w=IMG_W, img_h=IMG_H,
                )

            auds, durs = generate_voice(scenes, genre, outdir, float(speed))

            result["final"] = assemble(
                scenes, imgs, auds, durs, genre, outdir,
                title=title,
                out_w=OUT_W, out_h=OUT_H,
                caption_font=cap_font   or "Arial",
                caption_size=int(cap_size or 0),
                caption_color=cap_color or "white",
                caption_bold=bool(cap_bold),
                caption_style=cap_style or "box",
                caption_position=cap_pos or "bottom",
            )

            # Thumbnail
            if config.GENERATE_THUMBNAIL and imgs:
                try:
                    from pipeline.thumbnail import generate_thumbnail, set_logger as _tl
                    _tl(tee)
                    best = imgs[min(1, len(imgs) - 1)]
                    raw  = best.with_name(best.stem.removesuffix("_up") + best.suffix)
                    generate_thumbnail(title, raw if raw.exists() else best,
                                       GENRE_ID, outdir)
                except Exception as e:
                    tee(f"Thumbnail skipped: {e}")

            (outdir / "metadata.json").write_text(
                json.dumps({
                    "title":                 title,
                    "description":           (desc or "").strip(),
                    "tags":                  tag_list,
                    "genre":                 "Comic Documentary",
                    "madeForKids":           False,
                    "containsSyntheticMedia": True,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tee(f"Done — {outdir / 'final.mp4'}")
        except BaseException:
            tee("ERROR:\n" + traceback.format_exc())
        finally:
            tee(None)   # sentinel

    threading.Thread(target=work, daemon=True).start()

    lines = ["Starting render…"]
    yield _pbar(0.01, "Starting"), "\n".join(lines), None

    while True:
        try:
            msg = q.get(timeout=2)
        except queue.Empty:
            frac, stage = _estimate_progress(lines)
            yield _pbar(frac, stage), "\n".join(lines), None
            continue
        if msg is None:
            break
        lines.append(msg)
        frac, stage = _estimate_progress(lines)
        yield _pbar(frac, stage), "\n".join(lines), None

    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        m.set_logger(print)

    frac, stage = _estimate_progress(lines)
    yield (
        _pbar(frac, stage),
        "\n".join(lines),
        str(result["final"]) if "final" in result else None,
    )


# ── 3. YouTube Clipper ────────────────────────────────────────────────────
def clip_youtube(url: str, n_clips: int, clip_dur: int, model: str):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Paste a YouTube URL first.")
    from pipeline.youtube import check_ytdlp, process_youtube, set_logger as _yl
    if not check_ytdlp():
        raise gr.Error("yt-dlp not found. Install: pip install yt-dlp")
    if not _ollama_ok():
        raise gr.Error("Ollama is not reachable. Run `ollama serve` first.")

    q: queue.Queue = queue.Queue()
    result: dict   = {}

    def tee(msg):
        print(msg, flush=True)
        q.put(msg)

    _yl(tee)

    def work():
        try:
            final, _ = process_youtube(
                url, n_clips=int(n_clips), clip_duration=int(clip_dur),
                model=model or None,
            )
            result["final"] = final
            tee(f"Done — {final}")
        except BaseException:
            tee("ERROR:\n" + traceback.format_exc())
        finally:
            tee(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Downloading and analyzing…"]
    yield _pbar(0.05, "Starting"), "\n".join(lines), None

    while True:
        try:
            msg = q.get(timeout=2)
        except queue.Empty:
            yield _pbar(0.10, "Processing…"), "\n".join(lines), None
            continue
        if msg is None:
            break
        lines.append(msg)
        yield _pbar(0.50, "Clipping"), "\n".join(lines), None

    yield (
        _pbar(1.0 if "final" in result else 0.5, "Done" if "final" in result else "Error"),
        "\n".join(lines),
        str(result["final"]) if "final" in result else None,
    )


# ── Open folder helper ────────────────────────────────────────────────────
def open_folder(outdir_str: str):
    p = Path(outdir_str) if outdir_str else config.OUT_ROOT
    target = p if p.is_dir() else p.parent
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(target))
    except AttributeError:
        subprocess.Popen(["xdg-open", str(target)])
    return f"Opened: {target}"


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0f172a",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    block_border_width="1px",
    container_radius="12px",
    block_shadow="0 4px 6px -1px rgba(0,0,0,0.3)",
    block_title_text_color="#f1f5f9",
    block_title_text_size="14px",
    block_title_text_weight="600",
    block_label_text_color="#94a3b8",
    block_label_text_size="12px",
    input_background_fill="#0f172a",
    input_border_color="#334155",
    input_border_width="1px",
    input_radius="8px",
    input_text_size="14px",
    input_shadow="none",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_primary_shadow="none",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_hover="#334155",
    button_secondary_text_color="#94a3b8",
    button_secondary_border_color="#334155",
    button_secondary_shadow="none",
    button_border_width="1px",
    button_large_radius="8px",
    button_large_text_size="14px",
    button_large_text_weight="600",
    layout_gap="16px",
    section_header_text_size="11px",
    section_header_text_weight="600",
)

CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }

.app-header { text-align:center; padding: 28px 0 20px; }
.app-header h1 { font-size:26px; font-weight:700; color:#f1f5f9; margin:0 0 6px; }
.app-header p  { font-size:13px; color:#64748b; margin:0; }

.section-label {
  font-size:11px; font-weight:600; color:#3b82f6;
  text-transform:uppercase; letter-spacing:.06em;
  margin: 0 0 10px 2px;
}
.hint {
  font-size:12px; color:#64748b;
  background:#1e293b; border:1px solid #334155;
  border-radius:8px; padding:10px 14px; margin-bottom:4px;
  line-height:1.6;
}

/* dark console log */
.console textarea {
  background:#0a0f1a !important; color:#a3e635 !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:12px !important; line-height:1.7 !important;
  border:none !important; border-radius:8px !important;
  padding:14px !important;
}

/* status badge */
.status span {
  display:inline-block; background:#1e3a5f; color:#60a5fa;
  border-radius:999px; padding:4px 14px;
  font-size:12px; font-weight:500;
}

/* open folder small btn */
.small-btn button { font-size:12px !important; padding:6px 14px !important; }

/* progress bar container */
.pbar-wrap { margin: 4px 0 8px; }
"""

# ── Build UI ───────────────────────────────────────────────────────────────
with gr.Blocks(title="Comic Doc Studio") as demo:

    outdir_state = gr.State("")

    gr.HTML(
        '<div class="app-header">'
        '<h1>🎬 Comic Documentary Studio</h1>'
        '<p>Topic → Script → Comic Images → Voiceover → YouTube Short (9:16)</p>'
        '</div>'
    )

    with gr.Tabs():

        # ═══════════════ TAB 1 : CREATE ════════════════════════════════
        with gr.TabItem("Create Video"):

            # ── STEP 1 ─────────────────────────────────────────────────
            with gr.Group():
                gr.HTML('<p class="section-label">Step 1 — Topic & Style</p>')

                topic_tb = gr.Textbox(
                    label="Topic",
                    placeholder='e.g. "Why the Roman Empire really fell"  or  "The secret life of Tesla"',
                    lines=2,
                )

                custom_prompt_tb = gr.Textbox(
                    label="Custom Visual Style  (optional — overrides default)",
                    placeholder=(
                        "Leave blank to use the default hybrid style.\n"
                        "Or paste your own style description here."
                    ),
                    lines=3,
                    info="Overrides the image_style field for every scene.",
                )

                character_tb = gr.Textbox(
                    label="Narrator Character  (optional — overrides default)",
                    placeholder=(
                        "Leave blank to use the default narrator:\n"
                        "Round white head, dot eyes, yellow beanie, teal shirt, stick arms.\n\n"
                        "Or describe your own character here — this exact text is injected\n"
                        "into every single image prompt to keep the character consistent."
                    ),
                    lines=5,
                    info="This is injected word-for-word into every image prompt.",
                )

                gr.HTML(
                    '<div class="hint">'
                    '<b>Default style:</b> flat 2D cartoon narrator in a detailed painterly 3D environment. '
                    'Warm cinematic lighting, deadpan dry humor tone, 9:16 portrait.<br><br>'
                    '<b>Default narrator:</b> round white head · dot eyes · yellow beanie · teal shirt · stick arms · deadpan expression.'
                    '</div>'
                )

                with gr.Row():
                    n_scenes_sl = gr.Slider(5, 14, value=8, step=1,
                                            label="Scenes (more = longer video)")
                    voice_dd = gr.Dropdown(
                        list(VOICES.keys()),
                        value="Adam (male, documentary)",
                        label="Narrator voice",
                    )

                with gr.Row():
                    model_dd = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL,
                        label="LLM model",
                        allow_custom_value=True,
                    )
                    refine_cb = gr.Checkbox(
                        value=True,
                        label="Refine script (second LLM pass — better hooks & pacing)",
                    )

                # Trend suggestions
                with gr.Accordion("💡 Trending topic suggestions", open=False):
                    trends_md = gr.Markdown(_trends_md())
                    gr.Button("Refresh trends", variant="secondary", size="sm"
                              ).click(refresh_trends, outputs=[trends_md])

                btn_write = gr.Button("1. Write Script", variant="primary", size="lg")

            # ── STEP 2 ─────────────────────────────────────────────────
            with gr.Group():
                gr.HTML('<p class="section-label">Step 2 — Review & Edit Script</p>')
                status_md = gr.Markdown("", elem_classes=["status"])
                title_tb  = gr.Textbox(label="Video title", interactive=True)
                with gr.Row():
                    desc_tb = gr.Textbox(label="Description", lines=2,
                                         interactive=True, scale=2)
                    tags_tb = gr.Textbox(label="Tags (comma separated)",
                                         interactive=True, scale=1)

                scene_table = gr.Dataframe(
                    headers=SCENE_HEADERS, datatype=["str"] * 3,
                    column_count=(3, "fixed"), row_count=(8, "dynamic"),
                    interactive=True, wrap=True, label="Scenes",
                )
                gr.HTML(
                    '<div class="hint">'
                    '<b>Tip:</b> Edit the <i>Narration</i> and <i>Image prompt</i> directly in the table. '
                    'The image prompt drives what SD 1.5 draws for each scene — be specific about '
                    'camera angle, expressions, lighting, and setting.'
                    '</div>'
                )

            # ── STEP 3 ─────────────────────────────────────────────────
            with gr.Group():
                gr.HTML('<p class="section-label">Step 3 — Render Settings</p>')

                with gr.Row():
                    speed_sl  = gr.Slider(0.75, 1.15, value=0.90, step=0.05,
                                          label="Voice speed")
                    seed_num  = gr.Number(value=1234, precision=0,
                                          label="Image seed  (change for different visuals)")

                use_anim_cb = gr.Checkbox(
                    value=False,
                    label="Animated scenes (AnimateDiff-Lightning)",
                    info=(
                        "Generates moving clips instead of still images. "
                        "Needs ~3 GB free RAM. First run downloads ~180 MB."
                    ),
                )

                with gr.Accordion("Caption settings", open=False):
                    with gr.Row():
                        cap_font_dd  = gr.Dropdown(
                            ["Arial", "Impact", "Verdana", "Georgia",
                             "Trebuchet MS", "Comic Sans MS", "Courier New"],
                            value="Arial", label="Font",
                        )
                        cap_size_num = gr.Number(value=0, precision=0,
                                                  label="Size (0 = auto 26pt)",
                                                  info="0 = auto")
                        cap_bold_cb  = gr.Checkbox(value=True, label="Bold")
                    with gr.Row():
                        cap_color_dd = gr.Dropdown(
                            ["white", "yellow", "cyan", "orange",
                             "pink", "green", "black", "red"],
                            value="white", label="Color",
                        )
                        cap_style_dd = gr.Dropdown(
                            ["box", "outline", "shadow", "plain"],
                            value="box", label="Style",
                        )
                        cap_pos_dd   = gr.Dropdown(
                            ["bottom", "top", "middle"],
                            value="bottom", label="Position",
                        )

                btn_render = gr.Button("2. Render Video", variant="primary", size="lg")

            # ── OUTPUT ─────────────────────────────────────────────────
            with gr.Group():
                gr.HTML('<p class="section-label">Output</p>')
                pbar1    = gr.HTML(_pbar(0, "Waiting…"), elem_classes=["pbar-wrap"])
                log1     = gr.Textbox(label="Progress log", lines=12, max_lines=16,
                                      interactive=False, elem_classes=["console"])
                vid1     = gr.Video(label="Final video  (9:16 portrait)")
                btn_open1 = gr.Button("Open output folder",
                                      variant="secondary", size="sm",
                                      elem_classes=["small-btn"])

        # ═══════════════ TAB 2 : YOUTUBE CLIPPER ═══════════════════════
        with gr.TabItem("YouTube Clipper"):

            with gr.Group():
                gr.HTML('<p class="section-label">Clip highlights from a YouTube video</p>')
                gr.HTML(
                    '<div class="hint">'
                    'Paste a YouTube URL. The system downloads the video, runs the transcript '
                    'through the local LLM to find the best moments, clips them, and assembles '
                    'a highlight reel.'
                    '</div>'
                )
                yt_url_tb = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                with gr.Row():
                    yt_n_sl   = gr.Slider(2, 10, value=5, step=1, label="Number of clips")
                    yt_dur_sl = gr.Slider(5, 30, value=15, step=5, label="Clip length (sec)")
                    yt_mdl_dd = gr.Dropdown(
                        ["qwen2.5:7b", "llama3.2:3b"],
                        value=config.OLLAMA_MODEL,
                        label="LLM model",
                        allow_custom_value=True,
                    )
                btn_clip = gr.Button("Clip Highlights", variant="primary")

            with gr.Group():
                gr.HTML('<p class="section-label">Output</p>')
                pbar2     = gr.HTML(_pbar(0, "Waiting…"), elem_classes=["pbar-wrap"])
                log2      = gr.Textbox(label="Progress log", lines=10, max_lines=14,
                                       interactive=False, elem_classes=["console"])
                vid2      = gr.Video(label="Highlight reel")
                btn_open2 = gr.Button("Open output folder",
                                      variant="secondary", size="sm",
                                      elem_classes=["small-btn"])

    gr.HTML(
        '<p style="text-align:center;font-size:11px;color:#334155;padding:20px 0 8px">'
        'Output folder contains final.mp4 · thumbnail.png · metadata.json · script.json'
        '</p>'
    )

    # ── Wiring ─────────────────────────────────────────────────────────
    btn_write.click(
        write_script,
        inputs=[topic_tb, custom_prompt_tb, character_tb, n_scenes_sl,
                voice_dd, model_dd, refine_cb],
        outputs=[title_tb, desc_tb, tags_tb, scene_table, outdir_state, status_md],
    )

    btn_render.click(
        render_video,
        inputs=[
            outdir_state, title_tb, desc_tb, tags_tb, scene_table,
            custom_prompt_tb, character_tb, voice_dd, speed_sl, seed_num, use_anim_cb,
            cap_font_dd, cap_size_num, cap_color_dd, cap_bold_cb,
            cap_style_dd, cap_pos_dd,
        ],
        outputs=[pbar1, log1, vid1],
    )

    btn_clip.click(
        clip_youtube,
        inputs=[yt_url_tb, yt_n_sl, yt_dur_sl, yt_mdl_dd],
        outputs=[pbar2, log2, vid2],
    )

    btn_open1.click(open_folder, inputs=[outdir_state], outputs=[status_md])
    btn_open2.click(open_folder, inputs=[outdir_state], outputs=[status_md])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=1)
    demo.launch(
        inbrowser=True,
        theme=THEME,
        css=CSS,
        allowed_paths=[str(config.OUT_ROOT.resolve())],
    )
