#!/usr/bin/env python3
"""
AI Video Creator — polished responsive UI with all features wired.

Run:  python app/main.py    →  http://127.0.0.1:7860
"""
import json, os, queue, re, subprocess, sys, threading, time, traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import gradio as gr
from pipeline import config
from pipeline.script import generate_script, set_logger as set_script_log
from pipeline.images import generate_images, set_logger as set_img_log
from pipeline.voice import generate_voice, set_logger as set_voice_log
from pipeline.assembler import assemble, set_logger as set_asm_log
from genres.loader import load_genre, genre_ids
from trends.fetcher import update_trends, suggest_topics, cache_age_hours
import requests as _req

# ── Genre + Voice maps ────────────────────────────────────────────────
VOICES = {
    "Adam (male, documentary)": "am_adam",
    "Michael (male, deep)": "am_michael",
    "Heart (female, warm)": "af_heart",
    "Bella (female, smooth)": "af_bella",
    "Nicole (female, calm)": "af_nicole",
    "Sarah (female, bright)": "af_sarah",
}
FORMATS = {"Shorts 9:16": (1080, 1920, 432, 768), "Landscape 16:9": (1280, 720, 768, 432)}

def _genre_choices(): return genre_ids()
def _slugify(t): return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:36] or "video"
def _ollama_ok():
    try: return _req.get("http://localhost:11434/api/tags", timeout=3).ok
    except Exception: return False

# ── Progress ──────────────────────────────────────────────────────────
def _pbar(frac, stage):
    pct = max(0, min(100, int(frac * 100)))
    c = "#2563eb" if pct < 100 else "#16a34a"
    return (f'<div style="font-size:12px;color:#94a3b8;margin:4px 0 2px">{stage}</div>'
            f'<div style="background:#1e293b;border-radius:99px;height:8px;overflow:hidden">'
            f'<div style="background:{c};height:100%;width:{pct}%;border-radius:99px;'
            f'transition:width .4s"></div></div>'
            f'<div style="font-size:11px;color:#475569;margin-top:2px;text-align:right">{pct}%</div>')

def _est(lines):
    t = "\n".join(lines).lower()
    if "done" in t and ("—" in t or "final" in t): return 1.0, "Complete"
    if "loudness" in t: return 0.95, "Loudness normalization"
    if "upload package" in t: return 0.93, "Upload package"
    if "thumbnail" in t: return 0.92, "Thumbnail"
    if "mixing" in t: return 0.88, "Mixing audio"
    if "chaining" in t or "transition" in t: return 0.82, "Transitions"
    if m := re.search(r"clip (\d+)/(\d+)", t): return 0.55 + 0.25 * int(m[1]) / int(m[2]), f"Clip {m[1]}/{m[2]}"
    if m := re.search(r"voice (\d+)/(\d+)", t): return 0.40 + 0.12 * int(m[1]) / int(m[2]), f"Voice {m[1]}/{m[2]}"
    if m := re.search(r"(?:upscale|image) (\d+)/(\d+)", t): return 0.10 + 0.28 * int(m[1]) / int(m[2]), f"Image {m[1]}/{m[2]}"
    if "research" in t: return 0.04, "Researching topic"
    if "loading" in t: return 0.06, "Loading model"
    if "starting" in t or "render" in t: return 0.02, "Starting"
    return 0.01, "Working"

# ── Trends ────────────────────────────────────────────────────────────
def _trends(gid):
    items = suggest_topics(gid, limit=5)
    if not items:
        age = cache_age_hours()
        return "_No trends cached. Click Refresh._" if age is None else f"_Cache {age:.0f}h old._"
    return "\n".join(f"- {s}" for s in items)
def refresh_trends(gid):
    try: update_trends()
    except Exception: pass
    return _trends(gid)

# ── Genre change ──────────────────────────────────────────────────────
def on_genre(gid):
    try: g = load_genre(gid)
    except FileNotFoundError: return [gr.update()] * 4
    fmt = g.get("default_format", "Shorts 9:16")
    return (
        gr.update(value=g.get("voice_speed", 0.90)),
        gr.update(value=g.get("target_scenes", 8)),
        gr.update(value=fmt if fmt in FORMATS else "Shorts 9:16"),
        _trends(gid),
    )

# ── Write Script ──────────────────────────────────────────────────────
def write_script(gid, topic, custom_style, character, n_scenes, model, do_refine, do_research, progress=gr.Progress()):
    topic = (topic or "").strip()
    if not topic: raise gr.Error("Enter a topic first.")
    progress(0.05, desc="Checking Ollama")
    if not _ollama_ok(): raise gr.Error("Ollama not reachable. Run `ollama serve`.")
    try: genre = load_genre(gid)
    except FileNotFoundError: raise gr.Error(f"Genre '{gid}' not found.")

    genre = dict(genre)
    if custom_style and custom_style.strip(): genre["image_style"] = custom_style.strip()
    if character and character.strip(): genre["character_block"] = character.strip()
    config.OLLAMA_MODEL = model or config.OLLAMA_MODEL

    outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{gid}-{_slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional research
    if do_research:
        progress(0.10, desc="Researching topic")
        try:
            from pipeline.research import research_topic
            brief = research_topic(topic, model=model)
            if brief: (outdir / "research_brief.txt").write_text(brief, encoding="utf-8")
        except Exception: pass

    progress(0.20, desc="Writing script")
    try:
        script = generate_script(topic, genre, n_scenes=int(n_scenes), refine=bool(do_refine))
    except SystemExit as e: raise gr.Error(str(e))
    except Exception as e: raise gr.Error(f"Script failed: {e}")

    progress(0.95, desc="Saving")
    (outdir / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [[s["narration"], s["image_prompt"], s.get("mood", "curious")] for s in script["scenes"]]
    tag = " (refined)" if do_refine else ""
    progress(1.0, desc="Done")
    return (script.get("title", topic), script.get("description", ""), ", ".join(script.get("tags", [])),
            rows, str(outdir), f"{len(rows)} scenes ready{tag}")

# ── Render Video ──────────────────────────────────────────────────────
def render_video(gid, outdir_str, title, desc, tags, table, custom_style, character,
                 voice_label, speed, seed, use_anim, fmt,
                 cap_font, cap_size, cap_color, cap_bold, cap_style, cap_pos, use_karaoke):
    if not outdir_str: raise gr.Error("Write the script first.")
    try: genre = load_genre(gid)
    except FileNotFoundError: raise gr.Error(f"Genre '{gid}' not found.")

    genre = dict(genre)
    if custom_style and custom_style.strip(): genre["image_style"] = custom_style.strip()
    if character and character.strip(): genre["character_block"] = character.strip()
    genre["voice_map"] = {"narrator": VOICES.get(voice_label, "am_adam")}

    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    scenes = [{"narration": (r[0] or "").strip(), "image_prompt": (r[1] or "").strip() or (r[0] or "").strip(),
               "speaker": "narrator", "mood": (r[2] or "").strip() or "curious"}
              for r in rows if (r[0] or "").strip()]
    if not scenes: raise gr.Error("Scene table is empty.")

    outdir = Path(outdir_str)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    ow, oh, iw, ih = FORMATS.get(fmt, (1080, 1920, 432, 768))

    (outdir / "script.json").write_text(json.dumps({"title": title, "description": desc, "tags": tag_list, "scenes": scenes}, indent=2, ensure_ascii=False), encoding="utf-8")

    q = queue.Queue()
    import pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler
    def tee(msg):
        if msg is not None: print(msg, flush=True)
        q.put(msg)
    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler):
        m.set_logger(tee)
    for mn in ("pipeline.sfx", "pipeline.thumbnail", "pipeline.animate", "pipeline.orchestrator", "pipeline.research"):
        try: __import__(mn, fromlist=["set_logger"]).set_logger(tee)
        except Exception: pass

    result = {}
    def work():
        try:
            from pipeline.orchestrator import PipelineRun, set_logger as sl
            sl(tee)
            run = PipelineRun(outdir=outdir, scenes=scenes, genre=genre, title=title,
                              seed=int(seed), speed=float(speed), out_w=ow, out_h=oh,
                              img_w=iw, img_h=ih, use_anim=use_anim,
                              caption_font=cap_font or "Arial", caption_size=int(cap_size or 0),
                              caption_color=cap_color or "white", caption_bold=bool(cap_bold),
                              caption_style=cap_style or "box", caption_position=cap_pos or "bottom")
            resume = run.last_failed_stage if run.can_resume else None
            if resume: tee(f"Resuming from '{resume}'")
            result["final"] = run.execute(resume_from=resume)

            # Karaoke captions (post-process if enabled)
            if use_karaoke and result.get("final"):
                try:
                    from pipeline.karaoke import generate_karaoke_ass, build_karaoke_filter
                    ass_path = generate_karaoke_ass(scenes, run.durations, outdir / "karaoke.ass",
                                                    font=cap_font or "Arial", font_size=int(cap_size or 26),
                                                    primary_color="#FFFFFF", highlight_color="#FFD700",
                                                    position=cap_pos or "bottom", is_vertical=oh > ow,
                                                    scene_pad=config.SCENE_PAD, transition_dur=config.TRANSITION_DUR)
                    tee("Karaoke captions generated")
                except Exception as e: tee(f"Karaoke skipped: {e}")

            if config.GENERATE_THUMBNAIL and run.images:
                try:
                    from pipeline.thumbnail import generate_thumbnail
                    best = run.images[min(1, len(run.images) - 1)]
                    raw = best.with_name(best.stem.removesuffix("_up") + best.suffix)
                    generate_thumbnail(title, raw if raw.exists() else best, gid, outdir)
                except Exception as e: tee(f"Thumbnail skipped: {e}")

            (outdir / "metadata.json").write_text(json.dumps({"title": title, "description": (desc or "").strip(),
                "tags": tag_list, "genre": genre.get("name", ""), "madeForKids": genre.get("made_for_kids", False),
                "containsSyntheticMedia": True}, indent=2, ensure_ascii=False), encoding="utf-8")
            tee(f"Done — {outdir / 'final.mp4'}")
        except BaseException: tee("ERROR:\n" + traceback.format_exc())
        finally: tee(None)

    threading.Thread(target=work, daemon=True).start()
    lines = ["Starting render..."]
    yield _pbar(0.01, "Starting"), "\n".join(lines), None
    while True:
        try: msg = q.get(timeout=2)
        except queue.Empty:
            f, s = _est(lines); yield _pbar(f, s), "\n".join(lines), None; continue
        if msg is None: break
        lines.append(msg); f, s = _est(lines); yield _pbar(f, s), "\n".join(lines), None
    for m in (pipeline.script, pipeline.images, pipeline.voice, pipeline.assembler): m.set_logger(print)
    f, s = _est(lines)
    yield _pbar(f, s), "\n".join(lines), str(result["final"]) if "final" in result else None

# ── YouTube Clipper ───────────────────────────────────────────────────
def clip_youtube(url, n_clips, clip_dur, model):
    url = (url or "").strip()
    if not url: raise gr.Error("Paste a YouTube URL.")
    from pipeline.youtube import check_ytdlp, process_youtube, set_logger as yl
    if not check_ytdlp(): raise gr.Error("yt-dlp not found. pip install yt-dlp")
    if not _ollama_ok(): raise gr.Error("Ollama not reachable.")
    q = queue.Queue(); result = {}
    def tee(msg): print(msg, flush=True) if msg else None; q.put(msg)
    yl(tee)
    def work():
        try:
            final, _ = process_youtube(url, n_clips=int(n_clips), clip_duration=int(clip_dur), model=model or None)
            result["final"] = final; tee(f"Done — {final}")
        except BaseException: tee("ERROR:\n" + traceback.format_exc())
        finally: tee(None)
    threading.Thread(target=work, daemon=True).start()
    lines = ["Downloading and analyzing..."]; yield _pbar(0.05, "Starting"), "\n".join(lines), None
    while True:
        try: msg = q.get(timeout=2)
        except queue.Empty: yield _pbar(0.15, "Processing"), "\n".join(lines), None; continue
        if msg is None: break
        lines.append(msg); yield _pbar(0.50, "Clipping"), "\n".join(lines), None
    yield (_pbar(1.0 if "final" in result else 0.5, "Done" if "final" in result else "Error"),
           "\n".join(lines), str(result["final"]) if "final" in result else None)

def open_folder(d):
    p = Path(d) if d else config.OUT_ROOT; t = p if p.is_dir() else p.parent; t.mkdir(parents=True, exist_ok=True)
    try: os.startfile(str(t))
    except AttributeError: subprocess.Popen(["xdg-open", str(t)])
    return f"Opened {t}"

# ══════════════════════════════════════════════════════════════════════
# THEME + CSS
# ══════════════════════════════════════════════════════════════════════
try:
    THEME = gr.themes.Base(
        primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"), font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0f172a", body_text_color="#e2e8f0", body_text_color_subdued="#94a3b8",
        block_background_fill="#1e293b", block_border_color="#334155", block_border_width="1px",
        block_radius="12px", block_shadow="0 2px 8px rgba(0,0,0,0.25)",
        block_title_text_color="#f1f5f9", block_title_text_size="14px", block_title_text_weight="600",
        block_label_text_color="#94a3b8", block_label_text_size="12px",
        input_background_fill="#0f172a", input_border_color="#334155", input_border_width="1px",
        input_radius="8px", input_text_size="14px", input_shadow="none",
        button_primary_background_fill="#2563eb", button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#fff", button_primary_border_color="transparent",
        button_secondary_background_fill="#1e293b", button_secondary_background_fill_hover="#334155",
        button_secondary_text_color="#94a3b8", button_secondary_border_color="#334155",
        button_border_width="1px", button_shadow="none",
        button_large_radius="8px", button_large_text_size="14px", button_large_text_weight="600",
        layout_gap="14px",
    )
except Exception:
    THEME = gr.themes.Base()

CSS = """
/* ── responsive container ── */
.gradio-container { max-width: 960px !important; margin: 0 auto !important; padding: 0 12px !important; }
@media (max-width: 768px) {
  .gradio-container { padding: 0 6px !important; }
  .gr-row { flex-direction: column !important; }
  .gr-row > * { min-width: 100% !important; }
}

/* ── header ── */
.hdr { text-align:center; padding:24px 0 16px; }
.hdr h1 { font-size:24px; font-weight:700; color:#f1f5f9; margin:0 0 4px; letter-spacing:-0.02em; }
.hdr p { font-size:13px; color:#64748b; margin:0; }

/* ── section labels ── */
.sl { font-size:11px; font-weight:600; color:#3b82f6; text-transform:uppercase; letter-spacing:.06em; margin:0 0 8px 2px; }

/* ── hint box ── */
.hnt { font-size:12px; color:#64748b; background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px 14px; margin-bottom:4px; line-height:1.6; }

/* ── console ── */
.con textarea { background:#0a0f1a !important; color:#a3e635 !important; font-family:'JetBrains Mono',monospace !important; font-size:11px !important; line-height:1.7 !important; border:none !important; border-radius:8px !important; padding:12px !important; }

/* ── status pill ── */
.st span { display:inline-block; background:#1e3a5f; color:#60a5fa; border-radius:99px; padding:4px 14px; font-size:12px; font-weight:500; }

/* ── small buttons ── */
.sm button { font-size:12px !important; padding:6px 14px !important; }

/* ── progress ── */
.pb { margin:4px 0 8px; }

/* ── footer ── */
.ft { text-align:center; font-size:11px; color:#334155; padding:16px 0 8px; }
"""

# ══════════════════════════════════════════════════════════════════════
# BUILD UI
# ══════════════════════════════════════════════════════════════════════
with gr.Blocks(title="AI Video Creator", theme=THEME, css=CSS) as demo:
    outdir_state = gr.State("")

    gr.HTML('<div class="hdr"><h1>AI Video Creator</h1><p>Script → Images → Voice → Video. All local, all genres.</p></div>')

    with gr.Tabs():

        # ═══ TAB 1: CREATE ════════════════════════════════════════════
        with gr.TabItem("Create Video"):

            with gr.Group():
                gr.HTML('<p class="sl">Genre & Topic</p>')
                with gr.Row():
                    genre_dd = gr.Dropdown(choices=_genre_choices(), value="comic_documentary", label="Genre", scale=1)
                    fmt_dd = gr.Dropdown(list(FORMATS.keys()), value="Shorts 9:16", label="Format", scale=1)
                topic_tb = gr.Textbox(label="Topic", placeholder="Why the Roman Empire really fell...", lines=2)

                with gr.Accordion("Style & Character (optional)", open=False):
                    custom_style_tb = gr.Textbox(label="Custom visual style", lines=3, placeholder="Leave blank for genre default.")
                    character_tb = gr.Textbox(label="Narrator character", lines=3, placeholder="Leave blank for genre default.")

                with gr.Row():
                    n_scenes_sl = gr.Slider(4, 14, value=8, step=1, label="Scenes")
                    voice_dd = gr.Dropdown(list(VOICES.keys()), value="Adam (male, documentary)", label="Voice")
                with gr.Row():
                    model_dd = gr.Dropdown(["qwen2.5:7b", "llama3.2:3b"], value=config.OLLAMA_MODEL, label="Model", allow_custom_value=True)
                    speed_sl = gr.Slider(0.75, 1.15, value=0.90, step=0.05, label="Voice speed")

                with gr.Row():
                    refine_cb = gr.Checkbox(value=True, label="Refine script")
                    research_cb = gr.Checkbox(value=False, label="Research topic first")
                    use_anim_cb = gr.Checkbox(value=False, label="Animated (AnimateDiff)")

                with gr.Accordion("Trending topics", open=False):
                    trends_md = gr.Markdown(_trends("comic_documentary"))
                    gr.Button("Refresh", variant="secondary", size="sm").click(
                        refresh_trends, [genre_dd], [trends_md])

                btn_write = gr.Button("Write Script", variant="primary", size="lg")

            with gr.Group():
                gr.HTML('<p class="sl">Script Review</p>')
                status_md = gr.Markdown("", elem_classes=["st"])
                title_tb = gr.Textbox(label="Title", interactive=True)
                with gr.Row():
                    desc_tb = gr.Textbox(label="Description", lines=2, interactive=True, scale=2)
                    tags_tb = gr.Textbox(label="Tags", interactive=True, scale=1)
                scene_table = gr.Dataframe(headers=["Narration", "Image prompt", "Mood"],
                    datatype=["str"]*3,
                    interactive=True, wrap=True, label="Scenes")

            with gr.Group():
                gr.HTML('<p class="sl">Render Settings</p>')
                seed_num = gr.Number(value=1234, precision=0, label="Image seed")
                with gr.Accordion("Caption settings", open=False):
                    with gr.Row():
                        cap_font = gr.Dropdown(["Arial","Impact","Verdana","Georgia"], value="Arial", label="Font")
                        cap_size = gr.Number(value=0, precision=0, label="Size (0=auto)")
                        cap_bold = gr.Checkbox(value=True, label="Bold")
                    with gr.Row():
                        cap_color = gr.Dropdown(["white","yellow","cyan","orange","pink"], value="white", label="Color")
                        cap_style = gr.Dropdown(["box","outline","shadow","plain"], value="box", label="Style")
                        cap_pos = gr.Dropdown(["bottom","top","middle"], value="bottom", label="Position")
                    use_karaoke = gr.Checkbox(value=False, label="Karaoke word-by-word highlight",
                        info="Each word lights up as it's spoken (ASS subtitles)")
                btn_render = gr.Button("Render Video", variant="primary", size="lg")

            with gr.Group():
                gr.HTML('<p class="sl">Output</p>')
                pbar1 = gr.HTML(_pbar(0, "Waiting"), elem_classes=["pb"])
                log1 = gr.Textbox(label="Log", lines=12, max_lines=16, interactive=False, elem_classes=["con"])
                vid1 = gr.Video(label="Video")
                btn_open1 = gr.Button("Open output folder", variant="secondary", size="sm", elem_classes=["sm"])

        # ═══ TAB 2: YOUTUBE ═══════════════════════════════════════════
        with gr.TabItem("YouTube Clipper"):
            with gr.Group():
                gr.HTML('<p class="sl">Clip highlights from YouTube</p>')
                gr.HTML('<div class="hnt">Paste a URL. System downloads, analyzes transcript + audio peaks + scene cuts, picks best moments, assembles a highlight reel.</div>')
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                with gr.Row():
                    yt_n = gr.Slider(2, 10, value=5, step=1, label="Clips")
                    yt_dur = gr.Slider(5, 30, value=15, step=5, label="Clip length (sec)")
                    yt_model = gr.Dropdown(["qwen2.5:7b","llama3.2:3b"], value=config.OLLAMA_MODEL, label="Model", allow_custom_value=True)
                btn_clip = gr.Button("Clip Highlights", variant="primary")
            with gr.Group():
                gr.HTML('<p class="sl">Output</p>')
                pbar2 = gr.HTML(_pbar(0, "Waiting"), elem_classes=["pb"])
                log2 = gr.Textbox(label="Log", lines=10, max_lines=14, interactive=False, elem_classes=["con"])
                vid2 = gr.Video(label="Highlight Reel")
                btn_open2 = gr.Button("Open output folder", variant="secondary", size="sm", elem_classes=["sm"])

    gr.HTML('<p class="ft">Output includes final.mp4, thumbnail.png, metadata.json, upload_package.json, qa_report.json</p>')

    # ── Wiring ────────────────────────────────────────────────────────
    genre_dd.change(on_genre, [genre_dd], [speed_sl, n_scenes_sl, fmt_dd, trends_md])

    btn_write.click(write_script,
        [genre_dd, topic_tb, custom_style_tb, character_tb, n_scenes_sl, model_dd, refine_cb, research_cb],
        [title_tb, desc_tb, tags_tb, scene_table, outdir_state, status_md])

    btn_render.click(render_video,
        [genre_dd, outdir_state, title_tb, desc_tb, tags_tb, scene_table,
         custom_style_tb, character_tb, voice_dd, speed_sl, seed_num, use_anim_cb, fmt_dd,
         cap_font, cap_size, cap_color, cap_bold, cap_style, cap_pos, use_karaoke],
        [pbar1, log1, vid1])

    btn_clip.click(clip_youtube, [yt_url, yt_n, yt_dur, yt_model], [pbar2, log2, vid2])
    btn_open1.click(open_folder, [outdir_state], [status_md])
    btn_open2.click(open_folder, [outdir_state], [status_md])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=1)
    demo.launch(inbrowser=True, allowed_paths=[str(config.OUT_ROOT.resolve())])
