#!/usr/bin/env python3
"""
Web front end for the kids video pipeline.
Run:  python app.py   (opens in your browser at http://127.0.0.1:7860)
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

# ── make_video.py lives one level up from this file (APP/) ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)   # so output/ and music/ resolve relative to the project root

import gradio as gr
import make_video as mv

VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael"]
RESOLUTIONS = {"720p (faster)": (1280, 720), "1080p (slower)": (1920, 1080)}
HEADERS = ["Narration (what the voice says)", "Image prompt (what to draw)"]


# ---------------------------- Step 1: script ----------------------------
def write_script(topic, age, n_scenes, model):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Please type a topic first.")
    mv.OLLAMA_MODEL = model
    outdir = mv.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{mv.slugify(topic)}"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        script = mv.generate_script(topic, age, int(n_scenes))
    except SystemExit as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Could not reach Ollama: {e}")
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [[s["narration"], s["image_prompt"]] for s in script["scenes"]]
    return (script["title"], script["description"], ", ".join(script["tags"]),
            rows, str(outdir),
            "Script ready. Read it, fix any wrong facts, then click 'Make the video'.")


# ---------------------------- Step 2: render ----------------------------
def make_video(outdir, title, description, tags, table, voice, speed, res, seed):
    if not outdir:
        raise gr.Error("Write the script first (step 1).")
    rows = table.values.tolist() if hasattr(table, "values") else table
    scenes = []
    for narration, prompt in rows:
        narration = (narration or "").strip()
        prompt = (prompt or "").strip() or narration
        if narration:
            scenes.append({"narration": narration, "image_prompt": prompt})
    if len(scenes) < 1:
        raise gr.Error("The script table is empty.")

    outdir = Path(outdir)
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    script = {"title": title, "description": description, "tags": tag_list, "scenes": scenes}
    (outdir / "script.json").write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    # apply settings
    mv.VOICE = voice
    mv.VOICE_SPEED = float(speed)
    mv.OUT_W, mv.OUT_H = RESOLUTIONS[res]

    # live log: tee the pipeline's log() into a queue
    q = queue.Queue()
    original_log = mv.log

    def tee(msg):
        original_log(msg)
        q.put(msg)

    mv.log = tee
    result = {}

    def work():
        try:
            images = mv.generate_images(scenes, outdir, int(seed))
            audios, durations = mv.generate_voice(scenes, outdir)
            result["final"] = mv.assemble(scenes, images, audios, durations, outdir)
            meta = {
                "title": title,
                "description": (description or "").strip()
                               + "\n\nThis video was created with the help of AI tools.",
                "tags": tag_list,
                "madeForKids": True,
                "containsSyntheticMedia": True,
            }
            (outdir / "metadata.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            q.put(f"DONE. Saved to {outdir / 'final.mp4'}")
        except BaseException:
            result["error"] = traceback.format_exc()
            q.put("ERROR:\n" + result["error"])
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    lines = ["Starting... (first run downloads ~4GB of image model files)"]
    yield "\n".join(lines), None
    try:
        while True:
            try:
                # timeout so the generator can't block forever if the thread
                # dies without putting None (e.g. Gradio cancels the request)
                msg = q.get(timeout=300)
            except queue.Empty:
                lines.append("⚠️  Timed out waiting for the render thread (5 min). "
                              "The process may still be running in the background.")
                break
            if msg is None:
                break
            lines.append(msg)
            yield "\n".join(lines), None
    finally:
        # always restore the real log function, even if Gradio drops the stream
        mv.log = original_log

    video = str(result["final"]) if "final" in result else None
    yield "\n".join(lines), video


# --------------------------------- UI ----------------------------------
with gr.Blocks(title="Kids Video Maker") as demo:
    gr.Markdown("# Kids Educational Video Maker\n"
                "Topic -> script -> images -> voice -> finished video. All local.")
    outdir_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Write the script")
            topic = gr.Textbox(label="Topic", placeholder="Why do giraffes have long necks?")
            age = gr.Dropdown(["2-4", "4-6", "5-8"], value="4-6", label="Age group")
            n_scenes = gr.Slider(3, 12, value=6, step=1, label="Number of scenes (more = longer video)")
            model = gr.Dropdown(["qwen2.5:7b", "llama3.2:3b"], value=mv.OLLAMA_MODEL,
                                label="Ollama model", allow_custom_value=True)
            btn_script = gr.Button("1. Write script", variant="primary")

            gr.Markdown("### Voice & video settings")
            voice = gr.Dropdown(VOICES, value=mv.VOICE, label="Voice")
            speed = gr.Slider(0.7, 1.2, value=mv.VOICE_SPEED, step=0.05, label="Voice speed")
            res = gr.Dropdown(list(RESOLUTIONS), value="720p (faster)", label="Resolution")
            seed = gr.Number(value=1234, precision=0,
                             label="Image seed (change it to get different pictures)")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Check and edit (fix any wrong facts!)")
            status = gr.Markdown("")
            title = gr.Textbox(label="Video title", interactive=True)
            description = gr.Textbox(label="Description", lines=3, interactive=True)
            tags = gr.Textbox(label="Tags (comma separated)", interactive=True)
            table = gr.Dataframe(headers=HEADERS, datatype=["str", "str"],
                                 col_count=(2, "fixed"), row_count=(6, "dynamic"),
                                 interactive=True, wrap=True, label="Scenes")
            btn_make = gr.Button("2. Make the video", variant="primary")

    gr.Markdown("### 3. Result")
    log_box = gr.Textbox(label="Progress log", lines=10, max_lines=14, interactive=False)
    video = gr.Video(label="Finished video")
    gr.Markdown("Before uploading to YouTube: watch it fully, set **'Yes, it's made for kids'**, "
                "and use the title/description/tags saved in `metadata.json` in the video's folder.")

    btn_script.click(write_script, [topic, age, n_scenes, model],
                     [title, description, tags, table, outdir_state, status])
    btn_make.click(make_video,
                   [outdir_state, title, description, tags, table, voice, speed, res, seed],
                   [log_box, video])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(inbrowser=True, allowed_paths=[str(mv.OUT_ROOT.resolve())])
