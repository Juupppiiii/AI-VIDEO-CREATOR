#!/usr/bin/env python3
"""
Kids educational video pipeline (runs locally on a 6GB GPU).

topic -> Ollama script -> Stable Diffusion images -> Kokoro voice
      -> FFmpeg (zoom/pan + captions + optional music) -> MP4 + metadata

Usage:
    python make_video.py "Why do giraffes have long necks?"
    python make_video.py "Learn the colors" --age 3-5 --scenes 7 --review
    python make_video.py --script output/<folder>/script.json      # reuse an edited script
    python make_video.py --topics-file topics.txt                  # batch, one topic per line
"""
import argparse
import gc
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

# =============================== CONFIG ===============================
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"          # fallback: "llama3.2:3b" if it's too slow

DEFAULT_AGE = "4-6"
DEFAULT_SCENES = 6

# Voice (Kokoro). Good options: af_heart, af_bella, af_nicole, am_adam
VOICE = "af_heart"
VOICE_SPEED = 0.9                    # slower = clearer for kids

# Images (Stable Diffusion 1.5 via diffusers)
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMG_W, IMG_H = 768, 432              # 16:9, multiples of 8
STEPS = 25
GUIDANCE = 7.0
STYLE = ("children's picture book illustration, bright cheerful colors, "
         "soft rounded shapes, flat cartoon style, simple clean background, "
         "cute and friendly, high quality")
NEGATIVE = ("text, letters, words, watermark, logo, scary, dark, creepy, "
            "realistic, photo, deformed, extra limbs, bad anatomy, blurry, ugly")

# Your original mascot. The LLM writes "Sunny" in image prompts; we swap in the look.
MASCOT_NAME = "Sunny"
MASCOT_LOOK = "a cute friendly orange fox with a green scarf"

# Video
OUT_W, OUT_H = 1280, 720             # use 1920, 1080 for full HD (slower)
FPS = 25
SCENE_PAD = 0.6                      # seconds of silence after each scene
MUSIC_DIR = Path("music")            # drop royalty-free .mp3/.wav here (optional)
MUSIC_VOLUME = 0.08
CAPTION_WORDS = 6                    # max words per caption chunk
OUT_ROOT = Path("output")
# ======================================================================


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd=None):
    """Run a command, show ffmpeg's error tail if it fails."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2500:])
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd[:4]))} ...")
    return p


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ----------------------------- 1. SCRIPT -----------------------------
def generate_script(topic, age, n_scenes):
    system = (
        "You write scripts for short educational videos for young children. "
        "You always reply with valid JSON only."
    )
    user = f"""Write a kids educational video script.

Topic: "{topic}"
Audience age: {age} years old
Number of scenes: {n_scenes}

Rules:
- Use very simple words and short sentences. Warm, cheerful, encouraging tone.
- Each scene's narration is 2 to 3 short sentences.
- Scene 1 greets the kids and introduces the topic with a fun hook.
- The middle scenes each teach ONE simple, TRUE fact. Do not invent facts.
- Include one simple question to the viewer in the second-to-last scene.
- The last scene recaps what was learned and says goodbye.
- The friendly mascot is a fox named {MASCOT_NAME}. Use him as a guide in some scenes.
- image_prompt: describe ONLY what to draw, in English, one concrete scene, no text or letters in the picture.
  If the mascot appears, write his name "{MASCOT_NAME}".

Reply with JSON in exactly this shape:
{{
  "title": "catchy YouTube title under 70 characters",
  "description": "2-3 sentence YouTube description",
  "tags": ["kids", "learning", "..."],
  "scenes": [
    {{"narration": "...", "image_prompt": "..."}}
  ]
}}"""
    for attempt in range(1, 4):
        log(f"Asking {OLLAMA_MODEL} for a script (attempt {attempt})...")
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "keep_alive": 0,          # unload from VRAM right after, SD needs the memory
            "options": {"temperature": 0.7},
        }, timeout=900)
        r.raise_for_status()
        try:
            data = json.loads(r.json()["message"]["content"])
            scenes = data["scenes"]
            assert isinstance(scenes, list) and len(scenes) >= 3
            for s in scenes:
                assert s["narration"].strip() and s["image_prompt"].strip()
            data.setdefault("title", topic)
            data.setdefault("description", "")
            data.setdefault("tags", ["kids", "learning"])
            return data
        except Exception as e:
            log(f"Bad script format ({e}), retrying...")
    sys.exit("Could not get a valid script from Ollama. Is `ollama serve` running and the model pulled?")


# ----------------------------- 2. IMAGES -----------------------------
def build_image_prompt(raw):
    raw = raw.replace(MASCOT_NAME, MASCOT_LOOK)
    return f"{raw}, {STYLE}"


def generate_images(scenes, outdir, base_seed):
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found by PyTorch. Install the CUDA build of torch (see README).")

    log("Loading Stable Diffusion (first run downloads ~4GB)...")
    pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.to("cuda")

    paths = []
    for i, scene in enumerate(scenes):
        path = outdir / "images" / f"scene_{i:02d}.png"
        path.parent.mkdir(exist_ok=True)
        prompt = build_image_prompt(scene["image_prompt"])
        for retry in range(3):   # safety checker may blank an image; reseed and retry
            gen = torch.Generator("cuda").manual_seed(base_seed + i + retry * 1000)
            out = pipe(prompt, negative_prompt=NEGATIVE, width=IMG_W, height=IMG_H,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen)
            flagged = getattr(out, "nsfw_content_detected", None)
            if not (flagged and flagged[0]):
                break
            log(f"  scene {i}: image flagged by safety checker, retrying")
        out.images[0].save(path)
        paths.append(path)
        log(f"Image {i + 1}/{len(scenes)} done")

    del pipe
    free_gpu()
    return paths


# ------------------------------ 3. VOICE ------------------------------
def generate_voice(scenes, outdir):
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    log("Loading Kokoro TTS...")
    tts = KPipeline(lang_code="a")   # American English
    (outdir / "audio").mkdir(exist_ok=True)

    paths, durations = [], []
    for i, scene in enumerate(scenes):
        chunks = []
        for _, _, audio in tts(scene["narration"], voice=VOICE, speed=VOICE_SPEED):
            chunks.append(audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio))
        if not chunks:
            # Kokoro returned nothing (very short text); synthesise 0.5 s of silence
            log(f"  scene {i}: TTS returned no audio, inserting silence")
            chunks = [np.zeros(int(24000 * 0.5), dtype=np.float32)]
        wav = np.concatenate(chunks)
        path = outdir / "audio" / f"scene_{i:02d}.wav"
        sf.write(path, wav, 24000)
        paths.append(path)
        durations.append(len(wav) / 24000)
        log(f"Voice {i + 1}/{len(scenes)} done ({durations[-1]:.1f}s)")

    del tts
    free_gpu()
    return paths, durations


# ---------------------------- 4. ASSEMBLY -----------------------------
def srt_time(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def caption_chunks(text):
    chunks = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sent.split()
        for j in range(0, len(words), CAPTION_WORDS):
            chunks.append(" ".join(words[j:j + CAPTION_WORDS]))
    return [c for c in chunks if c]


def write_srt(scenes, durations, path):
    lines, idx, t0 = [], 1, 0.0
    for scene, dur in zip(scenes, durations):
        chunks = caption_chunks(scene["narration"])
        total = sum(len(c) for c in chunks) or 1
        t = t0
        for c in chunks:
            span = dur * len(c) / total
            lines += [str(idx), f"{srt_time(t)} --> {srt_time(t + span)}", c, ""]
            idx += 1
            t += span
        t0 += dur + SCENE_PAD
    path.write_text("\n".join(lines), encoding="utf-8")


def make_clip(img, wav, dur, out, idx):
    frames = int(math.ceil((dur + SCENE_PAD) * FPS))
    # alternate slow zoom-in / zoom-out for variety
    # (on-1) so the first frame starts at exactly 1.0 or 1.12, not one step ahead
    z = f"1+0.12*(on-1)/{frames}" if idx % 2 == 0 else f"1.12-0.12*(on-1)/{frames}"
    vf = (f"scale={OUT_W * 2}:{OUT_H * 2}:flags=lanczos,"
          f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
          f":d={frames}:s={OUT_W}x{OUT_H}:fps={FPS},format=yuv420p")
    run(["ffmpeg", "-y", "-i", str(img), "-i", str(wav),
         "-vf", vf, "-af", f"apad=pad_dur={SCENE_PAD}",
         "-t", f"{dur + SCENE_PAD:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", str(out)])


def assemble(scenes, images, audios, durations, outdir):
    clips_dir = outdir / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips = []
    for i, (img, wav, dur) in enumerate(zip(images, audios, durations)):
        out = clips_dir / f"scene_{i:02d}.mp4"
        make_clip(img, wav, dur, out, i)
        clips.append(out)
        log(f"Clip {i + 1}/{len(clips)} rendered")

    # concat (run inside outdir with relative paths to avoid Windows path quirks)
    (outdir / "list.txt").write_text(
        "\n".join(f"file 'clips/{c.name}'" for c in clips), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
         "-c", "copy", "merged.mp4"], cwd=outdir)

    write_srt(scenes, durations, outdir / "captions.srt")
    style = ("FontName=Arial,FontSize=22,Bold=1,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,Outline=3,Shadow=0,Alignment=2,MarginV=35")
    # On Windows, ffmpeg's subtitles filter chokes on backslashes and single-quote
    # wrapping inside a double-quoted shell string.  Writing the SRT path with
    # forward slashes and escaping the colon keeps it portable across platforms.
    srt_path = (outdir / "captions.srt").resolve().as_posix().replace(":", "\\:")
    sub = f"subtitles='{srt_path}':force_style='{style}'"

    music = None
    if MUSIC_DIR.exists():
        found = sorted(list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav")))
        music = found[0].resolve() if found else None

    # Base output flags (must come BEFORE the output filename)
    out_flags = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    if music:
        log(f"Mixing background music: {music.name}")
        cmd = ["ffmpeg", "-y", "-i", "merged.mp4", "-stream_loop", "-1", "-i", str(music),
               "-filter_complex",
               f"[0:v]{sub}[v];[1:a]volume={MUSIC_VOLUME}[m];"
               f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
               "-map", "[v]", "-map", "[a]", "-shortest",
               *out_flags, "-c:a", "aac", "-b:a", "192k", "final.mp4"]
    else:
        cmd = ["ffmpeg", "-y", "-i", "merged.mp4", "-vf", sub,
               *out_flags, "-c:a", "copy", "final.mp4"]
    run(cmd, cwd=outdir)
    return outdir / "final.mp4"


# ------------------------------- MAIN --------------------------------
def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"


def process(topic, args):
    if args.script:
        script_path = Path(args.script)
        outdir = script_path.parent
        script = json.loads(script_path.read_text(encoding="utf-8"))
        log(f"Using existing script: {script_path}")
    else:
        outdir = OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(topic)}"
        outdir.mkdir(parents=True, exist_ok=True)
        script = generate_script(topic, args.age, args.scenes)
        script_path = outdir / "script.json"
        script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.review:
        print("\n" + "=" * 60)
        print("TITLE:", script["title"])
        for i, s in enumerate(script["scenes"], 1):
            print(f"\nScene {i}: {s['narration']}")
        print("\n" + "=" * 60)
        input(f"Check the facts. Edit {script_path} if needed, save, then press Enter to render...")
        script = json.loads(script_path.read_text(encoding="utf-8"))

    scenes = script["scenes"]
    images = generate_images(scenes, outdir, args.seed)
    audios, durations = generate_voice(scenes, outdir)
    final = assemble(scenes, images, audios, durations, outdir)

    meta = {
        "title": script["title"],
        "description": (script["description"].strip()
                        + "\n\nThis video was created with the help of AI tools."),
        "tags": script["tags"],
        "madeForKids": True,        # REQUIRED for kids content (COPPA)
        "containsSyntheticMedia": True,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    log(f"DONE -> {final}")
    return final


def main():
    ap = argparse.ArgumentParser(description="Kids educational video generator")
    ap.add_argument("topic", nargs="?", help="Video topic, in quotes")
    ap.add_argument("--topics-file", help="Text file with one topic per line")
    ap.add_argument("--script", help="Path to an existing script.json (skips the LLM)")
    ap.add_argument("--age", default=DEFAULT_AGE, help="Audience age range, e.g. 3-5")
    ap.add_argument("--scenes", type=int, default=DEFAULT_SCENES)
    ap.add_argument("--seed", type=int, default=1234, help="Base seed for images")
    ap.add_argument("--review", action="store_true",
                    help="Pause after the script so you can fact-check/edit it")
    args = ap.parse_args()

    for tool in ("ffmpeg",):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found in PATH. Install it first.")

    if args.script:
        process(None, args)
    elif args.topics_file:
        topics = [t.strip() for t in Path(args.topics_file).read_text(encoding="utf-8").splitlines()
                  if t.strip()]
        for t in topics:
            log(f"=== Topic: {t} ===")
            process(t, args)
    elif args.topic:
        process(args.topic, args)
    else:
        ap.error("Give a topic, --topics-file, or --script")


if __name__ == "__main__":
    main()
