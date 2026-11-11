#!/usr/bin/env python3
"""
Command-line interface for the video production pipeline.

Usage:
    python server/_cli.py --genre drama_story "A mother discovers her son's secret"
    python server/_cli.py --genre kids_educational "Why is the sky blue?" --age 3-5
    python server/_cli.py --genre trivia_quiz "Amazing animal facts"
    python server/_cli.py --list-genres
    python server/_cli.py --script output/<folder>/script.json
    python server/_cli.py --topics-file topics.txt --genre horror_story
    python server/_cli.py --no-refine "Quick test topic"
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# ── path setup (allows running as ``python server/_cli.py`` from project root)
_SERVER = Path(__file__).resolve().parent
_ROOT = _SERVER.parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))
import os
os.chdir(_ROOT)

from pipeline import config
from pipeline.script import generate_script, set_logger as set_script_log
from pipeline.images import generate_images, set_logger as set_img_log
from pipeline.voice import generate_voice, set_logger as set_voice_log
from pipeline.assembler import assemble, set_logger as set_asm_log
from genres.loader import load_genre, list_genres
import re


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "video"


def show_genres():
    print("\nAvailable genres:\n")
    for g in list_genres():
        print(f"  {g['id']:20s}  {g['name']} — {g['description']}")
    print()


def process(topic, genre, genre_id, args):
    # Wire up loggers
    for fn in (set_script_log, set_img_log, set_voice_log, set_asm_log):
        fn(log)

    if args.script:
        script_path = Path(args.script)
        outdir = script_path.parent
        script = json.loads(script_path.read_text(encoding="utf-8"))
        log(f"Using existing script: {script_path}")
    else:
        outdir = config.OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(topic)}"
        outdir.mkdir(parents=True, exist_ok=True)
        script = generate_script(
            topic, genre,
            n_scenes=args.scenes,
            age=getattr(args, "age", None),
            model=args.model,
            refine=not args.no_refine,
        )
        script_path = outdir / "script.json"
        script_path.write_text(
            json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.review:
        print("\n" + "=" * 60)
        print("TITLE:", script["title"])
        for i, s in enumerate(script["scenes"], 1):
            speaker = s.get("speaker", "narrator")
            mood = s.get("mood", "")
            print(f"\n  Scene {i} [{speaker}] ({mood}): {s['narration']}")
        print("\n" + "=" * 60)
        input(
            f"Check the facts. Edit {script_path} if needed, "
            "save, then press Enter to render..."
        )
        script = json.loads(script_path.read_text(encoding="utf-8"))

    scenes = script["scenes"]
    images = generate_images(scenes, genre, outdir, args.seed)
    audios, durations = generate_voice(scenes, genre, outdir)
    final = assemble(
        scenes, images, audios, durations, genre, outdir,
        title=script.get("title", topic or ""),
    )

    # Thumbnail
    if config.GENERATE_THUMBNAIL and images:
        try:
            from pipeline.thumbnail import (
                generate_thumbnail, set_logger as set_thumb_log,
            )
            set_thumb_log(log)
            # Use the second scene image (usually more interesting than the intro)
            best_img = images[min(1, len(images) - 1)]
            # Use the raw (non-upscaled) version for thumbnail
            raw_name = best_img.stem.removesuffix("_up") + best_img.suffix
            raw = best_img.with_name(raw_name)
            if raw.exists() and raw != best_img:
                best_img = raw
            generate_thumbnail(script.get("title", topic), best_img, genre_id, outdir)
        except Exception as exc:
            log(f"Thumbnail generation skipped: {exc}")

    # Metadata
    made_for_kids = genre.get("made_for_kids", False)
    meta = {
        "title": script["title"],
        "description": (
            (script.get("description", "").strip() or "")
            + "\n\nThis video was created with the help of AI tools."
        ),
        "tags": script.get("tags", []),
        "genre": genre.get("name", ""),
        "madeForKids": made_for_kids,
        "containsSyntheticMedia": True,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"DONE -> {final}")
    return final


def main():
    ap = argparse.ArgumentParser(description="AI Video Creator — multi-genre pipeline")
    ap.add_argument("topic", nargs="?", help="Video topic, in quotes")
    ap.add_argument("--genre", default="drama_story",
                    help="Genre template id (use --list-genres to see options)")
    ap.add_argument("--list-genres", action="store_true", help="Show available genres")
    ap.add_argument("--topics-file", help="Text file with one topic per line")
    ap.add_argument("--script", help="Path to an existing script.json (skips the LLM)")
    ap.add_argument("--age", help="Audience age range (for kids_educational)")
    ap.add_argument("--scenes", type=int, help="Override scene count")
    ap.add_argument("--seed", type=int, default=1234, help="Base seed for images")
    ap.add_argument("--model", default=None, help="Ollama model override")
    ap.add_argument("--review", action="store_true",
                    help="Pause after script for fact-check/edit")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip the script refinement pass (faster)")
    args = ap.parse_args()

    if args.list_genres:
        show_genres()
        return

    for tool in ("ffmpeg",):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found in PATH. Install it first.")

    try:
        genre = load_genre(args.genre)
    except FileNotFoundError:
        print(f"Unknown genre: {args.genre}")
        show_genres()
        sys.exit(1)

    genre_id = args.genre

    if args.scenes is None:
        args.scenes = genre.get("target_scenes", 7)

    if args.script:
        process(None, genre, genre_id, args)
    elif args.topics_file:
        topics = [
            t.strip()
            for t in Path(args.topics_file).read_text(encoding="utf-8").splitlines()
            if t.strip()
        ]
        for t in topics:
            log(f"=== Topic: {t} ===")
            process(t, genre, genre_id, args)
    elif args.topic:
        process(args.topic, genre, genre_id, args)
    else:
        ap.error("Give a topic, --topics-file, or --script")


if __name__ == "__main__":
    main()
