#!/usr/bin/env python3
"""
Extract key frames from reference MP4 videos for LoRA training.

Usage:
    python tools/extract_frames.py video.mp4 --out dataset/ --count 25
    python tools/extract_frames.py video1.mp4 video2.mp4 --out dataset/
"""
import argparse
import subprocess
import sys
from pathlib import Path


def extract(video: Path, outdir: Path, count: int = 25, size: int = 512):
    """Extract evenly-spaced frames, cropped to square."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Get duration
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    dur = float(p.stdout.strip())
    interval = dur / (count + 1)

    extracted = 0
    for i in range(1, count + 1):
        t = interval * i
        out = outdir / f"{video.stem}_{i:03d}.png"
        subprocess.run([
            "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video),
            "-vframes", "1",
            "-vf", f"crop=min(iw\\,ih):min(iw\\,ih),scale={size}:{size}",
            str(out),
        ], capture_output=True)
        if out.exists():
            extracted += 1
            print(f"  [{extracted}/{count}] {out.name} (t={t:.1f}s)")

    print(f"\nExtracted {extracted} frames to {outdir}")
    return extracted


def main():
    ap = argparse.ArgumentParser(description="Extract frames from reference videos for LoRA training")
    ap.add_argument("videos", nargs="+", help="MP4 video files")
    ap.add_argument("--out", default="dataset", help="Output directory")
    ap.add_argument("--count", type=int, default=25, help="Frames per video")
    ap.add_argument("--size", type=int, default=512, help="Output size (square)")
    args = ap.parse_args()

    outdir = Path(args.out)
    total = 0
    for v in args.videos:
        vp = Path(v)
        if not vp.exists():
            print(f"Skipping {v} — not found")
            continue
        print(f"\n=== {vp.name} ===")
        total += extract(vp, outdir, args.count, args.size)

    print(f"\nTotal: {total} frames in {outdir}/")
    print("Next: run  python tools/caption_dataset.py dataset/  to generate captions")


if __name__ == "__main__":
    main()
