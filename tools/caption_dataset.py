#!/usr/bin/env python3
"""
Auto-caption dataset images for LoRA training using Ollama vision or manual templates.

Usage:
    python tools/caption_dataset.py dataset/
    python tools/caption_dataset.py dataset/ --style "flat 2D cartoon character in 3D environment"
"""
import argparse
import json
import sys
from pathlib import Path


def caption_with_template(image_path: Path, style_prefix: str) -> str:
    """Generate a caption from filename + style prefix (no AI needed)."""
    name = image_path.stem.replace("_", " ").replace("-", " ")
    return f"{style_prefix}, {name}"


def caption_with_ollama(image_path: Path, style_prefix: str, model: str = "llava:7b") -> str:
    """Use Ollama's vision model to describe the image."""
    import base64
    import requests

    img_b64 = base64.b64encode(image_path.read_bytes()).decode()

    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": (
                f"Describe this image in one sentence for Stable Diffusion training. "
                f"Start with: {style_prefix}. "
                f"Describe the subject, action, environment, lighting, and camera angle. "
                f"Keep it under 50 words."
            ),
            "images": [img_b64],
            "stream": False,
        }, timeout=60)
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception as e:
        print(f"  Ollama vision failed ({e}) — using template caption")
        return caption_with_template(image_path, style_prefix)


def main():
    ap = argparse.ArgumentParser(description="Caption dataset images for LoRA training")
    ap.add_argument("dataset", help="Directory with PNG images")
    ap.add_argument("--style", default="flat 2D cartoon character in a detailed 3D environment, digital painting",
                    help="Style prefix for every caption")
    ap.add_argument("--vision", action="store_true",
                    help="Use Ollama vision model (llava) for smart captions")
    ap.add_argument("--model", default="llava:7b", help="Ollama vision model")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"Directory not found: {dataset}")

    images = sorted(list(dataset.glob("*.png")) + list(dataset.glob("*.jpg")))
    if not images:
        sys.exit(f"No images found in {dataset}")

    print(f"Captioning {len(images)} images...")
    if args.vision:
        print(f"Using Ollama vision model: {args.model}")
        print("(Make sure `ollama serve` is running and the model is pulled)")

    for img in images:
        txt_path = img.with_suffix(".txt")

        if txt_path.exists():
            print(f"  {img.name} — caption exists, skipping")
            continue

        if args.vision:
            caption = caption_with_ollama(img, args.style, args.model)
        else:
            caption = caption_with_template(img, args.style)

        txt_path.write_text(caption, encoding="utf-8")
        print(f"  {img.name} → {caption[:80]}...")

    print(f"\nDone. {len(images)} caption .txt files in {dataset}/")
    print("Next: train your LoRA with kohya_ss or run the training script.")


if __name__ == "__main__":
    main()
