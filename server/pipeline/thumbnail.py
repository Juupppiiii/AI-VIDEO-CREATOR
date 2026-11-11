"""
Thumbnail and intro/outro card generation using Pillow.

Produces:
  - YouTube thumbnail (1280×720 with bold title overlay)
  - Intro card image (title + genre branding)
  - Outro card image ("Thanks for watching" + subscribe prompt)
"""
import textwrap
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from pipeline.config import OUT_W, OUT_H, GENRE_COLORS

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Try to load Arial from the Windows fonts directory."""
    font_names = (
        ["arialbd.ttf", "arial.ttf"] if bold
        else ["arial.ttf", "arialbd.ttf"]
    )
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    # Fallback for non-Windows
    return ImageFont.load_default()


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def _genre_style(genre_id: str) -> dict:
    return GENRE_COLORS.get(genre_id, {"bg": "#1A1A2E", "text": "#FFFFFF"})


# ── Thumbnail ───────────────────────────────────────────────────────────

def generate_thumbnail(
    title: str,
    scene_image: Path,
    genre_id: str,
    outdir: Path,
) -> Path:
    """Create a YouTube thumbnail from a scene image + title text."""
    img = Image.open(scene_image).convert("RGB")
    img = img.resize((OUT_W, OUT_H), Image.LANCZOS)

    draw = ImageDraw.Draw(img)

    # Darken bottom third for text readability
    overlay = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for y in range(OUT_H // 2, OUT_H):
        alpha = int(180 * (y - OUT_H // 2) / (OUT_H // 2))
        overlay_draw.line([(0, y), (OUT_W, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Title text
    style = _genre_style(genre_id)
    text_color = _hex_to_rgb(style["text"])
    font_large = _get_font(56, bold=True)

    # Word-wrap title
    wrapped = textwrap.fill(title, width=28)
    lines = wrapped.split("\n")

    # Draw each line with outline
    y_start = OUT_H - 60 - len(lines) * 65
    for i, line in enumerate(lines):
        y = y_start + i * 65
        x = 40
        # Black outline
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                draw.text((x + dx, y + dy), line, fill=(0, 0, 0), font=font_large)
        draw.text((x, y), line, fill=text_color, font=font_large)

    path = outdir / "thumbnail.png"
    img.save(path, quality=95)
    _ts(f"Thumbnail generated: {path}")
    return path


# ── Intro card ──────────────────────────────────────────────────────────

def generate_intro_card(
    title: str,
    genre: dict[str, Any],
    genre_id: str,
    outdir: Path,
) -> Path:
    """Create a title card image for the video intro."""
    style = _genre_style(genre_id)
    bg_color = _hex_to_rgb(style["bg"])
    text_color = _hex_to_rgb(style["text"])

    img = Image.new("RGB", (OUT_W, OUT_H), bg_color)
    draw = ImageDraw.Draw(img)

    # Genre label
    font_small = _get_font(28, bold=False)
    genre_name = genre.get("name", genre_id).upper()
    bbox = draw.textbbox((0, 0), genre_name, font=font_small)
    tw = bbox[2] - bbox[0]
    draw.text(((OUT_W - tw) // 2, OUT_H // 2 - 100), genre_name,
              fill=(*text_color[:3],), font=font_small)

    # Title
    font_large = _get_font(52, bold=True)
    wrapped = textwrap.fill(title, width=30)
    lines = wrapped.split("\n")
    y_start = OUT_H // 2 - 30
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(((OUT_W - tw) // 2, y_start + i * 62), line,
                  fill=text_color, font=font_large)

    # Subtle bottom line accent
    accent_y = OUT_H // 2 + len(lines) * 62
    draw.line(
        [(OUT_W // 2 - 80, accent_y), (OUT_W // 2 + 80, accent_y)],
        fill=text_color, width=3,
    )

    path = outdir / "intro_card.png"
    img.save(path)
    return path


# ── Outro card ──────────────────────────────────────────────────────────

def generate_outro_card(
    genre: dict[str, Any],
    genre_id: str,
    outdir: Path,
) -> Path:
    """Create an end card with thanks + subscribe prompt."""
    style = _genre_style(genre_id)
    bg_color = _hex_to_rgb(style["bg"])
    text_color = _hex_to_rgb(style["text"])

    img = Image.new("RGB", (OUT_W, OUT_H), bg_color)
    draw = ImageDraw.Draw(img)

    font_large = _get_font(48, bold=True)
    font_med = _get_font(30, bold=False)

    # Main text
    main_text = "Thanks for watching"
    bbox = draw.textbbox((0, 0), main_text, font=font_large)
    tw = bbox[2] - bbox[0]
    draw.text(((OUT_W - tw) // 2, OUT_H // 2 - 60), main_text,
              fill=text_color, font=font_large)

    # Subscribe prompt
    sub_text = "Like & Subscribe for more"
    bbox = draw.textbbox((0, 0), sub_text, font=font_med)
    tw = bbox[2] - bbox[0]
    draw.text(((OUT_W - tw) // 2, OUT_H // 2 + 20), sub_text,
              fill=(*text_color[:3],), font=font_med)

    # Bottom accent
    draw.line(
        [(OUT_W // 2 - 60, OUT_H // 2 + 80), (OUT_W // 2 + 60, OUT_H // 2 + 80)],
        fill=text_color, width=2,
    )

    path = outdir / "outro_card.png"
    img.save(path)
    return path
