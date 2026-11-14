"""
Central configuration for the video production pipeline.

Hardware target: RTX 4050 (6 GB VRAM), Intel i5-13th gen, 8 GB RAM.
Strategy: load one model at a time, unload before loading the next.
"""
from pathlib import Path

# ──────────────────────── Paths ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # AI-VIDEO-CREATOR/
SERVER_ROOT = PROJECT_ROOT / "server"
OUT_ROOT = PROJECT_ROOT / "output"
MUSIC_DIR = PROJECT_ROOT / "music"
SFX_DIR = PROJECT_ROOT / "sfx"
DATA_DIR = PROJECT_ROOT / "data"
GENRES_DIR = SERVER_ROOT / "genres" / "templates"

# ──────────────────────── Ollama (script) ──────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

# ──────────────────────── Image generation ─────────────
# SD 1.5 — stable on 5-6 GB VRAM, loads in ~2 GB, faster than SDXL.
# Switch to "stabilityai/stable-diffusion-xl-base-1.0" if you have 8+ GB VRAM.
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
STEPS = 25           # SD 1.5 benefits from slightly more steps than SDXL
GUIDANCE = 7.0
USE_CPU_OFFLOAD = False  # SD 1.5 fits fully on 5 GB VRAM — no offload needed

# Shared negative prompt applied to every genre
NEGATIVE_BASE = (
    "text, letters, words, watermark, logo, deformed, extra limbs, "
    "bad anatomy, blurry, ugly, low quality, jpeg artifacts, "
    "oversaturated, underexposed"
)

# ──────────────────────── Image upscaling ──────────────
# Lanczos 2x upscale before Ken Burns gives the zoompan more pixels.
# Zero extra VRAM — runs on CPU via Pillow after SD is unloaded.
UPSCALE_IMAGES = True
UPSCALE_FACTOR = 2    # 768×432 → 1536×864

# ──────────────────────── Video output ─────────────────
# Landscape 16:9 (YouTube standard)
OUT_W = 1280
OUT_H = 720
# Vertical 9:16 (YouTube Shorts)
OUT_W_SHORT = 1080
OUT_H_SHORT = 1920
# Image generation sizes — SD 1.5 native multiples of 64
IMG_W = 768
IMG_H = 432          # 16:9
IMG_W_SHORT = 432
IMG_H_SHORT = 768    # 9:16
FPS = 25
SCENE_PAD = 0.5          # silence after narration before transition
TRANSITION_DUR = 0.5     # cross-dissolve seconds between scenes
CAPTION_WORDS = 6        # max words per caption line

# ──────────────────────── Audio ────────────────────────
MUSIC_VOLUME = 0.08
MUSIC_DUCK_VOLUME = 0.03     # volume during narration (ducking)

# ──────────────────────── Sound effects ────────────────
SFX_ENABLED = True
SFX_VOLUME = 0.12

# ──────────────────────── Script refinement ────────────
# Second LLM pass to critique and improve the script for better hooks.
REFINE_SCRIPT = True

# ──────────────────────── AnimateDiff-Lightning ────────
# Set USE_ANIMATEDIFF=True to generate short animated clips per scene
# instead of still images with Ken Burns.
# Uses AnimateDiff-Lightning (ByteDance) on top of SD 1.5 — fits 5GB VRAM.
USE_ANIMATEDIFF    = False          # flip to True to enable
ANIMATEDIFF_STEPS  = 4             # 4-step Lightning model — fast + good quality
ANIMATEDIFF_FRAMES = 16            # frames per scene clip (~0.64 s at 25 fps)
ANIMATEDIFF_BASE   = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ANIMATEDIFF_REPO   = "ByteDance/AnimateDiff-Lightning"
ANIMATEDIFF_CKPT   = "animatediff_lightning_4step_diffusers.safetensors"

# ──────────────────────── Intro / Outro ────────────────
INTRO_DURATION = 2.0     # seconds — title card before the story
OUTRO_DURATION = 3.0     # seconds — "thanks for watching" end card
GENERATE_THUMBNAIL = True

# ──────────────────────── Voice map ────────────────────
# Keys used in genre templates and script JSON ("speaker" field).
VOICE_MAP = {
    "narrator":      "af_heart",
    "narrator_male": "am_adam",
    "character_1":   "af_bella",
    "character_2":   "am_michael",
    "character_3":   "af_nicole",
    "character_4":   "af_sarah",
}
DEFAULT_VOICE = "af_heart"
DEFAULT_VOICE_SPEED = 0.9

# ──────────────────────── Mood-based transitions ──────
# Maps scene mood → FFmpeg xfade transition name.
# Unmatched moods fall back to "dissolve".
MOOD_TRANSITIONS: dict[str, str] = {
    # dark / tense
    "tense": "fadeblack",
    "dread": "fadeblack",
    "eerie": "fadeblack",
    "terrifying": "fadeblack",
    "unsettling": "fadeblack",
    "hopeless": "fadeblack",
    "suspenseful": "fadeblack",
    "fearful": "fadeblack",
    # reveals / positive
    "shocking": "wipeleft",
    "revealing": "fadewhite",
    "triumphant": "fadewhite",
    "impressed": "fadewhite",
    # soft / emotional
    "emotional": "dissolve",
    "heartbreaking": "dissolve",
    "warm": "dissolve",
    "peaceful": "dissolve",
    "heavy": "dissolve",
    "struggling": "dissolve",
    # energetic
    "cheerful": "slideleft",
    "excited": "slideleft",
    "curious": "slideleft",
    "determined": "slideleft",
    "hopeful": "slideleft",
    # mystery
    "suspicious": "smoothleft",
    "intriguing": "smoothleft",
    "mysterious": "smoothleft",
    "satisfying": "dissolve",
}

# ──────────────────────── Genre brand colors ───────────
# Used for intro/outro cards and thumbnail overlays.
GENRE_COLORS: dict[str, dict[str, str]] = {
    "kids_educational": {"bg": "#FF9F1C", "text": "#FFFFFF"},
    "drama_story":      {"bg": "#1A1A2E", "text": "#E94560"},
    "horror_story":     {"bg": "#0D0D0D", "text": "#8B0000"},
    "trivia_quiz":      {"bg": "#1B0A55", "text": "#FFFFFF"},
    "motivational":     {"bg": "#1B4332", "text": "#FFD700"},
    "mystery":          {"bg": "#1A1A2E", "text": "#6C63FF"},
}
