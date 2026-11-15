"""
Central configuration — Comic Documentary Studio
Hardware target: RTX 4050 Laptop (6 GB VRAM), 7.7 GB RAM, Windows 11.
Strategy: load one model at a time, unload before the next.
All output is 9:16 portrait (YouTube Shorts).
"""
from pathlib import Path

# ──────────────────────── Paths ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT     = PROJECT_ROOT / "output"
MUSIC_DIR    = PROJECT_ROOT / "music"
SFX_DIR      = PROJECT_ROOT / "sfx"
DATA_DIR     = PROJECT_ROOT / "data"
GENRES_DIR   = PROJECT_ROOT / "genres" / "templates"

# ──────────────────────── Ollama (script generation) ───────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b"

# ──────────────────────── Image generation (SD 1.5) ────────────────────
# SD 1.5 loads in ~2 GB VRAM — fits the RTX 4050 comfortably.
# SDXL needs 6+ GB and OOMs on this machine — keep SD 1.5.
SD_MODEL        = "stable-diffusion-v1-5/stable-diffusion-v1-5"
STEPS           = 28          # 28 steps: noticeably sharper than 25, still fast
GUIDANCE        = 7.5         # slight increase for stronger comic-art style adherence
USE_CPU_OFFLOAD = False       # SD 1.5 fits fully on 6 GB — no offload needed

# Shared negative prompt — appended to every scene's negative
NEGATIVE_BASE = (
    "text, letters, words, watermark, logo, signature, "
    "deformed, extra limbs, bad anatomy, blurry, low quality, "
    "jpeg artifacts, overexposed, underexposed, duplicate"
)

# ──────────────────────── Image upscaling (CPU, zero VRAM) ─────────────
# Lanczos 2× before Ken Burns gives zoompan more pixels to work with.
# Runs after SD is unloaded so VRAM is free.
UPSCALE_IMAGES = True
UPSCALE_FACTOR = 2            # 432×768 → 864×1536

# ──────────────────────── Video output — always 9:16 portrait ──────────
OUT_W = 1080
OUT_H = 1920

# Legacy landscape constants (kept so other code doesn't break)
OUT_W_SHORT = 1080
OUT_H_SHORT = 1920

# SD 1.5 image generation sizes — 9:16 native (multiples of 64)
IMG_W       = 432
IMG_H       = 768

# Legacy landscape size (kept for _cli.py compat)
IMG_W_SHORT = 432
IMG_H_SHORT = 768

FPS             = 30          # 30 fps feels more cinematic than 25 for Shorts
SCENE_PAD       = 0.4         # silence after each narration clip (seconds)
TRANSITION_DUR  = 0.6         # xfade cross-dissolve duration (seconds)
CAPTION_WORDS   = 5           # max words per caption line (tighter = more readable)

# ──────────────────────── Audio ─────────────────────────────────────────
MUSIC_VOLUME      = 0.07      # background music level under narration
MUSIC_DUCK_VOLUME = 0.02      # ducked level during speech (sidechain compressor)

# ──────────────────────── Sound effects ─────────────────────────────────
SFX_ENABLED = True
SFX_VOLUME  = 0.10

# ──────────────────────── Script ────────────────────────────────────────
REFINE_SCRIPT = True          # second LLM pass for better hooks & pacing

# ──────────────────────── AnimateDiff-Lightning ─────────────────────────
# Generates 16-frame animated clips instead of still images.
# Requires ~3 GB free RAM at load time.  Toggle in the UI.
USE_ANIMATEDIFF    = False
ANIMATEDIFF_STEPS  = 4        # 4-step Lightning distilled model
ANIMATEDIFF_FRAMES = 16       # ~0.53 s per clip at 30 fps
ANIMATEDIFF_BASE   = "stable-diffusion-v1-5/stable-diffusion-v1-5"
ANIMATEDIFF_REPO   = "ByteDance/AnimateDiff-Lightning"
ANIMATEDIFF_CKPT   = "animatediff_lightning_4step_diffusers.safetensors"

# ──────────────────────── Thumbnail & cards ─────────────────────────────
GENERATE_THUMBNAIL = True
INTRO_DURATION     = 0.0      # disabled — video starts directly on scene 1
OUTRO_DURATION     = 0.0      # disabled

# ──────────────────────── Voice map ─────────────────────────────────────
VOICE_MAP = {
    "narrator":      "am_adam",
    "narrator_male": "am_adam",
    "narrator_f":    "af_heart",
    "character_1":   "af_bella",
    "character_2":   "am_michael",
    "character_3":   "af_nicole",
    "character_4":   "af_sarah",
}
DEFAULT_VOICE       = "am_adam"
DEFAULT_VOICE_SPEED = 0.90

# ──────────────────────── Mood → xfade transition map ───────────────────
MOOD_TRANSITIONS: dict[str, str] = {
    # dark / tense
    "tense":       "fadeblack",
    "shocking":    "fadeblack",
    "dread":       "fadeblack",
    "eerie":       "fadeblack",
    "fearful":     "fadeblack",
    "suspenseful": "fadeblack",
    # reveals
    "revealing":   "fadewhite",
    "triumphant":  "fadewhite",
    "hopeful":     "fadewhite",
    # emotional / soft
    "emotional":   "dissolve",
    "mysterious":  "dissolve",
    "curious":     "dissolve",
    "warm":        "dissolve",
    "peaceful":    "dissolve",
    # energetic
    "excited":     "slideleft",
    "determined":  "slideleft",
    "cheerful":    "slideleft",
}

# ──────────────────────── Genre brand colors ────────────────────────────
# Used for thumbnail overlays.
GENRE_COLORS: dict[str, dict[str, str]] = {
    "comic_documentary": {"bg": "#1a1a2e", "text": "#d4a853"},
    "kids_educational":  {"bg": "#FF9F1C", "text": "#FFFFFF"},
    "drama_story":       {"bg": "#1A1A2E", "text": "#E94560"},
    "horror_story":      {"bg": "#0D0D0D", "text": "#8B0000"},
    "trivia_quiz":       {"bg": "#1B0A55", "text": "#FFFFFF"},
    "motivational":      {"bg": "#1B4332", "text": "#FFD700"},
    "mystery":           {"bg": "#1A1A2E", "text": "#6C63FF"},
    "stickman":          {"bg": "#ffffff", "text": "#000000"},
}
