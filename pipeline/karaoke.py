"""
Karaoke word-by-word captions.

Generates ASS (Advanced SubStation Alpha) subtitles with per-word
highlight timing — the current word lights up as it's spoken.

Timing source: word durations estimated from the TTS audio by splitting
narration text proportionally to audio duration (syllable-weighted).
For higher accuracy, Whisper word-level timestamps can be used if available.
"""
import math
import re
import subprocess
import time
from pathlib import Path

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _syllable_count(word: str) -> int:
    """Rough English syllable count for timing estimation."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp: H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s - int(s)) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _ass_color(hex_color: str) -> str:
    """Convert #RRGGBB to ASS &HBBGGRR format."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
        return f"&H00{b}{g}{r}"
    return "&H00FFFFFF"


# ── Word timing estimation ──────────────────────────────────────────────

def estimate_word_timings(
    text: str,
    start_time: float,
    duration: float,
) -> list[dict]:
    """Estimate per-word start/end times based on syllable weight.

    Returns [{word, start, end}, ...] for each word.
    """
    words = text.split()
    if not words:
        return []

    # Weight each word by syllable count
    syllables = [_syllable_count(w) for w in words]
    total_syl = sum(syllables)
    if total_syl == 0:
        total_syl = len(words)
        syllables = [1] * len(words)

    # Small padding at start/end for natural feel
    pad = min(0.15, duration * 0.05)
    usable = duration - pad * 2
    t = start_time + pad

    result = []
    for word, syl in zip(words, syllables):
        word_dur = usable * syl / total_syl
        result.append({
            "word": word,
            "start": t,
            "end": t + word_dur,
        })
        t += word_dur

    return result


# ── ASS subtitle generation ─────────────────────────────────────────────

def generate_karaoke_ass(
    scenes: list[dict],
    durations: list[float],
    outpath: Path,
    *,
    font: str = "Arial",
    font_size: int = 26,
    primary_color: str = "#FFFFFF",
    highlight_color: str = "#FFD700",
    outline_color: str = "#000000",
    position: str = "bottom",
    scene_pad: float = 0.4,
    transition_dur: float = 0.6,
    is_vertical: bool = True,
) -> Path:
    """Generate an ASS subtitle file with karaoke word highlighting.

    Each word appears with the normal color, then highlights to
    ``highlight_color`` when it's being spoken.
    """
    _ts("Generating karaoke captions...")

    # ASS alignment: 2=bottom-center, 8=top-center, 5=middle
    alignment = {"bottom": 2, "top": 8, "middle": 5}.get(position, 2)
    margin_v = 120 if is_vertical else 40

    primary = _ass_color(primary_color)
    highlight = _ass_color(highlight_color)
    outline = _ass_color(outline_color)

    # ASS header
    header = f"""[Script Info]
Title: Karaoke Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary},{highlight},{outline},&H80000000,1,0,0,0,100,100,0,0,3,2,0,{alignment},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    t0 = 0.0

    for si, (scene, dur) in enumerate(zip(scenes, durations)):
        text = scene.get("narration", "").strip()
        if not text:
            t0 += dur + scene_pad - (transition_dur if si < len(scenes) - 1 else 0)
            continue

        word_timings = estimate_word_timings(text, t0, dur)

        if word_timings:
            # Build karaoke line: show all words, highlight current one
            line_start = word_timings[0]["start"]
            line_end = word_timings[-1]["end"]

            # Group words into lines of ~5-6 words for readability
            words_per_line = 5 if is_vertical else 6
            for chunk_start in range(0, len(word_timings), words_per_line):
                chunk = word_timings[chunk_start:chunk_start + words_per_line]
                if not chunk:
                    continue

                c_start = chunk[0]["start"]
                c_end = chunk[-1]["end"]

                # Build the ASS karaoke text with \k timing
                parts = []
                for wt in chunk:
                    # Duration in centiseconds for the \k tag
                    k_dur = int((wt["end"] - wt["start"]) * 100)
                    parts.append(f"{{\\kf{k_dur}}}{wt['word']}")

                karaoke_text = " ".join(parts)
                events.append(
                    f"Dialogue: 0,{_ass_time(c_start)},{_ass_time(c_end)},"
                    f"Default,,0,0,0,,{karaoke_text}"
                )

        # Advance timeline
        clip_dur = dur + scene_pad
        if si < len(durations) - 1:
            t0 += clip_dur - transition_dur
        else:
            t0 += clip_dur

    ass_content = header + "\n".join(events) + "\n"
    outpath.write_text(ass_content, encoding="utf-8")
    _ts(f"Karaoke ASS: {outpath} ({len(events)} events)")
    return outpath


def build_karaoke_filter(ass_path: Path) -> str:
    """Build the FFmpeg subtitle filter string for an ASS file."""
    posix = ass_path.resolve().as_posix().replace(":", "\\:")
    return f"ass='{posix}'"
