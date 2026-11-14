"""
Video assembly with FFmpeg.

Produces the final MP4 through these stages:

1. Per-scene clips — image (Ken Burns zoom/pan) + narration audio
2. Cross-fade chain — mood-specific transitions between scenes
3. Intro / outro cards — prepended and appended as short clips
4. Captions — styled SRT overlay
5. Audio mix — narration + mood-matched music (ducked) + SFX track
"""
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pipeline.config import (
    OUT_W, OUT_H, FPS, SCENE_PAD, TRANSITION_DUR,
    CAPTION_WORDS, MUSIC_DIR, MUSIC_VOLUME, MUSIC_DUCK_VOLUME,
    MOOD_TRANSITIONS, SFX_ENABLED,
    INTRO_DURATION, OUTRO_DURATION,
)

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2500:]
        raise RuntimeError(
            f"Command failed: {' '.join(map(str, cmd[:5]))} ...\n{tail}"
        )
    return p


# ── captions ────────────────────────────────────────────────────────────

def _caption_chunks(text: str) -> list[str]:
    chunks = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sent.split()
        for j in range(0, len(words), CAPTION_WORDS):
            c = " ".join(words[j:j + CAPTION_WORDS])
            if c:
                chunks.append(c)
    return chunks


def _srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"


def _write_srt(scenes, durations, path, transition_dur=0.0, time_offset=0.0):
    """Write an SRT file aligned to the final timeline."""
    lines, idx = [], 1
    t0 = time_offset
    for si, (scene, dur) in enumerate(zip(scenes, durations)):
        chunks = _caption_chunks(scene["narration"])
        total_chars = sum(len(c) for c in chunks) or 1
        t = t0
        for c in chunks:
            span = dur * len(c) / total_chars
            lines += [str(idx), f"{_srt_time(t)} --> {_srt_time(t + span)}", c, ""]
            idx += 1
            t += span
        clip_dur = dur + SCENE_PAD
        if si < len(durations) - 1:
            t0 += clip_dur - transition_dur
        else:
            t0 += clip_dur
    path.write_text("\n".join(lines), encoding="utf-8")


# ── per-scene clip ──────────────────────────────────────────────────────

def _make_clip(img: Path, wav: Path, dur: float, out: Path, idx: int,
               out_w: int = OUT_W, out_h: int = OUT_H):
    """Render one scene: image with Ken Burns + audio + padding."""
    total = dur + SCENE_PAD
    frames = int(math.ceil(total * FPS))

    # Alternate zoom-in / zoom-out
    if idx % 2 == 0:
        z = f"1+0.10*(on-1)/{frames}"
    else:
        z = f"1.10-0.10*(on-1)/{frames}"

    vf = (
        f"scale={out_w * 2}:{out_h * 2}:flags=lanczos,"
        f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={out_w}x{out_h}:fps={FPS},format=yuv420p"
    )
    _run([
        "ffmpeg", "-y",
        "-i", str(img), "-i", str(wav),
        "-vf", vf,
        "-af", f"apad=pad_dur={SCENE_PAD}",
        "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(out),
    ])


# ── intro / outro clips ────────────────────────────────────────────────

def _make_card_clip(img: Path, dur: float, out: Path,
                    out_w: int = OUT_W, out_h: int = OUT_H):
    """Render a static image into a silent video clip of *dur* seconds."""
    frames = int(math.ceil(dur * FPS))
    _run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(img),
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-vf", f"scale={out_w}:{out_h},format=yuv420p",
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-shortest",
        str(out),
    ])


# ── mood-based transition selection ─────────────────────────────────────

def _transition_for_mood(mood: str) -> str:
    return MOOD_TRANSITIONS.get(mood, "dissolve")


# ── cross-fade chain (with mood-specific transitions) ──────────────────

def _build_xfade_cmd(
    clips: list[Path],
    durations: list[float],
    scenes: list[dict],
    transition_dur: float,
    outpath: Path,
) -> list[str]:
    """Chain clips with mood-specific xfade + acrossfade."""
    n = len(clips)
    if n == 1:
        return ["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(outpath)]

    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    v_filters = []
    a_filters = []
    offset = durations[0] - transition_dur

    for i in range(1, n):
        # Pick transition from the INCOMING scene's mood
        mood = scenes[i].get("mood", "neutral") if i < len(scenes) else "neutral"
        trans = _transition_for_mood(mood)

        vin1 = f"v{i - 1}" if i > 1 else "0:v"
        vout = f"v{i}" if i < n - 1 else "outv"
        v_filters.append(
            f"[{vin1}][{i}:v]xfade=transition={trans}"
            f":duration={transition_dur:.3f}:offset={offset:.3f}[{vout}]"
        )

        ain1 = f"a{i - 1}" if i > 1 else "0:a"
        aout = f"a{i}" if i < n - 1 else "outa"
        a_filters.append(
            f"[{ain1}][{i}:a]acrossfade=d={transition_dur:.3f}"
            f":c1=tri:c2=tri[{aout}]"
        )

        if i < n - 1:
            offset += durations[i] - transition_dur

    fc = ";".join(v_filters + a_filters)

    return [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", fc,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(outpath),
    ]


# ── music selection ─────────────────────────────────────────────────────

def _find_music(mood: str) -> Path | None:
    mood_dir = MUSIC_DIR / mood
    if mood_dir.is_dir():
        found = sorted(
            list(mood_dir.glob("*.mp3")) + list(mood_dir.glob("*.wav"))
        )
        if found:
            return found[0].resolve()
    if MUSIC_DIR.is_dir():
        found = sorted(
            list(MUSIC_DIR.glob("*.mp3")) + list(MUSIC_DIR.glob("*.wav"))
        )
        if found:
            return found[0].resolve()
    return None


# ── compute total video duration ────────────────────────────────────────

def _total_duration(
    clip_durations: list[float],
    transition_dur: float,
) -> float:
    """Compute total video length accounting for transitions."""
    total = sum(clip_durations)
    if len(clip_durations) > 1 and transition_dur > 0:
        total -= (len(clip_durations) - 1) * transition_dur
    return total


# ── final assembly ──────────────────────────────────────────────────────

def assemble(
    scenes: list[dict],
    images: list[Path],
    audios: list[Path],
    durations: list[float],
    genre: dict[str, Any],
    outdir: Path,
    title: str = "",
    out_w: int | None = None,
    out_h: int | None = None,
) -> Path:
    """Full video assembly pipeline.

    ``out_w`` / ``out_h`` override config defaults so callers can render
    vertical 9:16 video for YouTube Shorts.
    """
    # Resolve output dimensions
    w = out_w or OUT_W
    h = out_h or OUT_H
    is_vertical = h > w

    clips_dir = outdir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # ── Step 1: per-scene clips ──
    clip_paths: list[Path] = []
    clip_durations: list[float] = []
    for i, (img, wav, dur) in enumerate(zip(images, audios, durations)):
        out = clips_dir / f"scene_{i:02d}.mp4"
        _make_clip(img, wav, dur, out, i, out_w=w, out_h=h)
        clip_paths.append(out)
        clip_durations.append(dur + SCENE_PAD)
        _ts(f"Clip {i + 1}/{len(images)} rendered")

    # ── Step 2: intro / outro cards ──
    intro_clip = None
    outro_clip = None
    try:
        from pipeline.thumbnail import generate_intro_card, generate_outro_card
        genre_id = _guess_genre_id(genre)

        # Use the video title for the intro card (fall back to first narration)
        intro_title = title or (scenes[0].get("narration", "")[:80] if scenes else "")
        intro_img = generate_intro_card(
            intro_title,
            genre, genre_id, outdir,
        )
        intro_clip = clips_dir / "intro.mp4"
        _make_card_clip(intro_img, INTRO_DURATION, intro_clip, out_w=w, out_h=h)
        _ts("Intro card rendered")

        outro_img = generate_outro_card(genre, genre_id, outdir)
        outro_clip = clips_dir / "outro.mp4"
        _make_card_clip(outro_img, OUTRO_DURATION, outro_clip, out_w=w, out_h=h)
        _ts("Outro card rendered")
    except Exception as exc:
        _ts(f"Intro/outro generation skipped: {exc}")
        intro_clip = None
        outro_clip = None

    # Build the full clip sequence: [intro] + scenes + [outro]
    full_clips: list[Path] = []
    full_durations: list[float] = []
    # Scenes for transition mapping (intro/outro get "neutral" mood)
    full_scenes: list[dict] = []

    if intro_clip:
        full_clips.append(intro_clip)
        full_durations.append(INTRO_DURATION)
        full_scenes.append({"mood": "neutral"})

    full_clips.extend(clip_paths)
    full_durations.extend(clip_durations)
    full_scenes.extend(scenes)

    if outro_clip:
        full_clips.append(outro_clip)
        full_durations.append(OUTRO_DURATION)
        full_scenes.append({"mood": "neutral"})

    # ── Step 3: mood-based transitions ──
    merged = outdir / "merged.mp4"
    use_transitions = len(full_clips) > 1 and TRANSITION_DUR > 0
    t_dur = TRANSITION_DUR if use_transitions else 0.0

    if use_transitions:
        _ts("Chaining clips with mood-based transitions...")
        cmd = _build_xfade_cmd(full_clips, full_durations, full_scenes, t_dur, merged)
        _run(cmd)
    else:
        if len(full_clips) == 1:
            shutil.copy2(full_clips[0], merged)
        else:
            concat_list = outdir / "list.txt"
            # Use forward-slash posix paths for FFmpeg compatibility on Windows
            concat_list.write_text(
                "\n".join(f"file '{c.as_posix()}'" for c in full_clips),
                encoding="utf-8",
            )
            _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                  "-i", str(concat_list), "-c", "copy", str(merged)])

    # ── Step 4: captions ──
    # Caption offset = intro duration minus the transition into the first scene
    if intro_clip and t_dur > 0:
        caption_offset = INTRO_DURATION - t_dur
    elif intro_clip:
        caption_offset = INTRO_DURATION
    else:
        caption_offset = 0.0
    _write_srt(scenes, durations, outdir / "captions.srt", t_dur, caption_offset)

    srt_posix = (outdir / "captions.srt").resolve().as_posix().replace(":", "\\:")
    # Vertical (Shorts): bigger font, higher margin so text clears the bottom UI
    if is_vertical:
        style = (
            "FontName=Arial,FontSize=42,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,BorderStyle=3,"
            "Outline=3,Shadow=0,Alignment=2,MarginV=180"
        )
    else:
        style = (
            "FontName=Arial,FontSize=24,Bold=1,"
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,BorderStyle=3,"
            "Outline=2,Shadow=0,Alignment=2,MarginV=40"
        )
    sub_filter = f"subtitles='{srt_posix}':force_style='{style}'"

    # ── Step 5: SFX track ──
    sfx_path = None
    if SFX_ENABLED:
        try:
            from pipeline.sfx import build_sfx_track, set_logger as set_sfx_log
            set_sfx_log(_log)
            total_dur = _total_duration(full_durations, t_dur)
            # SFX needs to know where the scene content starts (after intro)
            sfx_offset = caption_offset
            sfx_path = build_sfx_track(
                scenes, durations, t_dur, total_dur, time_offset=sfx_offset,
            )
        except Exception as exc:
            _ts(f"SFX generation skipped: {exc}")

    # ── Step 6: final mix — captions + music (ducked) + SFX ──
    music_mood = genre.get("music_mood", "")
    music = _find_music(music_mood)

    out_flags = [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]

    final = outdir / "final.mp4"

    if music:
        _ts(f"Mixing music ({music_mood}: {music.name}) with ducking + SFX...")
        inputs = ["-i", str(merged), "-stream_loop", "-1", "-i", str(music)]
        # Build the audio filtergraph with ducking
        # Split narration: one copy for output, one for sidechain detection
        af_parts = [
            f"[0:v]{sub_filter}[v]",
            "[0:a]asplit=2[narr_out][narr_sc]",
            f"[1:a]volume={MUSIC_VOLUME}[music_base]",
            (
                "[music_base][narr_sc]sidechaincompress="
                "level_in=1:threshold=0.015:ratio=6:"
                f"attack=200:release=1000:level_sc=1[music_ducked]"
            ),
        ]

        if sfx_path:
            inputs += ["-i", str(sfx_path)]
            sfx_idx = 2
            af_parts.append(f"[{sfx_idx}:a]aresample=44100[sfx]")
            af_parts.append(
                "[narr_out][music_ducked][sfx]amix=inputs=3:"
                "duration=first:dropout_transition=0:normalize=0[a]"
            )
        else:
            af_parts.append(
                "[narr_out][music_ducked]amix=inputs=2:"
                "duration=first:dropout_transition=0:normalize=0[a]"
            )

        fc = ";".join(af_parts)
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]", "-shortest",
            *out_flags, "-c:a", "aac", "-b:a", "192k",
            str(final),
        ]
    elif sfx_path:
        _ts("Mixing SFX (no background music)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(merged), "-i", str(sfx_path),
            "-filter_complex",
            (
                f"[0:v]{sub_filter}[v];"
                "[1:a]aresample=44100[sfx];"
                "[0:a][sfx]amix=inputs=2:duration=first:normalize=0[a]"
            ),
            "-map", "[v]", "-map", "[a]",
            *out_flags, "-c:a", "aac", "-b:a", "192k",
            str(final),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(merged),
            "-vf", sub_filter,
            *out_flags, "-c:a", "copy",
            str(final),
        ]

    _run(cmd)
    _ts(f"Final video: {final}")
    return final


def _guess_genre_id(genre: dict) -> str:
    """Best-effort genre id from the template dict."""
    name = genre.get("name", "")
    return name.lower().replace(" ", "_") if name else "drama_story"
