"""
server/core/style_engine.py

Intelligent Editing Style Engine that analyzes reference videos, extracts
cinematic and short-form pacing patterns, and manages reusable style profiles.
"""

import os
import json
import logging
import uuid
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from server.core.database import SessionLocal, EditingStyleProfile

class CaptionStyle(BaseModel):
    words_per_group: int = 3
    font_name: str = "Arial"
    font_size: int = 24
    primary_color: str = "#FFFFFF" # Hex
    highlight_color: str = "#66FCF1"
    position: str = "lower_middle" # lower_middle, center, upper_middle
    animation: str = "pop" # pop, fade, bounce, none
    bold: bool = True
    outline_width: float = 2.0

class EditingRules(BaseModel):
    avg_shot_duration_sec: float = 2.8
    cut_frequency_per_min: float = 22.0
    adaptive_cutting: bool = True
    semantic_zoom: bool = True
    zoom_scale: float = 1.15
    broll_interval_sec: float = 6.0
    preserve_natural_pauses: bool = True

class AudioStyle(BaseModel):
    duck_music_under_dialogue: bool = True
    music_volume_ducked: float = 0.08
    normalize_loudness: bool = True
    target_lufs: float = -14.0 # Standard for YouTube/TikTok

class StyleProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    target_duration: int = 45
    caption_style: CaptionStyle = Field(default_factory=CaptionStyle)
    editing: EditingRules = Field(default_factory=EditingRules)
    audio: AudioStyle = Field(default_factory=AudioStyle)

# ---------------------------------------------------------------------------
# Default Professional Presets
# ---------------------------------------------------------------------------
DEFAULT_STYLE_PRESETS: List[StyleProfile] = [
    StyleProfile(
        id="preset_documentary_short",
        name="High-Retention Documentary Short",
        description="Cinematic pacing with moderate shot duration, vivid pop captions, and ambient score ducking.",
        target_duration=45,
        caption_style=CaptionStyle(
            words_per_group=3,
            font_name="Montserrat",
            font_size=24,
            primary_color="#FFFFFF",
            highlight_color="#66FCF1",
            position="lower_middle",
            animation="pop"
        ),
        editing=EditingRules(
            avg_shot_duration_sec=3.2,
            cut_frequency_per_min=18.0,
            adaptive_cutting=True,
            semantic_zoom=True,
            broll_interval_sec=5.5
        ),
        audio=AudioStyle(
            duck_music_under_dialogue=True,
            music_volume_ducked=0.07,
            normalize_loudness=True,
            target_lufs=-14.0
        )
    ),
    StyleProfile(
        id="preset_viral_tech",
        name="Viral Fast-Paced Tech",
        description="High-energy cuts with aggressive punch-ins, neon cyber accents, and rapid dialogue flow.",
        target_duration=30,
        caption_style=CaptionStyle(
            words_per_group=2,
            font_name="Impact",
            font_size=28,
            primary_color="#FFFF00",
            highlight_color="#FF0055",
            position="center",
            animation="bounce"
        ),
        editing=EditingRules(
            avg_shot_duration_sec=1.8,
            cut_frequency_per_min=32.0,
            adaptive_cutting=True,
            semantic_zoom=True,
            broll_interval_sec=3.5
        ),
        audio=AudioStyle(
            duck_music_under_dialogue=True,
            music_volume_ducked=0.10,
            normalize_loudness=True,
            target_lufs=-13.0
        )
    ),
    StyleProfile(
        id="preset_minimalist_podcast",
        name="Minimalist Podcast & Dialogue",
        description="Clean conversational rhythm with subtitle clarity and natural breath preservation.",
        target_duration=60,
        caption_style=CaptionStyle(
            words_per_group=4,
            font_name="Inter",
            font_size=22,
            primary_color="#FFFFFF",
            highlight_color="#A1A1AA",
            position="lower_middle",
            animation="none"
        ),
        editing=EditingRules(
            avg_shot_duration_sec=4.5,
            cut_frequency_per_min=13.0,
            adaptive_cutting=False,
            semantic_zoom=False,
            broll_interval_sec=10.0,
            preserve_natural_pauses=True
        ),
        audio=AudioStyle(
            duck_music_under_dialogue=True,
            music_volume_ducked=0.05,
            normalize_loudness=True,
            target_lufs=-16.0
        )
    )
]

def analyze_reference_video(video_path: str, name: str = "Extracted Reference Style") -> StyleProfile:
    """
    Analyzes an uploaded reference video to extract cut pacing, shot duration,
    and visual rhythm without copying copyright footage or branding.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Unable to read video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 30.0

    cuts = 0
    prev_hist = None
    sample_step = max(1, int(fps / 4)) # Sample 4 times per second

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and compute 2D Hue-Saturation Histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if prev_hist is not None:
            # Correlation comparison: drop below 0.6 indicates a camera/shot cut
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity < 0.60:
                cuts += 1
        prev_hist = hist
        frame_idx += sample_step

    cap.release()

    cuts = max(1, cuts)
    avg_shot_dur = round(max(0.8, min(8.0, duration_sec / cuts)), 2)
    cuts_per_min = round((cuts / max(1.0, duration_sec)) * 60.0, 1)

    profile = StyleProfile(
        name=name,
        description=f"Auto-extracted from reference video. Average shot duration: {avg_shot_dur}s ({cuts_per_min} cuts/min).",
        target_duration=int(duration_sec),
        caption_style=CaptionStyle(
            words_per_group=3 if avg_shot_dur > 2.0 else 2,
            position="lower_middle",
            animation="pop" if cuts_per_min > 20 else "fade"
        ),
        editing=EditingRules(
            avg_shot_duration_sec=avg_shot_dur,
            cut_frequency_per_min=cuts_per_min,
            adaptive_cutting=True,
            semantic_zoom=True if cuts_per_min > 18 else False,
            broll_interval_sec=round(avg_shot_dur * 2.0, 1)
        ),
        audio=AudioStyle(
            duck_music_under_dialogue=True,
            normalize_loudness=True
        )
    )

    save_style_profile(profile)
    return profile

def get_all_style_profiles() -> List[StyleProfile]:
    """Retrieves all stored custom profiles and defaults."""
    db = SessionLocal()
    try:
        # Seed defaults if empty
        existing = db.query(EditingStyleProfile).all()
        if not existing:
            for preset in DEFAULT_STYLE_PRESETS:
                db_p = EditingStyleProfile(
                    id=preset.id,
                    name=preset.name,
                    description=preset.description,
                    target_duration=preset.target_duration,
                    config_json=json.dumps(preset.model_dump())
                )
                db.add(db_p)
            db.commit()
            existing = db.query(EditingStyleProfile).all()

        profiles = []
        for p in existing:
            try:
                data = json.loads(p.config_json)
                profiles.append(StyleProfile(**data))
            except Exception:
                continue
        return profiles
    finally:
        db.close()

def save_style_profile(profile: StyleProfile) -> StyleProfile:
    """Saves or updates a style profile in SQLite."""
    db = SessionLocal()
    try:
        record = db.query(EditingStyleProfile).filter(
            (EditingStyleProfile.id == profile.id) | (EditingStyleProfile.name == profile.name)
        ).first()

        json_data = json.dumps(profile.model_dump())
        if record:
            record.name = profile.name
            record.description = profile.description
            record.target_duration = profile.target_duration
            record.config_json = json_data
        else:
            record = EditingStyleProfile(
                id=profile.id,
                name=profile.name,
                description=profile.description,
                target_duration=profile.target_duration,
                config_json=json_data
            )
            db.add(record)
        db.commit()
        return profile
    except Exception as e:
        db.rollback()
        logging.error(f"Error saving style profile: {e}")
        raise e
    finally:
        db.close()

def delete_style_profile(profile_id: str) -> bool:
    """Deletes a custom style profile (preserves default presets)."""
    if profile_id.startswith("preset_"):
        return False
    db = SessionLocal()
    try:
        record = db.query(EditingStyleProfile).filter(EditingStyleProfile.id == profile_id).first()
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
    finally:
        db.close()
