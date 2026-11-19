import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from server.core.timeline_engine import (
    ProjectTimeline, create_default_multitrack_timeline,
    edit_transcript_text, process_conversational_ai_edit
)

router = APIRouter()

class CreateTimelineRequest(BaseModel):
    name: str = "New Timeline Project"
    duration: float = 45.0
    aspect_ratio: str = "9:16"

class TranscriptEditRequest(BaseModel):
    timeline: ProjectTimeline
    clip_id: str
    action: str = "delete" # 'delete', 'trim', 'replace'
    new_text: Optional[str] = None
    new_duration: Optional[float] = None

class AICommandRequest(BaseModel):
    timeline: ProjectTimeline
    command: str

@router.post("/create")
def create_timeline(req: CreateTimelineRequest) -> Dict[str, Any]:
    timeline = create_default_multitrack_timeline(duration=req.duration, aspect_ratio=req.aspect_ratio)
    timeline.name = req.name
    return {"message": "Timeline created.", "timeline": timeline.model_dump()}

@router.post("/transcript-edit")
def handle_transcript_edit(req: TranscriptEditRequest) -> Dict[str, Any]:
    """Text-based video editing: cuts and ripples video based on transcript changes."""
    updated = edit_transcript_text(
        timeline=req.timeline,
        clip_id=req.clip_id,
        action=req.action,
        new_text=req.new_text,
        new_duration=req.new_duration
    )
    return {"message": "Transcript edit applied.", "timeline": updated.model_dump()}

@router.post("/ai-command")
def handle_ai_command(req: AICommandRequest) -> Dict[str, Any]:
    """Executes conversational natural language AI editing commands."""
    res = process_conversational_ai_edit(timeline=req.timeline, user_command=req.command)
    return res
