import pytest
import time
from server.core.queue import (
    create_job, update_job, get_job, stop_job, add_log, 
    save_stage_checkpoint, recover_interrupted_jobs, list_jobs
)
from server.core.database import SessionLocal, Job as DBJob

def test_job_lifecycle_persistence():
    # 1. Create job
    job_id = create_job("Test AI Autonomous Storytelling", job_type="pipeline")
    assert job_id is not None
    
    # 2. Check initial state
    job = get_job(job_id)
    assert job["id"] == job_id
    assert job["topic"] == "Test AI Autonomous Storytelling"
    assert job["status"] == "queued"
    assert job["progress"] == 0.0

    # 3. Update status and progress
    update_job(job_id, status="running", progress=0.25, step="Generating Script")
    job = get_job(job_id)
    assert job["status"] == "running"
    assert job["progress"] == 0.25
    assert job["step"] == "Generating Script"

    # 4. Add logs
    add_log(job_id, "Script research completed successfully.")
    job = get_job(job_id)
    assert any("Script research completed" in log for log in job["logs"])

    # 5. Checkpoints
    save_stage_checkpoint(job_id, "script_generation", {"characters": 240, "scenes": 4})

    # 6. Mark completed
    update_job(job_id, status="completed", progress=1.0, result="workspace/test/final.mp4")
    job = get_job(job_id)
    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["result"] == "workspace/test/final.mp4"

def test_job_cancellation():
    job_id = create_job("Test Job Cancellation")
    stop_job(job_id)
    job = get_job(job_id)
    assert job["status"] == "cancelled"

def test_recovery_of_interrupted_jobs():
    # Simulate a crashed job in the database
    db = SessionLocal()
    crashed_id = "crashed-test-job-999"
    try:
        crashed_job = DBJob(
            id=crashed_id,
            topic="Crashed Pipeline Job",
            status="running",
            progress=50,
            step="Rendering video"
        )
        db.merge(crashed_job)
        db.commit()
    finally:
        db.close()

    # Run recovery
    recover_interrupted_jobs()

    # Verify transition to interrupted
    recovered = get_job(crashed_id)
    assert recovered["status"] == "interrupted"
    assert "interrupted" in recovered["error"].lower()
