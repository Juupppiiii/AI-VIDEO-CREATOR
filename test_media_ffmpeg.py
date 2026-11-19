import os
import subprocess
import pytest
import shutil
import tempfile

def is_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

@pytest.fixture
def synthetic_video(tmp_path):
    """Generates a 1-second synthetic 640x360 video with test audio."""
    if not is_ffmpeg_available():
        pytest.skip("FFmpeg is not installed on this system.")
        
    out_video = str(tmp_path / "synthetic_sample.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        "-shortest",
        out_video
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    assert os.path.exists(out_video)
    return out_video

def test_ffmpeg_synthetic_generation_and_crop(synthetic_video, tmp_path):
    """Tests 9:16 portrait cropping and scaling filtergraph."""
    portrait_out = str(tmp_path / "portrait_720x1280.mp4")
    # Crop to 9:16 and scale to 720:1280
    cmd = [
        "ffmpeg", "-y",
        "-i", synthetic_video,
        "-vf", "crop=ih*9/16:ih,scale=720:1280",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "copy",
        portrait_out
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    assert os.path.exists(portrait_out)
    assert os.path.getsize(portrait_out) > 0

def test_ffmpeg_audio_ducking_filtergraph(synthetic_video, tmp_path):
    """Tests audio mix and volume ducking filtergraph."""
    bgm_audio = str(tmp_path / "bgm.mp3")
    # Generate 1-second BGM sine tone
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
        "-c:a", "libmp3lame",
        bgm_audio
    ], check=True, capture_output=True)

    mixed_out = str(tmp_path / "mixed_output.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", synthetic_video,
        "-i", bgm_audio,
        "-filter_complex",
        "[1:a]volume=0.15[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=1[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        mixed_out
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Audio mix filtergraph failed: {res.stderr}"
    assert os.path.exists(mixed_out)
