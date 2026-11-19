# AI Video Creator

Automated video production pipeline. Generates scripts, images, voiceover, and assembles
finished videos — all locally on your GPU. Supports multiple genres and YouTube highlight clipping.

```
Topic → Script (Ollama) → Images (SD 1.5) → Voice (Kokoro TTS)
      → Assembly (FFmpeg + transitions + SFX + captions) → MP4
```

## Genres

| Genre | Style |
|-------|-------|
| comic_documentary | Flat 2D narrator in 3D environments — dry humor documentary |
| whatif_story | What-if animated stories — flat 2D characters in 3D worlds |
| drama_story | Dramatic shorts with twists — Drama Box style |
| kids_educational | Colorful learning videos for ages 3-8 |
| trivia_quiz | Addictive quiz videos with surprising facts |
| horror_story | Atmospheric dread with chilling reveals |
| motivational | Inspiring stories of resilience |
| mystery | Whodunit puzzles that challenge the viewer |
| stickman | Stickman animation style |

## Quick Start

```powershell
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install -e .

# 3. Pull the LLM
ollama pull qwen2.5:7b

# 4. Start Ollama
ollama serve

# 5. Launch the web UI
python app/main.py
```

Or double-click `launch.bat` (auto-starts Ollama + opens the UI).

## Project Structure

```
├── app/                     Gradio web UI
│   └── main.py              Main application
├── pipeline/                Core video generation
│   ├── config.py            Central configuration
│   ├── script.py            LLM script generation + refinement
│   ├── images.py            SD 1.5 image generation + upscale
│   ├── voice.py             Kokoro TTS (multi-voice)
│   ├── assembler.py         FFmpeg assembly + transitions + captions
│   ├── animate.py           AnimateDiff-Lightning animated clips
│   ├── sfx.py               Sound effects system
│   ├── thumbnail.py         Thumbnail + intro/outro generation
│   ├── youtube.py           YouTube highlight clipper
│   ├── orchestrator.py      Stage-based pipeline with checkpoints
│   ├── upload_package.py    Title + description + hashtag generator
│   └── qa.py                Output quality checks
├── genres/templates/        Genre JSON definitions
├── trends/                  Trending topic fetcher
├── reference/               Reference materials (zips, docs, models)
├── music/                   Background music by mood
├── sfx/                     User-provided sound effects
├── docs/                    Documentation
├── output/                  Generated videos (gitignored)
├── data/                    Cached data (gitignored)
├── _cli.py                  CLI entry point
├── _update_trends.py        Trend cache updater
├── launch.bat               Windows one-click launcher
└── pyproject.toml           Python package definition
```

## Pipeline Stages

The orchestrator runs each stage with checkpoints. If a stage crashes,
earlier work is preserved and the pipeline resumes from the failure point.

```
images → voice → assemble → upload_package → qa
```

## Requirements

- Windows 10/11, NVIDIA GPU 6 GB+ VRAM, Python 3.11+
- FFmpeg, Ollama

See [docs/setup_guide.md](docs/setup_guide.md) for full setup instructions.



####error

[17:05:02] ERROR in stage images:
Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\orchestrator.py", line 288, in execute
    runner()
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\orchestrator.py", line 156, in _run_images
    self.images = generate_animated_clips(
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\animate.py", line 171, in generate_animated_clips
    raise RuntimeError(f"AnimateDiff failed to load:\n{error['tb']}")
RuntimeError: AnimateDiff failed to load:
Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\animate.py", line 119, in _load_pipe
    pipe.enable_vae_tiling()       # fixes 864 MiB VAE decode OOM
    ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\diffusers\configuration_utils.py", line 175, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'AnimateDiffPipeline' object has no attribute 'enable_vae_tiling'


ERROR:
Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\orchestrator.py", line 288, in execute
    runner()
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\orchestrator.py", line 156, in _run_images
    self.images = generate_animated_clips(
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\animate.py", line 171, in generate_animated_clips
    raise RuntimeError(f"AnimateDiff failed to load:\n{error['tb']}")
RuntimeError: AnimateDiff failed to load:
Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\animate.py", line 119, in _load_pipe
    pipe.enable_vae_tiling()       # fixes 864 MiB VAE decode OOM
    ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\diffusers\configuration_utils.py", line 175, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'AnimateDiffPipeline' object has no attribute 'enable_vae_tiling'


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\app\main.py", line 181, in work
    result["final"] = run.execute(resume_from=resume)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\orchestrator.py", line 300, in execute
    raise RuntimeError(error_msg) from exc
RuntimeError: images failed: AnimateDiff failed to load:
Traceback (most recent call last):
  File "C:\Users\ACHI\Downloads\AI-VIDEO-CREATOR-main\pipeline\animate.py", line 119, in _load_pipe
    pipe.enable_vae_tiling()       # fixes 864 MiB VAE decode OOM
    ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ACHI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\diffusers\configuration_utils.py", line 175, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'AnimateDiffPipeline' object has no attribute 'enable_vae_tiling'

