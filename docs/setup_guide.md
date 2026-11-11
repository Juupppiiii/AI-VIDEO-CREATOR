# Setup Guide

## Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with 6 GB+ VRAM (tested on RTX 4050)
- **Python 3.11+**
- **FFmpeg** in PATH — install with: `winget install Gyan.FFmpeg`
- **Ollama** — install from https://ollama.com/download

## Installation

### 1. Install PyTorch with CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Install project dependencies

From the project root:

```powershell
pip install -e server/
```

Or without editable install:

```powershell
pip install -r server/pyproject.toml
```

### 3. Pull the LLM

```powershell
ollama pull qwen2.5:7b
```

### 4. (Optional) Add background music

Drop royalty-free `.mp3` or `.wav` files into the `music/` subdirectories:

```
music/
├── cheerful/       ← kids, trivia
├── dramatic/       ← drama stories
├── suspenseful/    ← horror, mystery
├── inspirational/  ← motivational
└── mysterious/     ← mystery
```

## Running

### Web UI (recommended)

```powershell
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2
python server/app/main.py
```

Opens at http://127.0.0.1:7860

### CLI

```powershell
# Drama story
python server/_cli.py --genre drama_story "A mother discovers her son's secret"

# Kids educational
python server/_cli.py --genre kids_educational "Why is the sky blue?" --age 4-6

# Trivia quiz
python server/_cli.py --genre trivia_quiz "Amazing animal facts"

# See all genres
python server/_cli.py --list-genres

# Batch mode
python server/_cli.py --topics-file topics.txt --genre horror_story
```

### Update trends (needs internet)

```powershell
python server/_update_trends.py
```

## First run

The first run downloads the SDXL image model (~6 GB). This is cached locally —
subsequent runs start much faster. Kokoro TTS models are also downloaded on
first use (~200 MB).

## Troubleshooting

| Issue                       | Fix                                                |
|-----------------------------|----------------------------------------------------|
| `CUDA GPU not found`        | Install the CUDA build of PyTorch (step 1)         |
| `Could not reach Ollama`    | Run `ollama serve` in another terminal              |
| `ffmpeg not found`          | `winget install Gyan.FFmpeg`, then restart terminal |
| Out of VRAM                 | Close other GPU apps; ensure `USE_CPU_OFFLOAD=True` |
| Slow image generation       | Normal first run; cached model loads faster after   |
