# Architecture

## Pipeline overview

```
User picks Genre + Topic
        │
        ▼
┌──────────────────┐
│  Script (Ollama)  │  Genre template provides system prompt + rules
└────────┬─────────┘
         │  JSON: title, description, scenes[{narration, image_prompt, speaker, mood}]
         ▼
┌──────────────────┐
│  Images (SDXL)    │  Genre template provides image_style + negative_prompt
└────────┬─────────┘  model_cpu_offload() keeps peak VRAM ~4.5 GB
         │
         ▼
┌──────────────────┐
│  Voice (Kokoro)   │  Multi-speaker via voice_map: speaker → Kokoro voice id
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Assembly (FFmpeg)│  Ken Burns + xfade transitions + captions + music
└────────┬─────────┘
         │
         ▼
    final.mp4 + metadata.json
```

## Genre template system

Each genre is a JSON file in `server/genres/templates/`. It defines:

| Field               | Purpose                                         |
|---------------------|-------------------------------------------------|
| script_system_prompt| LLM system message (tone, rules)                |
| script_user_template| LLM user message with `{topic}`, `{scenes}` etc|
| image_style         | Appended to every image prompt                  |
| negative_prompt     | Appended to the global negative prompt           |
| voice, voice_speed  | Default voice settings                          |
| voice_map           | speaker label → Kokoro voice id                 |
| music_mood          | Subdirectory in `music/` to pick background from|
| target_scenes       | Default scene count for this genre              |
| made_for_kids       | YouTube COPPA flag                              |

## VRAM management strategy

Target: RTX 4050 with 6 GB VRAM, 8 GB system RAM.

1. **Sequential loading** — only one model in GPU memory at a time.
2. **Ollama `keep_alive: 0`** — unload the LLM immediately after script generation.
3. **SDXL `model_cpu_offload()`** — moves UNet/VAE to GPU layer by layer. Peak ~4.5 GB.
4. **Explicit `del pipe; torch.cuda.empty_cache()`** between stages.
5. **Kokoro runs on CPU** — no GPU needed, small enough for 8 GB RAM.

## Directory layout

```
AI-VIDEO-CREATOR/
├── docs/                    Documentation
├── server/                  Main application
│   ├── pyproject.toml       Python package definition
│   ├── app/                 Gradio web UI
│   ├── pipeline/            Core generation modules
│   ├── genres/              Genre templates
│   ├── trends/              Trend fetcher
│   ├── tests/               Test suite
│   ├── _cli.py              CLI entry point
│   └── _update_trends.py    Trend cache updater
├── music/                   Background music by mood
├── output/                  Generated videos (gitignored)
└── data/                    Cached trends (gitignored)
```
