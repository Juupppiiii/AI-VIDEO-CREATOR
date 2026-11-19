"""
Stage-based pipeline orchestrator with checkpoints.

Inspired by Agnes/EchoSync: runs the video pipeline stage-by-stage,
saving a checkpoint after each. If a later stage crashes, earlier work
is preserved and the pipeline can resume without re-doing everything.

Stages: script → images → voice → assemble → qa
"""
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

_log = print


def set_logger(fn):
    global _log
    _log = fn


def _ts(msg: str):
    _log(f"[{time.strftime('%H:%M:%S')}] {msg}")


STAGES = ["images", "voice", "assemble", "upload_package", "qa"]


class PipelineRun:
    """One end-to-end video render with checkpoint support.

    Usage::

        run = PipelineRun(outdir=..., scenes=..., genre=..., ...)
        run.execute()            # runs all stages
        run.execute("voice")     # resumes from voice stage (images are kept)
    """

    def __init__(
        self,
        outdir: Path,
        scenes: list[dict],
        genre: dict[str, Any],
        title: str = "",
        seed: int = 1234,
        speed: float = 0.9,
        out_w: int = 1080,
        out_h: int = 1920,
        img_w: int = 432,
        img_h: int = 768,
        use_anim: bool = False,
        # caption settings
        caption_font: str = "Arial",
        caption_size: int = 0,
        caption_color: str = "white",
        caption_bold: bool = True,
        caption_style: str = "box",
        caption_position: str = "bottom",
    ):
        self.outdir = Path(outdir)
        self.scenes = scenes
        self.genre = genre
        self.title = title
        self.seed = seed
        self.speed = speed
        self.out_w = out_w
        self.out_h = out_h
        self.img_w = img_w
        self.img_h = img_h
        self.use_anim = use_anim
        self.caption_font = caption_font
        self.caption_size = caption_size
        self.caption_color = caption_color
        self.caption_bold = caption_bold
        self.caption_style = caption_style
        self.caption_position = caption_position

        self.checkpoint_path = self.outdir / "checkpoint.json"
        self.state = self._load()

        # Populated by stages
        self.images: list[Path] = []
        self.audios: list[Path] = []
        self.durations: list[float] = []
        self.final: Path | None = None

    # ── checkpoint I/O ──────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.checkpoint_path.exists():
            try:
                return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"completed": [], "images": [], "audios": [], "durations": []}

    def _save(self):
        self.state["images"] = [str(p) for p in self.images]
        self.state["audios"] = [str(p) for p in self.audios]
        self.state["durations"] = self.durations
        self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _mark_done(self, stage: str):
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        self.state.pop("failed_stage", None)
        self.state.pop("error", None)
        self._save()

    def _mark_failed(self, stage: str, error: str):
        self.state["failed_stage"] = stage
        self.state["error"] = error
        self._save()

    # ── stage runners ───────────────────────────────────────────────────

    def _run_images(self):
        _ts("Stage: Image generation")
        if self.use_anim:
            from pipeline.animate import generate_animated_clips
            self.images = generate_animated_clips(
                self.scenes, self.genre, self.outdir, self.seed,
                img_w=self.img_w, img_h=self.img_h,
            )
        else:
            from pipeline.images import generate_images
            self.images = generate_images(
                self.scenes, self.genre, self.outdir, self.seed,
                img_w=self.img_w, img_h=self.img_h,
            )

    def _run_voice(self):
        _ts("Stage: Voice generation")
        from pipeline.voice import generate_voice
        self.audios, self.durations = generate_voice(
            self.scenes, self.genre, self.outdir, self.speed,
        )

    def _run_assemble(self):
        _ts("Stage: Video assembly")
        from pipeline.assembler import assemble
        self.final = assemble(
            self.scenes, self.images, self.audios, self.durations,
            self.genre, self.outdir,
            title=self.title,
            out_w=self.out_w, out_h=self.out_h,
            caption_font=self.caption_font,
            caption_size=self.caption_size,
            caption_color=self.caption_color,
            caption_bold=self.caption_bold,
            caption_style=self.caption_style,
            caption_position=self.caption_position,
        )

    def _run_qa(self):
        _ts("Stage: Output QA")
        from pipeline.qa import check_video
        issues = check_video(self.final, self.outdir)
        if issues:
            for issue in issues:
                _ts(f"  QA warning: {issue}")
        else:
            _ts("  QA passed — no issues found")

    def _run_upload_package(self):
        _ts("Stage: Upload package (titles, description, hashtags)")
        from pipeline.upload_package import generate_upload_package
        # Read the saved script
        script_path = self.outdir / "script.json"
        if script_path.exists():
            script = json.loads(script_path.read_text(encoding="utf-8"))
        else:
            script = {"title": self.title, "scenes": self.scenes}
        generate_upload_package(script, self.genre, self.outdir)

    # ── restore from checkpoint ─────────────────────────────────────────

    def _restore_from_checkpoint(self):
        """Load images/audios/durations from a previous run's checkpoint."""
        if self.state.get("images"):
            self.images = [Path(p) for p in self.state["images"]]
            _ts(f"  Restored {len(self.images)} images from checkpoint")
        if self.state.get("audios"):
            self.audios = [Path(p) for p in self.state["audios"]]
            self.durations = self.state.get("durations", [])
            _ts(f"  Restored {len(self.audios)} audio files from checkpoint")

    # ── main entry point ────────────────────────────────────────────────

    def execute(self, resume_from: str | None = None) -> Path:
        """Run the pipeline. Optionally resume from a specific stage.

        Returns the path to the final video.
        """
        _ts(f"Pipeline starting (outdir: {self.outdir.name})")

        # If resuming, restore previous stage outputs from checkpoint
        if resume_from:
            self._restore_from_checkpoint()
            _ts(f"Resuming from stage: {resume_from}")

        runners = {
            "images":          self._run_images,
            "voice":           self._run_voice,
            "assemble":        self._run_assemble,
            "upload_package":  self._run_upload_package,
            "qa":              self._run_qa,
        }

        skip = True if resume_from else False

        for stage in STAGES:
            # Skip stages until we reach the resume point
            if skip:
                if stage == resume_from:
                    skip = False
                else:
                    if stage in self.state.get("completed", []):
                        _ts(f"Skipping {stage} (already completed)")
                        continue

            runner = runners.get(stage)
            if not runner:
                continue

            try:
                runner()
                self._mark_done(stage)
                _ts(f"Stage {stage} completed")
            except Exception as exc:
                error_msg = f"{stage} failed: {exc}"
                _ts(f"ERROR in stage {stage}:\n{traceback.format_exc()}")
                self._mark_failed(stage, str(exc))
                raise RuntimeError(error_msg) from exc

        _ts(f"Pipeline complete: {self.final}")
        return self.final

    # ── convenience ─────────────────────────────────────────────────────

    @property
    def last_failed_stage(self) -> str | None:
        return self.state.get("failed_stage")

    @property
    def completed_stages(self) -> list[str]:
        return self.state.get("completed", [])

    @property
    def can_resume(self) -> bool:
        return bool(self.last_failed_stage)
