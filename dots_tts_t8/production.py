"""Resumable subtitle, multi-speaker and batch production engine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .media import (
    SubtitleFitMode,
    export_project_media,
    fit_audio_to_duration,
    parse_audio_formats,
)
from .pronunciation import (
    PronunciationRule,
    apply_pronunciation,
    parse_pronunciation_rules,
)
from .subtitles import format_timestamp, parse_plain_text, parse_subtitles

ProjectKind = Literal["text", "subtitle"]


@dataclass(frozen=True)
class VoiceProfile:
    speaker: str
    preset: str | None = None
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    language: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectSegment:
    index: int
    segment_id: str
    original_text: str
    text: str
    speaker: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    pronunciation_replacements: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductionPlan:
    source_name: str
    kind: ProjectKind
    segments: tuple[ProjectSegment, ...]
    pronunciation_rules: tuple[PronunciationRule, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "kind": self.kind,
            "segments": [segment.as_dict() for segment in self.segments],
            "pronunciation_rules": [
                rule.as_dict() for rule in self.pronunciation_rules
            ],
        }


@dataclass(frozen=True)
class GeneratedAudio:
    audio_path: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionResult:
    source_name: str
    project_dir: str
    audio_path: str
    timeline_path: str
    subtitle_path: str | None
    checkpoint_path: str
    succeeded: int
    failed: int
    resumed: int
    status: str
    export_paths: tuple[str, ...] = ()
    video_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchProductionResult:
    projects: tuple[ProductionResult, ...]
    failures: tuple[dict[str, str], ...]
    manifest_path: str
    archive_path: str
    status: str = "completed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "projects": [project.as_dict() for project in self.projects],
            "failures": list(self.failures),
            "manifest_path": self.manifest_path,
            "archive_path": self.archive_path,
            "status": self.status,
        }


ProgressCallback = Callable[[dict[str, Any]], None]
SegmentGenerator = Callable[[ProjectSegment, VoiceProfile | None, int], GeneratedAudio]


class ProductionCancelled(RuntimeError):
    """Raised at a safe segment boundary after the active batch is cancelled."""


class BatchControl:
    """Thread-safe pause/resume/cancel controller for a production queue."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._state = "running"

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    def pause(self) -> str:
        with self._condition:
            if self._state == "running":
                self._state = "paused"
            return self._state

    def resume(self) -> str:
        with self._condition:
            if self._state == "paused":
                self._state = "running"
                self._condition.notify_all()
            return self._state

    def cancel(self) -> str:
        with self._condition:
            self._state = "cancelled"
            self._condition.notify_all()
            return self._state

    def checkpoint(
        self,
        progress: ProgressCallback | None = None,
        **context: Any,
    ) -> None:
        announced = False
        with self._condition:
            while self._state == "paused":
                if progress is not None and not announced:
                    progress({"event": "queue_paused", **context})
                    announced = True
                self._condition.wait(timeout=0.25)
            if self._state == "cancelled":
                raise ProductionCancelled("Production queue was cancelled.")
        if announced and progress is not None:
            progress({"event": "queue_resumed", **context})


def parse_voice_profiles(
    specification: str | Mapping[str, Any] | None,
) -> dict[str, VoiceProfile]:
    if specification is None or specification == "":
        return {}
    value = (
        json.loads(specification) if isinstance(specification, str) else specification
    )
    if not isinstance(value, Mapping):
        raise ValueError("Speaker mapping must be a JSON object.")
    profiles: dict[str, VoiceProfile] = {}
    for raw_speaker, raw_profile in value.items():
        speaker = str(raw_speaker).strip()
        if not speaker:
            raise ValueError("Speaker names cannot be empty.")
        if isinstance(raw_profile, str):
            profile = VoiceProfile(speaker=speaker, preset=raw_profile.strip() or None)
        elif isinstance(raw_profile, Mapping):
            profile = VoiceProfile(
                speaker=speaker,
                preset=str(raw_profile.get("preset", "")).strip() or None,
                prompt_audio_path=(
                    str(
                        raw_profile.get(
                            "prompt_audio_path", raw_profile.get("audio", "")
                        )
                    ).strip()
                    or None
                ),
                prompt_text=(
                    str(
                        raw_profile.get(
                            "prompt_text", raw_profile.get("transcript", "")
                        )
                    ).strip()
                    or None
                ),
                language=str(raw_profile.get("language", "")).strip() or None,
            )
        else:
            raise ValueError(f"Invalid voice profile for speaker {speaker!r}.")
        if profile.prompt_text and not (profile.prompt_audio_path or profile.preset):
            raise ValueError(
                f"Speaker {speaker!r} has a transcript but no audio path or preset."
            )
        profiles[speaker] = profile
    return profiles


def build_production_plan(
    source_name: str,
    source_text: str,
    *,
    kind: Literal["auto", "text", "subtitle"] = "auto",
    pronunciation: str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> ProductionPlan:
    resolved_kind: ProjectKind
    if kind == "auto":
        suffix = Path(source_name).suffix.lower()
        resolved_kind = (
            "subtitle" if suffix in {".srt", ".vtt"} or "-->" in source_text else "text"
        )
    elif kind in {"text", "subtitle"}:
        resolved_kind = kind
    else:
        raise ValueError(f"Unsupported project kind: {kind}")

    rules = parse_pronunciation_rules(pronunciation)
    segments: list[ProjectSegment] = []
    if resolved_kind == "subtitle":
        for cue in parse_subtitles(source_text):
            report = apply_pronunciation(cue.text, rules)
            segments.append(
                ProjectSegment(
                    index=len(segments),
                    segment_id=cue.cue_id,
                    original_text=cue.text,
                    text=report.text,
                    speaker=cue.speaker,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    pronunciation_replacements=report.replacements,
                )
            )
    else:
        for segment in parse_plain_text(source_text):
            report = apply_pronunciation(segment.text, rules)
            segments.append(
                ProjectSegment(
                    index=len(segments),
                    segment_id=str(segment.index + 1),
                    original_text=segment.text,
                    text=report.text,
                    speaker=segment.speaker,
                    pronunciation_replacements=report.replacements,
                )
            )
    return ProductionPlan(source_name, resolved_kind, tuple(segments), rules)


def discover_project_files(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    discovered: dict[str, Path] = {}
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            candidates = path.rglob("*")
        elif path.is_file():
            candidates = (path,)
        else:
            raise FileNotFoundError(f"Batch input does not exist: {path}")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {
                ".txt",
                ".srt",
                ".vtt",
            }:
                discovered[str(candidate).casefold()] = candidate
    if not discovered:
        raise ValueError("No .txt, .srt or .vtt input files were found.")
    return tuple(discovered[key] for key in sorted(discovered))


def _slug(value: str) -> str:
    result = re.sub(r"[^\w.-]+", "-", Path(value).stem, flags=re.UNICODE).strip("-.")
    return (result or "project")[:80]


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_json_safe(item) for item in value]
        return str(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            # Antivirus/indexers can briefly hold a freshly replaced JSON file
            # on Windows. Keep checkpoint writes atomic while tolerating that race.
            time.sleep(0.05 * (attempt + 1))


def _fingerprint(
    plan: ProductionPlan,
    *,
    seed: int,
    segment_gap_ms: int,
    context: Mapping[str, Any] | None,
) -> str:
    payload = {
        "schema": 2,
        "plan": plan.as_dict(),
        "seed": int(seed),
        "segment_gap_ms": int(segment_gap_ms),
        "context": _json_safe(context or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_audio(path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] <= 0:
        raise ValueError(f"Generated segment is empty: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f"Generated segment contains NaN/Inf: {path}")
    return mono, int(sample_rate)


class ProductionRunner:
    """Generate every project segment independently and checkpoint each result."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        plan: ProductionPlan,
        generator: SegmentGenerator,
        *,
        voice_profiles: Mapping[str, VoiceProfile] | None = None,
        fallback_voice: VoiceProfile | None = None,
        seed: int = 42,
        max_retries: int = 2,
        resume: bool = True,
        segment_gap_ms: int = 200,
        fingerprint_context: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
        control: BatchControl | None = None,
        subtitle_fit_mode: SubtitleFitMode = "none",
        audio_formats: str | Sequence[str] | None = None,
        target_lufs: float | None = None,
        video_path: str | Path | None = None,
        ffmpeg_path: str | Path | None = None,
    ) -> ProductionResult:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        if segment_gap_ms < 0:
            raise ValueError("segment_gap_ms cannot be negative.")
        if subtitle_fit_mode not in {"none", "shrink", "fit", "trim"}:
            raise ValueError(f"Unsupported subtitle fit mode: {subtitle_fit_mode}")
        resolved_formats = parse_audio_formats(audio_formats)
        profiles = dict(voice_profiles or {})
        fingerprint = _fingerprint(
            plan,
            seed=seed,
            segment_gap_ms=segment_gap_ms,
            context={
                **dict(fingerprint_context or {}),
                "voices": {key: value.as_dict() for key, value in profiles.items()},
                "fallback_voice": fallback_voice.as_dict() if fallback_voice else None,
                "subtitle_fit_mode": subtitle_fit_mode,
            },
        )
        project_dir = self.output_root / f"{_slug(plan.source_name)}-{fingerprint[:12]}"
        segment_dir = project_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint: dict[str, Any] = {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "source_name": plan.source_name,
            "kind": plan.kind,
            "created_at": time.time(),
            "updated_at": time.time(),
            "segments": {},
        }
        if resume and checkpoint_path.is_file():
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if loaded.get("fingerprint") == fingerprint:
                checkpoint = loaded

        resumed = 0
        generated = 0
        segment_states: dict[str, Any] = checkpoint.setdefault("segments", {})
        for segment in plan.segments:
            if control is not None:
                control.checkpoint(
                    progress,
                    source_name=plan.source_name,
                    index=segment.index,
                    total=len(plan.segments),
                )
            key = str(segment.index)
            destination = segment_dir / f"{segment.index + 1:04d}.wav"
            prior = segment_states.get(key, {})
            if resume and prior.get("status") == "succeeded" and destination.is_file():
                try:
                    _read_audio(destination)
                    resumed += 1
                    if progress:
                        progress(
                            {
                                "event": "segment_resumed",
                                "source_name": plan.source_name,
                                "index": segment.index,
                                "total": len(plan.segments),
                            }
                        )
                    continue
                except Exception:
                    prior = {**prior, "status": "stale"}

            state = {
                **prior,
                "index": segment.index,
                "segment_id": segment.segment_id,
                "speaker": segment.speaker,
                "text": segment.text,
                "seed": (int(seed) + segment.index) & 0xFFFFFFFF,
                "status": "running",
                "attempts": int(prior.get("attempts", 0)),
                "error": None,
            }
            segment_states[key] = state
            checkpoint["updated_at"] = time.time()
            _atomic_json(checkpoint_path, checkpoint)
            profile = profiles.get(segment.speaker or "", fallback_voice)
            for attempt in range(max_retries + 1):
                if control is not None:
                    control.checkpoint(
                        progress,
                        source_name=plan.source_name,
                        index=segment.index,
                        attempt=attempt + 1,
                    )
                state["attempts"] += 1
                if progress:
                    progress(
                        {
                            "event": "segment_started",
                            "source_name": plan.source_name,
                            "index": segment.index,
                            "total": len(plan.segments),
                            "attempt": attempt + 1,
                            "speaker": segment.speaker,
                        }
                    )
                try:
                    output = generator(segment, profile, int(state["seed"]))
                    source = Path(output.audio_path).expanduser().resolve()
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"Generator output is missing: {source}"
                        )
                    if source != destination.resolve():
                        shutil.copy2(source, destination)
                    waveform, sample_rate = _read_audio(destination)
                    state.update(
                        {
                            "status": "succeeded",
                            "error": None,
                            "audio_path": str(destination),
                            "sample_rate": sample_rate,
                            "samples": int(waveform.shape[0]),
                            "metrics": _json_safe(output.metrics),
                        }
                    )
                    generated += 1
                    break
                except Exception as exc:
                    state.update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if progress:
                        progress(
                            {
                                "event": "segment_retry"
                                if attempt < max_retries
                                else "segment_failed",
                                "source_name": plan.source_name,
                                "index": segment.index,
                                "attempt": attempt + 1,
                                "error": state["error"],
                            }
                        )
                finally:
                    checkpoint["updated_at"] = time.time()
                    _atomic_json(checkpoint_path, checkpoint)

        succeeded = sum(
            state.get("status") == "succeeded" for state in segment_states.values()
        )
        failed = len(plan.segments) - succeeded
        if succeeded == 0:
            raise RuntimeError(
                f"Project {plan.source_name!r} has no successfully generated segments."
            )
        audio_path, timeline_path, subtitle_path = self._render(
            plan,
            checkpoint,
            project_dir,
            segment_gap_ms=segment_gap_ms,
            subtitle_fit_mode=subtitle_fit_mode,
        )
        media = export_project_media(
            audio_path,
            audio_formats=resolved_formats,
            target_lufs=target_lufs,
            video_path=video_path,
            ffmpeg_path=ffmpeg_path,
        )
        audio_path = Path(media.primary_audio_path)
        timeline_document = json.loads(timeline_path.read_text(encoding="utf-8"))
        timeline_document["media"] = media.as_dict()
        _atomic_json(timeline_path, timeline_document)
        checkpoint.update(
            {
                "updated_at": time.time(),
                "status": "completed" if failed == 0 else "partial",
                "audio_path": str(audio_path),
                "timeline_path": str(timeline_path),
                "subtitle_path": str(subtitle_path) if subtitle_path else None,
                "media": media.as_dict(),
                "summary": {
                    "total": len(plan.segments),
                    "succeeded": succeeded,
                    "failed": failed,
                    "generated": generated,
                    "resumed": resumed,
                },
            }
        )
        _atomic_json(checkpoint_path, checkpoint)
        if progress:
            progress({"event": "project_completed", **checkpoint["summary"]})
        return ProductionResult(
            source_name=plan.source_name,
            project_dir=str(project_dir),
            audio_path=str(audio_path),
            timeline_path=str(timeline_path),
            subtitle_path=str(subtitle_path) if subtitle_path else None,
            checkpoint_path=str(checkpoint_path),
            succeeded=succeeded,
            failed=failed,
            resumed=resumed,
            status=checkpoint["status"],
            export_paths=media.export_paths,
            video_path=media.video_path,
        )

    @staticmethod
    def _render(
        plan: ProductionPlan,
        checkpoint: Mapping[str, Any],
        project_dir: Path,
        *,
        segment_gap_ms: int,
        subtitle_fit_mode: SubtitleFitMode,
    ) -> tuple[Path, Path, Path | None]:
        import numpy as np
        import soundfile as sf

        states = checkpoint["segments"]
        loaded: dict[int, tuple[Any, int]] = {}
        sample_rates: set[int] = set()
        for segment in plan.segments:
            state = states.get(str(segment.index), {})
            if state.get("status") != "succeeded":
                continue
            audio, sample_rate = _read_audio(Path(state["audio_path"]))
            loaded[segment.index] = (audio, sample_rate)
            sample_rates.add(sample_rate)
        if len(sample_rates) != 1:
            raise ValueError(
                f"Project segment sample rates differ: {sorted(sample_rates)}"
            )
        sample_rate = sample_rates.pop()

        placements: list[tuple[ProjectSegment, int, Any | None, dict[str, Any]]] = []
        cursor = 0
        for segment in plan.segments:
            audio = loaded.get(segment.index, (None, sample_rate))[0]
            fit_info: dict[str, Any] = {
                "mode": "none",
                "time_stretch_rate": 1.0,
                "original_samples": int(audio.shape[0]) if audio is not None else 0,
                "output_samples": int(audio.shape[0]) if audio is not None else 0,
                "target_samples": None,
                "trimmed_samples": 0,
                "padded_samples": 0,
            }
            if (
                plan.kind == "subtitle"
                and audio is not None
                and segment.start_ms is not None
                and segment.end_ms is not None
                and segment.end_ms > segment.start_ms
            ):
                fitted = fit_audio_to_duration(
                    audio,
                    sample_rate,
                    segment.end_ms - segment.start_ms,
                    mode=subtitle_fit_mode,
                )
                audio = fitted.audio
                fit_info = fitted.as_dict()
            if plan.kind == "subtitle":
                start = int(round((segment.start_ms or 0) * sample_rate / 1000.0))
            else:
                start = cursor
            placements.append((segment, start, audio, fit_info))
            if plan.kind == "text" and audio is not None:
                cursor = (
                    start
                    + int(audio.shape[0])
                    + int(round(segment_gap_ms * sample_rate / 1000.0))
                )

        scheduled_end = max(
            (
                int(round((segment.end_ms or 0) * sample_rate / 1000.0))
                for segment in plan.segments
            ),
            default=0,
        )
        generated_end = max(
            (
                start + int(audio.shape[0])
                for _, start, audio, _ in placements
                if audio is not None
            ),
            default=0,
        )
        total_samples = max(1, scheduled_end, generated_end)
        master = np.zeros(total_samples, dtype=np.float32)
        timeline: list[dict[str, Any]] = []
        for segment, start, audio, fit_info in placements:
            state = states.get(str(segment.index), {})
            if audio is not None:
                master[start : start + audio.shape[0]] += audio
            generated_duration_ms = (
                int(round(audio.shape[0] * 1000.0 / sample_rate))
                if audio is not None
                else 0
            )
            slot_ms = (
                (segment.end_ms or 0) - (segment.start_ms or 0)
                if plan.kind == "subtitle"
                else None
            )
            timeline.append(
                {
                    **segment.as_dict(),
                    "status": state.get("status", "missing"),
                    "attempts": state.get("attempts", 0),
                    "audio_path": state.get("audio_path"),
                    "original_generated_duration_ms": int(
                        round(
                            int(fit_info.get("original_samples", 0))
                            * 1000.0
                            / sample_rate
                        )
                    ),
                    "generated_duration_ms": generated_duration_ms,
                    "generated_start_ms": int(round(start * 1000.0 / sample_rate)),
                    "generated_end_ms": int(round(start * 1000.0 / sample_rate))
                    + generated_duration_ms,
                    "overflow_ms": (
                        max(0, generated_duration_ms - slot_ms)
                        if slot_ms is not None
                        else 0
                    ),
                    "error": state.get("error"),
                    "fit": fit_info,
                }
            )
        peak = float(np.max(np.abs(master))) if master.size else 0.0
        gain = 1.0
        if peak > 1.0:
            gain = 0.99 / peak
            master *= gain
        audio_path = project_dir / "master.wav"
        sf.write(audio_path, master, sample_rate)
        timeline_path = project_dir / "timeline.json"
        _atomic_json(
            timeline_path,
            {
                "schema_version": 2,
                "source_name": plan.source_name,
                "kind": plan.kind,
                "sample_rate": sample_rate,
                "samples": int(master.shape[0]),
                "duration_ms": int(round(master.shape[0] * 1000.0 / sample_rate)),
                "peak_before_normalization": peak,
                "normalization_gain": gain,
                "subtitle_fit_mode": subtitle_fit_mode,
                "segments": timeline,
            },
        )
        subtitle_path: Path | None = None
        if plan.kind == "subtitle":
            subtitle_path = project_dir / "timeline.srt"
            blocks = []
            for segment in plan.segments:
                prefix = f"[{segment.speaker}] " if segment.speaker else ""
                blocks.append(
                    f"{segment.segment_id}\n"
                    f"{format_timestamp(segment.start_ms or 0)} --> "
                    f"{format_timestamp(segment.end_ms or 0)}\n"
                    f"{prefix}{segment.original_text}"
                )
            subtitle_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        return audio_path, timeline_path, subtitle_path


def run_batch_production(
    inputs: Sequence[str | Path],
    output_root: str | Path,
    generator: SegmentGenerator,
    *,
    pronunciation: str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    voice_profiles: Mapping[str, VoiceProfile] | None = None,
    fallback_voice: VoiceProfile | None = None,
    seed: int = 42,
    max_retries: int = 2,
    resume: bool = True,
    segment_gap_ms: int = 200,
    fingerprint_context: Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    control: BatchControl | None = None,
    retry_failed_only: bool = False,
    subtitle_fit_mode: SubtitleFitMode = "none",
    audio_formats: str | Sequence[str] | None = None,
    target_lufs: float | None = None,
    video_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
) -> BatchProductionResult:
    files = discover_project_files(inputs)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    batch_identity = hashlib.sha256(
        "\n".join(str(path) for path in files).encode("utf-8")
    ).hexdigest()[:12]
    batch_config_fingerprint = hashlib.sha256(
        json.dumps(
            _json_safe(
                {
                    "pronunciation": pronunciation,
                    "voices": {
                        key: value.as_dict()
                        for key, value in (voice_profiles or {}).items()
                    },
                    "fallback_voice": (
                        fallback_voice.as_dict() if fallback_voice else None
                    ),
                    "seed": int(seed),
                    "max_retries": int(max_retries),
                    "segment_gap_ms": int(segment_gap_ms),
                    "fingerprint_context": fingerprint_context,
                    "subtitle_fit_mode": subtitle_fit_mode,
                    "audio_formats": parse_audio_formats(audio_formats),
                    "target_lufs": target_lufs,
                    "video_path": str(video_path or ""),
                }
            ),
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = root / f"batch-{batch_identity}.json"
    runner = ProductionRunner(root)
    projects: list[ProductionResult] = []
    failures: list[dict[str, str]] = []
    selected_files = files
    if retry_failed_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "No prior batch manifest exists for failed-only retry."
            )
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != batch_config_fingerprint:
            raise ValueError(
                "Failed-only retry requires the same voice, text-processing, seed, "
                "timeline and export settings as the prior batch."
            )
        projects = [ProductionResult(**item) for item in prior.get("projects", [])]
        failed_paths = {str(item.get("path", "")) for item in prior.get("failures", [])}
        selected_files = tuple(path for path in files if str(path) in failed_paths)
    if video_path and len(files) != 1:
        raise ValueError(
            "Video remux requires a batch containing exactly one project file."
        )
    batch_status = "completed"
    original_indices = {str(path): index for index, path in enumerate(files)}
    for file_index, path in enumerate(selected_files):
        original_index = original_indices[str(path)]
        if control is not None:
            try:
                control.checkpoint(
                    progress,
                    path=str(path),
                    index=file_index,
                    total=len(selected_files),
                )
            except ProductionCancelled:
                batch_status = "cancelled"
                break
        if progress:
            progress(
                {
                    "event": "project_started",
                    "index": file_index,
                    "total": len(selected_files),
                    "path": str(path),
                }
            )
        try:
            source_text = path.read_text(encoding="utf-8-sig")
            plan = build_production_plan(
                path.name,
                source_text,
                kind="auto",
                pronunciation=pronunciation,
            )
            projects.append(
                runner.run(
                    plan,
                    generator,
                    voice_profiles=voice_profiles,
                    fallback_voice=fallback_voice,
                    seed=(int(seed) + original_index * 100_000) & 0xFFFFFFFF,
                    max_retries=max_retries,
                    resume=resume,
                    segment_gap_ms=segment_gap_ms,
                    fingerprint_context=fingerprint_context,
                    progress=progress,
                    control=control,
                    subtitle_fit_mode=subtitle_fit_mode,
                    audio_formats=audio_formats,
                    target_lufs=target_lufs,
                    video_path=video_path,
                    ffmpeg_path=ffmpeg_path,
                )
            )
        except ProductionCancelled:
            batch_status = "cancelled"
            break
        except Exception as exc:
            failures.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
        _atomic_json(
            manifest_path,
            {
                "schema_version": 2,
                "inputs": [str(item) for item in files],
                "projects": [project.as_dict() for project in projects],
                "failures": failures,
                "status": batch_status,
                "config_fingerprint": batch_config_fingerprint,
                "retry_failed_only": bool(retry_failed_only),
                "updated_at": time.time(),
            },
        )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 2,
            "inputs": [str(item) for item in files],
            "projects": [project.as_dict() for project in projects],
            "failures": failures,
            "status": batch_status,
            "config_fingerprint": batch_config_fingerprint,
            "retry_failed_only": bool(retry_failed_only),
            "updated_at": time.time(),
        },
    )
    if not projects and batch_status != "cancelled":
        raise RuntimeError("Batch production did not complete any project.")
    archive_path = root / f"batch-{batch_identity}.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(manifest_path, manifest_path.name)
        for project in projects:
            project_root = Path(project.project_dir)
            for path in project_root.rglob("*"):
                if path.is_file():
                    archive.write(
                        path,
                        str(Path(project_root.name) / path.relative_to(project_root)),
                    )
    return BatchProductionResult(
        tuple(projects),
        tuple(failures),
        str(manifest_path),
        str(archive_path),
        batch_status,
    )


__all__ = [
    "BatchControl",
    "BatchProductionResult",
    "GeneratedAudio",
    "ProductionPlan",
    "ProductionCancelled",
    "ProductionResult",
    "ProductionRunner",
    "ProjectSegment",
    "VoiceProfile",
    "build_production_plan",
    "discover_project_files",
    "parse_voice_profiles",
    "run_batch_production",
]
