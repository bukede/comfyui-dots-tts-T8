from __future__ import annotations

import threading
from typing import Any

from dots_tts_t8.asr_sidecar import (
    sidecar_health,
    sidecar_release,
    sidecar_transcribe,
)
from dots_tts_t8.transcription import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ALIGNER_REVISION,
    DEFAULT_ASR_MODEL,
    DEFAULT_ASR_REVISION,
    QwenAsrEngine,
    TranscriptionResult,
    TranscriptionSegment,
    diarize_segments,
)

from .audio_adapter import prompt_audio_file
from .types import ModelHandle

_LOCK = threading.RLock()
_ENGINE: QwenAsrEngine | None = None
_ENGINE_KEY: tuple[Any, ...] | None = None


def run_transcription(
    audio: dict,
    *,
    asr_model: str = DEFAULT_ASR_MODEL,
    aligner_model: str | None = DEFAULT_ALIGNER_MODEL,
    language: str | None = None,
    diarize: bool = True,
    speaker_count: int = 0,
    similarity_threshold: float = 0.72,
    device: str = "cuda:0",
    speaker_model: ModelHandle | None = None,
    transport: str = "auto",
    release_after_run: bool = True,
) -> dict[str, Any]:
    global _ENGINE, _ENGINE_KEY
    model = asr_model.strip() or DEFAULT_ASR_MODEL
    aligner = (aligner_model or "").strip() or None
    model_revision = DEFAULT_ASR_REVISION if model == DEFAULT_ASR_MODEL else None
    aligner_revision = (
        DEFAULT_ALIGNER_REVISION if aligner == DEFAULT_ALIGNER_MODEL else None
    )
    key = (model, model_revision, aligner, aligner_revision, device)
    if diarize and speaker_model is None:
        raise ValueError("启用说话人分离时必须连接 dots.tts 模型。")
    if model != DEFAULT_ASR_MODEL or aligner != DEFAULT_ALIGNER_MODEL:
        raise ValueError("ASR 只接受固定清单中的官方模型与 revision。")
    if transport not in {"auto", "sidecar", "in_process_legacy"}:
        raise ValueError("ASR transport 必须为 auto、sidecar 或 in_process_legacy。")
    with _LOCK, prompt_audio_file(audio) as path:
        selected_transport = transport
        if selected_transport == "auto":
            try:
                sidecar_health()
                selected_transport = "sidecar"
            except Exception:
                selected_transport = "in_process_legacy"
        if selected_transport == "sidecar":
            try:
                report = sidecar_transcribe(
                    path, language=(language or "").strip() or None
                )
                if diarize:
                    assert speaker_model is not None
                    segments = tuple(
                        TranscriptionSegment(
                            index=int(item["index"]),
                            start_ms=int(item["start_ms"]),
                            end_ms=int(item["end_ms"]),
                            text=str(item["text"]),
                            speaker=item.get("speaker"),
                        )
                        for item in report["segments"]
                    )
                    segments = diarize_segments(
                        path,
                        segments,
                        speaker_model.model_dir,
                        speaker_count=int(speaker_count),
                        similarity_threshold=float(similarity_threshold),
                    )
                    result = TranscriptionResult(
                        text=str(report["text"]),
                        language=str(report["language"]),
                        segments=segments,
                        model=str(report["model"]),
                        model_revision=report.get("model_revision"),
                        aligner=report.get("aligner"),
                        aligner_revision=report.get("aligner_revision"),
                        diarized=True,
                    )
                    report = result.as_dict()
                report["transport"] = "sidecar"
                return report
            finally:
                if release_after_run:
                    try:
                        sidecar_release()
                    except Exception:
                        pass
        try:
            if _ENGINE is None or _ENGINE_KEY != key:
                if _ENGINE is not None:
                    _ENGINE.release()
                _ENGINE = QwenAsrEngine(
                    model,
                    model_revision=model_revision,
                    aligner_name_or_path=aligner,
                    aligner_revision=aligner_revision,
                    device=device,
                )
                _ENGINE_KEY = key
            result = _ENGINE.transcribe(
                path,
                language=(language or "").strip() or None,
                diarize=bool(diarize),
                speaker_count=int(speaker_count),
                speaker_model_dir=(
                    speaker_model.model_dir if speaker_model is not None else None
                ),
                speaker_similarity_threshold=float(similarity_threshold),
            )
            report = result.as_dict()
            report["transport"] = "in_process_legacy"
            return report
        finally:
            if release_after_run:
                release_transcription()


def release_transcription() -> bool:
    global _ENGINE, _ENGINE_KEY
    with _LOCK:
        engine = _ENGINE
        _ENGINE = None
        _ENGINE_KEY = None
        sidecar_released = False
        try:
            sidecar_released = sidecar_release()
        except Exception:
            pass
        if engine is not None:
            engine.release()
            return True
        return sidecar_released


__all__ = ["release_transcription", "run_transcription"]
