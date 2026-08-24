from __future__ import annotations

import threading
from typing import Any

from dots_tts_t8.transcription import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ALIGNER_REVISION,
    DEFAULT_ASR_MODEL,
    DEFAULT_ASR_REVISION,
    QwenAsrEngine,
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
    with _LOCK:
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
        with prompt_audio_file(audio) as path:
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
        return result.as_dict()


def release_transcription() -> bool:
    global _ENGINE, _ENGINE_KEY
    with _LOCK:
        engine = _ENGINE
        _ENGINE = None
        _ENGINE_KEY = None
        if engine is not None:
            engine.release()
            return True
        return False


__all__ = ["release_transcription", "run_transcription"]
