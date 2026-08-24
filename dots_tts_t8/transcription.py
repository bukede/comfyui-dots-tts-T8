"""Pinned Qwen3-ASR transcription, forced alignment, and CAM++ diarization."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .subtitles import SubtitleCue, format_srt

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
PUNCTUATION_END = re.compile(r"[。！？!?；;…][\"'”’）)]*$")
_SNAPSHOT_LOCK = threading.Lock()
_VERIFIED_SNAPSHOTS: dict[tuple[str, str], str] = {}


def _asr_manifest_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parents[1] / "manifests" / "qwen_asr_models.json",
        module_path.parents[2] / "manifests" / "qwen_asr_models.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("缺少 Qwen ASR 固定版本清单 qwen_asr_models.json。")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pinned_asr_snapshot(repo_id: str, revision: str | None) -> str | None:
    if not revision or Path(repo_id).expanduser().exists():
        return None
    manifest = json.loads(_asr_manifest_path().read_text(encoding="utf-8"))
    entry = next(
        (
            value
            for value in manifest.get("models", {}).values()
            if value.get("repo_id") == repo_id and value.get("revision") == revision
        ),
        None,
    )
    if entry is None:
        return None
    key = (repo_id, revision)
    with _SNAPSHOT_LOCK:
        cached = _VERIFIED_SNAPSHOTS.get(key)
        if cached and Path(cached).is_dir():
            return cached
        from huggingface_hub import snapshot_download

        directory = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=sorted(entry["files"]),
            )
        ).resolve()
        for relative, expected in entry["files"].items():
            path = directory / relative
            if not path.is_file():
                raise FileNotFoundError(f"ASR 固定版本缺少文件：{path}")
            if path.stat().st_size != int(expected["size"]):
                raise RuntimeError(f"ASR 文件大小校验失败：{path}")
            if _file_sha256(path) != str(expected["sha256"]).lower():
                raise RuntimeError(f"ASR 文件 SHA-256 校验失败：{path}")
        _VERIFIED_SNAPSHOTS[key] = str(directory)
        return str(directory)


@dataclass(frozen=True)
class TimedToken:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TranscriptionSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"duration_ms": self.duration_ms}


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    segments: tuple[TranscriptionSegment, ...]
    model: str
    model_revision: str | None
    aligner: str | None
    aligner_revision: str | None
    diarized: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["segments"] = [segment.as_dict() for segment in self.segments]
        value["srt"] = transcription_srt(self)
        return value


def _join_token_text(parts: list[str]) -> str:
    output = ""
    for raw in parts:
        piece = str(raw)
        if (
            output
            and piece
            and output[-1].isascii()
            and output[-1].isalnum()
            and piece[0].isascii()
            and piece[0].isalnum()
            and not output.endswith(" ")
            and not piece.startswith(" ")
        ):
            output += " "
        output += piece
    return re.sub(r"[ \t]+", " ", output).strip()


def group_timed_tokens(
    tokens: list[TimedToken],
    *,
    gap_ms: int = 650,
    max_segment_ms: int = 12_000,
    punctuation_min_ms: int = 1_200,
) -> tuple[TranscriptionSegment, ...]:
    if not tokens:
        return ()
    segments: list[TranscriptionSegment] = []
    group: list[TimedToken] = []
    for token in tokens:
        if token.end_ms <= token.start_ms:
            continue
        if group:
            gap = token.start_ms - group[-1].end_ms
            elapsed = token.end_ms - group[0].start_ms
            if gap >= int(gap_ms) or elapsed > int(max_segment_ms):
                text = _join_token_text([item.text for item in group])
                if text:
                    segments.append(
                        TranscriptionSegment(
                            len(segments), group[0].start_ms, group[-1].end_ms, text
                        )
                    )
                group = []
        group.append(token)
        elapsed = group[-1].end_ms - group[0].start_ms
        if elapsed >= int(punctuation_min_ms) and PUNCTUATION_END.search(
            str(token.text).strip()
        ):
            text = _join_token_text([item.text for item in group])
            if text:
                segments.append(
                    TranscriptionSegment(
                        len(segments), group[0].start_ms, group[-1].end_ms, text
                    )
                )
            group = []
    if group:
        text = _join_token_text([item.text for item in group])
        if text:
            segments.append(
                TranscriptionSegment(
                    len(segments), group[0].start_ms, group[-1].end_ms, text
                )
            )
    return tuple(segments)


def _normalized_embeddings(embeddings: Any):
    import numpy as np

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("说话人向量必须是非空二维数组。")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def cluster_speakers(
    embeddings: Any,
    *,
    speaker_count: int = 0,
    similarity_threshold: float = 0.72,
    max_auto_speakers: int = 8,
) -> list[int]:
    import numpy as np

    values = _normalized_embeddings(embeddings)
    count = values.shape[0]
    requested = int(speaker_count)
    if requested < 0 or requested > 32:
        raise ValueError("speaker_count 必须为 0（自动）或 1–32。")
    target = min(max(1, requested), count) if requested else None
    clusters: list[list[int]] = [[index] for index in range(count)]

    def centroid(indices: list[int]):
        value = values[indices].mean(axis=0)
        return value / max(float(np.linalg.norm(value)), 1e-8)

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_similarity = -math.inf
        centroids = [centroid(cluster) for cluster in clusters]
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                similarity = float(np.dot(centroids[left], centroids[right]))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (left, right)
        must_merge = target is not None and len(clusters) > target
        auto_overflow = target is None and len(clusters) > int(max_auto_speakers)
        if (
            not must_merge
            and not auto_overflow
            and best_similarity < float(similarity_threshold)
        ):
            break
        if best_pair is None:
            break
        left, right = best_pair
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    clusters.sort(key=lambda indices: min(indices))
    labels = [0] * count
    for label, indices in enumerate(clusters):
        for index in indices:
            labels[index] = label
    return labels


class CamPlusSpeakerEmbedder:
    """Loads only the speaker encoder artifact from a dots.tts checkpoint."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu") -> None:
        import torch
        from safetensors.torch import load_file

        from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures

        directory = Path(model_dir).expanduser().resolve()
        artifact = directory / "speaker_encoder.safetensors"
        if not artifact.is_file():
            raise FileNotFoundError(f"缺少说话人编码器：{artifact}")
        self.device = torch.device(device)
        self.encoder = SpeakerXVectorFeatures(
            sample_rate=48_000,
            campplus_embedding_size=512,
            max_audio_seconds=0,
        )
        mismatch = self.encoder.load_state_dict(
            load_file(artifact, device="cpu"), strict=False
        )
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"说话人编码器权重不匹配：{mismatch}")
        self.encoder.to(self.device).eval()

    def embed_segments(
        self, audio_path: str | Path, segments: tuple[TranscriptionSegment, ...]
    ):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio.functional as audio_functional

        waveform, sample_rate = sf.read(
            Path(audio_path).expanduser().resolve(), dtype="float32", always_2d=True
        )
        mono = np.mean(waveform, axis=1)
        tensor = torch.from_numpy(mono)
        if int(sample_rate) != 48_000:
            tensor = audio_functional.resample(tensor, int(sample_rate), 48_000)
            sample_rate = 48_000
        outputs = []
        for segment in segments:
            start = max(0, round(segment.start_ms * sample_rate / 1000))
            end = min(tensor.numel(), round(segment.end_ms * sample_rate / 1000))
            clip = tensor[start:end]
            minimum = round(0.5 * sample_rate)
            if clip.numel() < minimum:
                clip = torch.nn.functional.pad(clip, (0, minimum - clip.numel()))
            if clip.numel() > sample_rate * 10:
                center = clip.numel() // 2
                radius = sample_rate * 5
                clip = clip[center - radius : center + radius]
            with torch.inference_mode():
                embedding = self.encoder(clip[None, :].to(self.device))
            outputs.append(embedding.detach().float().cpu().squeeze(0).numpy())
        return np.stack(outputs)


def diarize_segments(
    audio_path: str | Path,
    segments: tuple[TranscriptionSegment, ...],
    speaker_model_dir: str | Path,
    *,
    speaker_count: int = 0,
    similarity_threshold: float = 0.72,
    device: str = "cpu",
) -> tuple[TranscriptionSegment, ...]:
    if not segments:
        return ()
    if int(speaker_count) == 1:
        labels = [0] * len(segments)
    else:
        embedder = CamPlusSpeakerEmbedder(speaker_model_dir, device=device)
        labels = cluster_speakers(
            embedder.embed_segments(audio_path, segments),
            speaker_count=int(speaker_count),
            similarity_threshold=float(similarity_threshold),
        )
    return tuple(
        replace(segment, speaker=f"说话人{labels[index] + 1}")
        for index, segment in enumerate(segments)
    )


def transcription_cues(result: TranscriptionResult) -> tuple[SubtitleCue, ...]:
    return tuple(
        SubtitleCue(
            index=index,
            cue_id=str(index + 1),
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            text=segment.text,
            speaker=segment.speaker,
        )
        for index, segment in enumerate(result.segments)
    )


def transcription_srt(result: TranscriptionResult) -> str:
    return format_srt(transcription_cues(result))


class QwenAsrEngine:
    """Lazy Qwen3-ASR wrapper with immutable default model revisions."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_ASR_MODEL,
        *,
        model_revision: str | None = DEFAULT_ASR_REVISION,
        aligner_name_or_path: str | None = DEFAULT_ALIGNER_MODEL,
        aligner_revision: str | None = DEFAULT_ALIGNER_REVISION,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.model_revision = model_revision
        self.aligner_name_or_path = (
            str(aligner_name_or_path) if aligner_name_or_path else None
        )
        self.aligner_revision = aligner_revision
        self.device = str(device)
        self.max_new_tokens = int(max_new_tokens)
        self._model: Any = None

    @staticmethod
    def _remote_revision(value: str, revision: str | None) -> str | None:
        return None if Path(value).expanduser().exists() else revision

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "ASR 组件未安装；Windows 整合包请重新运行环境准备脚本，"
                "ComfyUI 用户需手动安装 qwen-asr==0.0.6。"
            ) from exc
        model_source = (
            resolve_pinned_asr_snapshot(self.model_name_or_path, self.model_revision)
            or self.model_name_or_path
        )
        aligner_source = (
            resolve_pinned_asr_snapshot(
                self.aligner_name_or_path, self.aligner_revision
            )
            if self.aligner_name_or_path
            else None
        ) or self.aligner_name_or_path
        kwargs: dict[str, Any] = {
            "dtype": torch.float32,
            "device_map": self.device,
            "max_inference_batch_size": 1,
            "max_new_tokens": self.max_new_tokens,
        }
        revision = self._remote_revision(model_source, self.model_revision)
        if revision:
            kwargs["revision"] = revision
        if aligner_source:
            kwargs["forced_aligner"] = aligner_source
            aligner_kwargs: dict[str, Any] = {
                "dtype": torch.float32,
                "device_map": self.device,
            }
            aligner_revision = self._remote_revision(
                aligner_source, self.aligner_revision
            )
            if aligner_revision:
                aligner_kwargs["revision"] = aligner_revision
            kwargs["forced_aligner_kwargs"] = aligner_kwargs
        self._model = Qwen3ASRModel.from_pretrained(model_source, **kwargs)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        diarize: bool = False,
        speaker_count: int = 0,
        speaker_model_dir: str | Path | None = None,
        speaker_similarity_threshold: float = 0.72,
    ) -> TranscriptionResult:
        import soundfile as sf

        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ASR 音频不存在：{source}")
        raw = self._load().transcribe(
            audio=str(source),
            language=(str(language).strip() or None) if language else None,
            return_time_stamps=bool(self.aligner_name_or_path),
        )[0]
        text = str(raw.text).strip()
        if not text:
            raise ValueError("ASR 没有识别到语音。")
        tokens: list[TimedToken] = []
        for item in raw.time_stamps or []:
            value = item if isinstance(item, dict) else vars(item)
            tokens.append(
                TimedToken(
                    text=str(value.get("text", "")),
                    start_ms=max(0, round(float(value.get("start_time", 0)) * 1000)),
                    end_ms=max(0, round(float(value.get("end_time", 0)) * 1000)),
                )
            )
        segments = group_timed_tokens(tokens)
        if not segments:
            info = sf.info(source)
            duration_ms = max(1, round(info.frames / info.samplerate * 1000))
            segments = (TranscriptionSegment(0, 0, duration_ms, text),)
        if diarize:
            if not speaker_model_dir:
                raise ValueError("说话人分离需要 dots.tts 模型目录。")
            segments = diarize_segments(
                source,
                segments,
                speaker_model_dir,
                speaker_count=int(speaker_count),
                similarity_threshold=float(speaker_similarity_threshold),
            )
        return TranscriptionResult(
            text=text,
            language=str(raw.language).strip(),
            segments=segments,
            model=self.model_name_or_path,
            model_revision=self._remote_revision(
                self.model_name_or_path, self.model_revision
            ),
            aligner=self.aligner_name_or_path,
            aligner_revision=(
                self._remote_revision(self.aligner_name_or_path, self.aligner_revision)
                if self.aligner_name_or_path
                else None
            ),
            diarized=bool(diarize),
        )

    def release(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


__all__ = [
    "CamPlusSpeakerEmbedder",
    "DEFAULT_ALIGNER_MODEL",
    "DEFAULT_ALIGNER_REVISION",
    "DEFAULT_ASR_MODEL",
    "DEFAULT_ASR_REVISION",
    "QwenAsrEngine",
    "TimedToken",
    "TranscriptionResult",
    "TranscriptionSegment",
    "cluster_speakers",
    "diarize_segments",
    "group_timed_tokens",
    "resolve_pinned_asr_snapshot",
    "transcription_cues",
    "transcription_srt",
]
