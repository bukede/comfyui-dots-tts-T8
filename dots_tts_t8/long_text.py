"""Tokenizer-aware long-text audio assembly helpers.

The module deliberately imports torch only while assembling audio so importing a
ComfyUI node does not initialize the inference stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .text_planner import SpeechPlan


@dataclass(frozen=True)
class AudioAssembly:
    """A mono/stereo waveform assembled from a planned sequence of segments."""

    waveform: Any
    sample_rate: int
    segment_samples: tuple[int, ...]
    silence_samples: int

    @property
    def total_samples(self) -> int:
        return int(self.waveform.shape[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "total_samples": self.total_samples,
            "segment_samples": list(self.segment_samples),
            "silence_samples": self.silence_samples,
        }


def tokenizer_counter(tokenizer: Any) -> Callable[[str], int]:
    """Build a token counter for Hugging Face-compatible tokenizers."""

    def count(text: str) -> int:
        if hasattr(tokenizer, "encode"):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
        elif callable(tokenizer):
            try:
                encoded = tokenizer(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer(text)
        else:
            raise TypeError("The selected model tokenizer is not callable.")

        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded.get("input_ids")
        if encoded is None:
            raise ValueError("The selected model tokenizer returned no input_ids.")
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if (
            isinstance(encoded, (list, tuple))
            and len(encoded) == 1
            and isinstance(encoded[0], (list, tuple))
        ):
            encoded = encoded[0]
        try:
            return len(encoded)
        except TypeError as exc:
            raise TypeError("Unable to count tokens from tokenizer output.") from exc

    return count


def _normalize_segment_audio(audio: Any):
    try:
        waveform = audio.detach().float().cpu()
    except AttributeError as exc:
        raise TypeError("Segment audio must be a torch.Tensor-like value.") from exc
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    elif waveform.ndim != 2:
        raise ValueError(f"Unexpected segment waveform shape: {tuple(waveform.shape)}")
    if waveform.shape[-1] <= 0:
        raise ValueError("A generated long-text segment is empty.")
    return waveform


def assemble_speech_audio(
    plan: SpeechPlan,
    segment_audio: Sequence[Any],
    *,
    sample_rate: int,
) -> AudioAssembly:
    """Insert planned silence and concatenate generated segment waveforms."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if not plan.segments:
        raise ValueError("The speech plan contains no speakable text.")
    if len(segment_audio) != len(plan.segments):
        raise ValueError(
            "Generated segment count does not match the speech plan: "
            f"expected={len(plan.segments)} actual={len(segment_audio)}"
        )

    import torch

    waveforms = [_normalize_segment_audio(audio) for audio in segment_audio]
    channel_count = int(waveforms[0].shape[0])
    if any(int(waveform.shape[0]) != channel_count for waveform in waveforms):
        raise ValueError("All generated segments must have the same channel count.")

    chunks: list[Any] = []
    silence_samples = 0

    def append_silence(milliseconds: int) -> None:
        nonlocal silence_samples
        samples = int(round(sample_rate * max(0, milliseconds) / 1000.0))
        if samples <= 0:
            return
        chunks.append(torch.zeros((channel_count, samples), dtype=torch.float32))
        silence_samples += samples

    append_silence(plan.leading_silence_ms)
    segment_samples: list[int] = []
    for segment, waveform in zip(plan.segments, waveforms, strict=True):
        append_silence(segment.silence_before_ms)
        chunks.append(waveform)
        segment_samples.append(int(waveform.shape[-1]))
    append_silence(plan.trailing_silence_ms)

    combined = torch.cat(chunks, dim=-1)
    if not torch.isfinite(combined).all():
        raise ValueError("Generated long-text audio contains NaN or Inf values.")
    return AudioAssembly(
        waveform=combined,
        sample_rate=int(sample_rate),
        segment_samples=tuple(segment_samples),
        silence_samples=silence_samples,
    )


def speech_plan_dict(plan: SpeechPlan) -> dict[str, Any]:
    return {
        "leading_silence_ms": plan.leading_silence_ms,
        "trailing_silence_ms": plan.trailing_silence_ms,
        "total_silence_ms": plan.total_silence_ms,
        "segments": [
            {
                "index": segment.index,
                "text": segment.text,
                "token_count": segment.token_count,
                "silence_before_ms": segment.silence_before_ms,
            }
            for segment in plan.segments
        ],
    }


__all__ = [
    "AudioAssembly",
    "assemble_speech_audio",
    "speech_plan_dict",
    "tokenizer_counter",
]
