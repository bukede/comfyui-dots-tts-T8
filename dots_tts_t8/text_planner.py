"""Long-text and explicit-pause planning shared by desktop and ComfyUI."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

PAUSE_PATTERN = re.compile(
    r"<pause\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)?>",
    flags=re.IGNORECASE,
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;：:\n])")
SOFT_BOUNDARY = re.compile(r"(?<=[，、,])")


@dataclass(frozen=True)
class SpeechSegment:
    index: int
    text: str
    silence_before_ms: int
    token_count: int


@dataclass(frozen=True)
class SpeechPlan:
    segments: tuple[SpeechSegment, ...]
    leading_silence_ms: int
    trailing_silence_ms: int

    @property
    def total_silence_ms(self) -> int:
        return (
            self.leading_silence_ms
            + self.trailing_silence_ms
            + sum(segment.silence_before_ms for segment in self.segments)
        )

    @property
    def total_explicit_silence_ms(self) -> int:
        """Backward-compatible alias retained for the v0.1 integration API."""

        return self.total_silence_ms


def _pause_ms(match: re.Match[str]) -> int:
    value = float(match.group("value"))
    unit = (match.group("unit") or "s").lower()
    millis = value if unit == "ms" else value * 1000.0
    if not math.isfinite(millis) or millis >= 60_000:
        return 60_000
    return max(0, min(int(round(millis)), 60_000))


def _split_to_limit(
    text: str,
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if count_tokens(stripped) <= max_tokens:
        return [stripped]

    for boundary in (SENTENCE_BOUNDARY, SOFT_BOUNDARY):
        parts = [part.strip() for part in boundary.split(stripped) if part.strip()]
        if len(parts) <= 1:
            continue
        result: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{part}" if current else part
            if current and count_tokens(candidate) > max_tokens:
                result.extend(
                    _split_to_limit(
                        current, max_tokens=max_tokens, count_tokens=count_tokens
                    )
                )
                current = part
            else:
                current = candidate
        if current:
            result.extend(
                _split_to_limit(
                    current, max_tokens=max_tokens, count_tokens=count_tokens
                )
            )
        return result

    words = stripped.split()
    if len(words) > 1:
        result: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and count_tokens(candidate) > max_tokens:
                result.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            result.append(" ".join(current))
        return result

    # CJK and other unspaced text fallback. Binary search keeps tokenizers with
    # non-linear token counts safe while avoiding one-character-only chunks.
    result: list[str] = []
    remaining = stripped
    while remaining:
        low, high = 1, len(remaining)
        best = 1
        while low <= high:
            mid = (low + high) // 2
            if count_tokens(remaining[:mid]) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        result.append(remaining[:best].strip())
        remaining = remaining[best:].strip()
    return [part for part in result if part]


def plan_speech(
    text: str,
    *,
    max_tokens: int = 120,
    segment_gap_ms: int = 200,
    count_tokens: Callable[[str], int] | None = None,
) -> SpeechPlan:
    """Create a deterministic plan without loading model weights.

    ``count_tokens`` should use the selected model tokenizer in production. The
    character-count fallback exists for UI previews before a model is loaded.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if segment_gap_ms < 0:
        raise ValueError("segment_gap_ms cannot be negative.")
    token_counter = count_tokens or (lambda value: len(value))

    chunks: list[tuple[str, int]] = []
    pending_silence_ms = 0
    cursor = 0
    for match in PAUSE_PATTERN.finditer(text):
        before = text[cursor : match.start()]
        if before.strip():
            chunks.append((before, pending_silence_ms))
            pending_silence_ms = 0
        pending_silence_ms = min(60_000, pending_silence_ms + _pause_ms(match))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        chunks.append((tail, pending_silence_ms))
        pending_silence_ms = 0

    if not chunks:
        return SpeechPlan((), pending_silence_ms, 0)

    segments: list[SpeechSegment] = []
    leading_silence_ms = chunks[0][1]
    for chunk_index, (chunk, explicit_silence_ms) in enumerate(chunks):
        parts = _split_to_limit(
            chunk, max_tokens=max_tokens, count_tokens=token_counter
        )
        for part_index, part in enumerate(parts):
            silence_before = 0
            if segments:
                silence_before = segment_gap_ms
            if part_index == 0 and chunk_index > 0:
                silence_before = explicit_silence_ms
            segments.append(
                SpeechSegment(
                    index=len(segments),
                    text=part,
                    silence_before_ms=silence_before,
                    token_count=token_counter(part),
                )
            )

    return SpeechPlan(tuple(segments), leading_silence_ms, pending_silence_ms)
