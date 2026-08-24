"""SRT/WebVTT and speaker-aware plain-text project parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TIMESTAMP = re.compile(
    r"(?:(?P<hours>\d{1,3}):)?(?P<minutes>\d{2}):(?P<seconds>\d{2})"
    r"[,.](?P<milliseconds>\d{3})"
)
TIMELINE = re.compile(r"(?P<start>[^\s]+)\s*-->\s*(?P<end>[^\s]+)")
VOICE_TAG = re.compile(r"^<v(?:\s+([^>]+))?>(.*)$", re.IGNORECASE | re.DOTALL)
EXPLICIT_SPEAKER = re.compile(
    r"^\[speaker\s*=\s*([^\]]+)\]\s*(.*)$", re.IGNORECASE | re.DOTALL
)
BRACKET_SPEAKER = re.compile(r"^\[([^\]\r\n]{1,32})\]\s*(.*)$", re.DOTALL)
COLON_SPEAKER = re.compile(r"^([^\s:：\r\n][^:：\r\n]{0,31})[:：]\s*(.+)$", re.DOTALL)
SIMPLE_MARKUP = re.compile(r"</?(?:i|b|u|font)(?:\s+[^>]*)?>", re.IGNORECASE)


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    cue_id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cue_id": self.cue_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class PlainTextSegment:
    index: int
    text: str
    speaker: str | None = None


def parse_timestamp(value: str) -> int:
    match = TIMESTAMP.search(value.strip())
    if match is None:
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    parts = {name: int(match.group(name) or 0) for name in match.groupdict()}
    if parts["minutes"] >= 60 or parts["seconds"] >= 60:
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    return (
        (parts["hours"] * 60 + parts["minutes"]) * 60 + parts["seconds"]
    ) * 1000 + parts["milliseconds"]


def format_timestamp(milliseconds: int, *, decimal: str = ",") -> str:
    total = max(0, int(milliseconds))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{millis:03d}"


def extract_speaker(text: str) -> tuple[str | None, str]:
    cleaned = SIMPLE_MARKUP.sub("", text).strip()
    for pattern in (VOICE_TAG, EXPLICIT_SPEAKER, BRACKET_SPEAKER, COLON_SPEAKER):
        match = pattern.match(cleaned)
        if match is not None:
            speaker = (match.group(1) or "").strip() or None
            body = (match.group(2) or "").strip()
            return speaker, body
    return None, cleaned


def _subtitle_blocks(text: str) -> list[list[str]]:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("WEBVTT"):
        normalized = normalized.partition("\n")[2]
    return [
        [line for line in block.splitlines() if line.strip()]
        for block in re.split(r"\n\s*\n", normalized)
        if block.strip()
    ]


def parse_subtitles(text: str) -> tuple[SubtitleCue, ...]:
    cues: list[SubtitleCue] = []
    for block_number, lines in enumerate(_subtitle_blocks(text), start=1):
        timeline_index = next(
            (index for index, line in enumerate(lines) if "-->" in line), None
        )
        if timeline_index is None:
            # Ignore WEBVTT metadata such as STYLE/NOTE blocks.
            continue
        timeline = TIMELINE.search(lines[timeline_index])
        if timeline is None:
            raise ValueError(f"Invalid subtitle timeline in block {block_number}.")
        start_ms = parse_timestamp(timeline.group("start"))
        end_ms = parse_timestamp(timeline.group("end"))
        if end_ms <= start_ms:
            raise ValueError(f"Subtitle block {block_number} must end after it starts.")
        cue_id = (
            lines[timeline_index - 1].strip()
            if timeline_index > 0
            else str(block_number)
        )
        raw_text = "\n".join(lines[timeline_index + 1 :]).strip()
        speaker, body = extract_speaker(raw_text)
        if not body:
            continue
        cues.append(
            SubtitleCue(
                index=len(cues),
                cue_id=cue_id,
                start_ms=start_ms,
                end_ms=end_ms,
                text=body,
                speaker=speaker,
            )
        )
    if not cues:
        raise ValueError("Subtitle input contains no readable cues.")
    return tuple(cues)


def parse_plain_text(text: str) -> tuple[PlainTextSegment, ...]:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [
        block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()
    ]
    segments: list[PlainTextSegment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        annotated = [extract_speaker(line) for line in lines]
        if len(lines) > 1 and all(speaker is not None for speaker, _ in annotated):
            values = annotated
        else:
            values = [extract_speaker(" ".join(lines))]
        for speaker, body in values:
            if body:
                segments.append(PlainTextSegment(len(segments), body, speaker))
    if not segments:
        raise ValueError("Text input contains no readable content.")
    return tuple(segments)


def format_srt(cues: tuple[SubtitleCue, ...]) -> str:
    blocks = []
    for cue in cues:
        prefix = f"[{cue.speaker}] " if cue.speaker else ""
        blocks.append(
            f"{cue.cue_id}\n{format_timestamp(cue.start_ms)} --> "
            f"{format_timestamp(cue.end_ms)}\n{prefix}{cue.text}"
        )
    return "\n\n".join(blocks) + "\n"


__all__ = [
    "PlainTextSegment",
    "SubtitleCue",
    "extract_speaker",
    "format_srt",
    "format_timestamp",
    "parse_plain_text",
    "parse_subtitles",
    "parse_timestamp",
]
