"""Editable subtitle timelines and lightweight waveform previews."""

from __future__ import annotations

import hashlib
import html
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .subtitles import SubtitleCue, format_srt, format_timestamp, parse_subtitles

TimelineFormat = Literal["srt", "vtt", "json"]
TIMELINE_HEADERS = ("序号", "开始(秒)", "结束(秒)", "说话人", "文本")


def cues_to_rows(cues: Iterable[SubtitleCue]) -> list[list[Any]]:
    return [
        [
            index + 1,
            round(cue.start_ms / 1000.0, 3),
            round(cue.end_ms / 1000.0, 3),
            cue.speaker or "",
            cue.text,
        ]
        for index, cue in enumerate(cues)
    ]


def _records(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            return list(value.to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(value, Mapping):
        if "data" in value:
            return list(value["data"])
        return [value]
    return list(value)


def _pick(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    normalized = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return default


def rows_to_cues(rows: Any) -> tuple[SubtitleCue, ...]:
    cues: list[SubtitleCue] = []
    for row_number, record in enumerate(_records(rows), start=1):
        if isinstance(record, Mapping):
            start = _pick(record, "开始(秒)", "start", "start_seconds", "start_ms")
            end = _pick(record, "结束(秒)", "end", "end_seconds", "end_ms")
            speaker = _pick(record, "说话人", "speaker", default="")
            text = _pick(record, "文本", "text", default="")
            cue_id = _pick(record, "序号", "index", "cue_id", default=row_number)
            milliseconds = "start_ms" in {str(key).strip().lower() for key in record}
        else:
            values = list(record)
            if len(values) < 5:
                values.extend([""] * (5 - len(values)))
            cue_id, start, end, speaker, text = values[:5]
            milliseconds = False
        body = str(text or "").strip()
        if not body:
            continue
        try:
            start_ms = int(round(float(start) * (1 if milliseconds else 1000)))
            end_ms = int(round(float(end) * (1 if milliseconds else 1000)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"时间轴第 {row_number} 行时间不是数字。") from exc
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"时间轴第 {row_number} 行必须满足 0 <= 开始 < 结束。")
        cues.append(
            SubtitleCue(
                index=len(cues),
                cue_id=str(cue_id or row_number),
                start_ms=start_ms,
                end_ms=end_ms,
                text=body,
                speaker=str(speaker or "").strip() or None,
            )
        )
    if not cues:
        raise ValueError("时间轴没有可用字幕。")
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    return tuple(
        SubtitleCue(
            index=index,
            cue_id=str(index + 1),
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text=cue.text,
            speaker=cue.speaker,
        )
        for index, cue in enumerate(ordered)
    )


def timeline_document(cues: Sequence[SubtitleCue], format_name: TimelineFormat) -> str:
    normalized = str(format_name).strip().lower()
    if normalized == "srt":
        return format_srt(tuple(cues))
    if normalized == "vtt":
        blocks = []
        for cue in cues:
            prefix = f"[{cue.speaker}] " if cue.speaker else ""
            blocks.append(
                f"{cue.cue_id}\n{format_timestamp(cue.start_ms, decimal='.')} --> "
                f"{format_timestamp(cue.end_ms, decimal='.')}\n{prefix}{cue.text}"
            )
        return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"
    if normalized == "json":
        return json.dumps(
            {
                "schema_version": 1,
                "kind": "dots-tts-timeline",
                "cues": [cue.as_dict() for cue in cues],
            },
            ensure_ascii=False,
            indent=2,
        )
    raise ValueError("时间轴格式必须是 srt、vtt 或 json。")


def load_timeline(path: str | Path) -> tuple[SubtitleCue, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"时间轴文件不存在：{source}")
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() != ".json":
        return parse_subtitles(text)
    value = json.loads(text)
    raw_cues = value.get("cues", value) if isinstance(value, Mapping) else value
    if not isinstance(raw_cues, list):
        raise ValueError("时间轴 JSON 必须包含 cues 数组。")
    rows = [
        {
            "cue_id": item.get("cue_id", index + 1),
            "start_ms": item.get("start_ms"),
            "end_ms": item.get("end_ms"),
            "speaker": item.get("speaker", ""),
            "text": item.get("text", ""),
        }
        for index, item in enumerate(raw_cues)
        if isinstance(item, Mapping)
    ]
    return rows_to_cues(rows)


def save_timeline(
    output_dir: str | Path,
    name: str,
    rows: Any,
    format_name: TimelineFormat = "srt",
) -> Path:
    safe_name = (
        "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in str(name).strip()
        ).strip("-.")[:80]
        or "timeline"
    )
    suffix = ".json" if format_name == "json" else f".{format_name}"
    destination = Path(output_dir).expanduser().resolve() / f"{safe_name}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        timeline_document(rows_to_cues(rows), format_name), encoding="utf-8"
    )
    os.replace(temporary, destination)
    return destination


def _speaker_color(speaker: str | None) -> str:
    digest = hashlib.sha256((speaker or "未标注").encode("utf-8")).digest()
    return f"hsl({int.from_bytes(digest[:2], 'big') % 360} 72% 58%)"


def render_timeline_html(
    audio_path: str | Path | None,
    rows: Any,
    *,
    width: int = 1200,
) -> str:
    cues = rows_to_cues(rows)
    duration = max(cue.end_ms for cue in cues) / 1000.0
    peaks: list[float] = []
    if audio_path:
        import numpy as np
        import soundfile as sf

        waveform, sample_rate = sf.read(
            Path(audio_path).expanduser().resolve(), dtype="float32", always_2d=True
        )
        mono = np.mean(waveform, axis=1)
        duration = max(duration, len(mono) / int(sample_rate))
        bins = min(int(width), max(1, len(mono)))
        edges = np.linspace(0, len(mono), bins + 1, dtype=int)
        peaks = [
            float(np.max(np.abs(mono[edges[i] : max(edges[i + 1], edges[i] + 1)])))
            for i in range(bins)
        ]
    duration = max(duration, 0.001)
    waveform_height = 108
    lane_height = 34
    svg_height = waveform_height + lane_height * len(cues) + 36
    parts = [
        '<div class="t8-timeline" style="overflow-x:auto;border:1px solid #cbd5e1;border-radius:10px;padding:8px">',
        f'<svg viewBox="0 0 {width} {svg_height}" width="100%" role="img" aria-label="字幕波形时间轴">',
        f'<rect x="0" y="0" width="{width}" height="{svg_height}" fill="#0f172a" rx="8"/>',
    ]
    if peaks:
        maximum = max(max(peaks), 1e-6)
        center = waveform_height / 2
        points_top = []
        points_bottom = []
        for index, peak in enumerate(peaks):
            x = index * width / max(1, len(peaks) - 1)
            amplitude = peak / maximum * (waveform_height * 0.42)
            points_top.append(f"{x:.1f},{center-amplitude:.1f}")
            points_bottom.append(f"{x:.1f},{center+amplitude:.1f}")
        polygon = " ".join(points_top + list(reversed(points_bottom)))
        parts.append(f'<polygon points="{polygon}" fill="#38bdf8" opacity="0.68"/>')
    for second in range(0, int(duration) + 1, max(1, int(duration / 12) or 1)):
        x = second / duration * width
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{waveform_height}" stroke="#334155"/>'
        )
        parts.append(
            f'<text x="{x+3:.1f}" y="14" fill="#cbd5e1" font-size="11">{second}s</text>'
        )
    for lane, cue in enumerate(cues):
        y = waveform_height + lane * lane_height + 6
        x = cue.start_ms / 1000.0 / duration * width
        rect_width = max(2.0, cue.duration_ms / 1000.0 / duration * width)
        label = f"{cue.speaker + ' · ' if cue.speaker else ''}{cue.text}"
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{rect_width:.1f}" height="26" rx="5" fill="{_speaker_color(cue.speaker)}" opacity="0.88"/>'
        )
        parts.append(
            f'<text x="{x+5:.1f}" y="{y+18}" fill="#fff" font-size="12">{html.escape(label[:80])}</text>'
        )
    parts.extend(["</svg>", "</div>"])
    return "".join(parts)


__all__ = [
    "TIMELINE_HEADERS",
    "TimelineFormat",
    "cues_to_rows",
    "load_timeline",
    "render_timeline_html",
    "rows_to_cues",
    "save_timeline",
    "timeline_document",
]
