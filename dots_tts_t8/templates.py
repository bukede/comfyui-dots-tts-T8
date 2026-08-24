"""Portable voice libraries and dubbing project templates."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VoiceLibrary:
    name: str
    voices: dict[str, Any]
    schema_version: int = 1
    kind: str = "dots-tts-voice-library"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectTemplate:
    name: str
    voices: dict[str, Any]
    pronunciation: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    kind: str = "dots-tts-project-template"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_input(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Path):
        text = value.expanduser().read_text(encoding="utf-8-sig")
    else:
        text = str(value).strip()
        candidate = Path(text).expanduser()
        if not text.startswith(("{", "[")) and candidate.is_file():
            text = candidate.read_text(encoding="utf-8-sig")
    parsed = json.loads(text or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Template JSON must be an object.")
    return parsed


def parse_voice_library(
    value: str | Path | Mapping[str, Any],
) -> VoiceLibrary:
    parsed = _load_json_input(value)
    voices = parsed.get("voices", parsed)
    if not isinstance(voices, dict):
        raise ValueError("Voice library 'voices' must be an object.")
    cleaned: dict[str, Any] = {}
    for raw_name, profile in voices.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Voice library speaker names cannot be empty.")
        if not isinstance(profile, (str, dict)):
            raise ValueError(f"Invalid voice profile for speaker {name!r}.")
        cleaned[name] = profile
    return VoiceLibrary(
        name=str(parsed.get("name", "voice-library")).strip() or "voice-library",
        voices=cleaned,
        schema_version=int(parsed.get("schema_version", 1)),
    )


def parse_project_template(
    value: str | Path | Mapping[str, Any],
) -> ProjectTemplate:
    parsed = _load_json_input(value)
    voices = parsed.get("voices", parsed.get("voice_library", {}))
    if isinstance(voices, dict) and "voices" in voices:
        voices = voices["voices"]
    if not isinstance(voices, dict):
        raise ValueError("Project template voices must be an object.")
    settings = parsed.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Project template settings must be an object.")
    return ProjectTemplate(
        name=str(parsed.get("name", "project-template")).strip() or "project-template",
        voices=dict(voices),
        pronunciation=str(parsed.get("pronunciation", "")),
        settings=dict(settings),
        schema_version=int(parsed.get("schema_version", 1)),
    )


def _safe_name(value: str, fallback: str) -> str:
    result = re.sub(r"[^\w.-]+", "-", value, flags=re.UNICODE).strip("-.")
    return (result or fallback)[:80]


def _atomic_write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def save_voice_library(
    output_dir: str | Path,
    name: str,
    voices: str | Path | Mapping[str, Any],
) -> Path:
    library = parse_voice_library(voices)
    value = VoiceLibrary(name=name.strip() or library.name, voices=library.voices)
    destination = (
        Path(output_dir).expanduser().resolve()
        / f"{_safe_name(value.name, 'voice-library')}.voice-library.json"
    )
    return _atomic_write(destination, value.as_dict())


def save_project_template(
    output_dir: str | Path,
    name: str,
    *,
    voices: str | Path | Mapping[str, Any],
    pronunciation: str = "",
    settings: Mapping[str, Any] | None = None,
) -> Path:
    library = parse_voice_library(voices)
    template = ProjectTemplate(
        name=name.strip() or "project-template",
        voices=library.voices,
        pronunciation=pronunciation,
        settings=dict(settings or {}),
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        / f"{_safe_name(template.name, 'project-template')}.project-template.json"
    )
    return _atomic_write(destination, template.as_dict())


__all__ = [
    "ProjectTemplate",
    "VoiceLibrary",
    "parse_project_template",
    "parse_voice_library",
    "save_project_template",
    "save_voice_library",
]
