from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelHandle:
    model_key: str
    model_dir: Path
    precision: str
    optimize: bool
    release_after_run: bool
    content_fingerprint: str = ""


@dataclass(frozen=True)
class VoicePrompt:
    audio: dict
    transcript: str
    language: str


@dataclass(frozen=True)
class SpeakerVoiceMap:
    voices: dict[str, VoicePrompt]
    fallback: VoicePrompt | None = None


@dataclass(frozen=True)
class SamplingConfig:
    ode_method: str = "euler"
    num_steps: int = 10
    guidance_scale: float = 1.2
    speaker_scale: float = 1.5
    normalize_text: bool = False
    max_generate_length: int = 500
