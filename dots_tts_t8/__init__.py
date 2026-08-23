"""Shared integration helpers for the T8 Windows and ComfyUI distributions."""

from .long_text import (
    AudioAssembly,
    assemble_speech_audio,
    speech_plan_dict,
    tokenizer_counter,
)
from .model_catalog import ModelArtifact, ModelCatalog, ValidationReport
from .text_planner import SpeechPlan, SpeechSegment, plan_speech

__all__ = [
    "AudioAssembly",
    "ModelArtifact",
    "ModelCatalog",
    "SpeechPlan",
    "SpeechSegment",
    "ValidationReport",
    "assemble_speech_audio",
    "plan_speech",
    "speech_plan_dict",
    "tokenizer_counter",
]

__version__ = "0.2.0"
