"""Shared integration helpers for the T8 Windows and ComfyUI distributions."""

from .model_catalog import ModelArtifact, ModelCatalog, ValidationReport
from .text_planner import SpeechPlan, SpeechSegment, plan_speech

__all__ = [
    "ModelArtifact",
    "ModelCatalog",
    "SpeechPlan",
    "SpeechSegment",
    "ValidationReport",
    "plan_speech",
]

__version__ = "0.1.0"
