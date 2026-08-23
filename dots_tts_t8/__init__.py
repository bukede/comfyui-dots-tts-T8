"""Shared integration helpers for the T8 Windows and ComfyUI distributions."""

from .long_text import (
    AudioAssembly,
    assemble_speech_audio,
    speech_plan_dict,
    tokenizer_counter,
)
from .model_catalog import ModelArtifact, ModelCatalog, ValidationReport
from .production import (
    BatchProductionResult,
    GeneratedAudio,
    ProductionPlan,
    ProductionResult,
    ProductionRunner,
    ProjectSegment,
    VoiceProfile,
    build_production_plan,
    discover_project_files,
    parse_voice_profiles,
    run_batch_production,
)
from .pronunciation import (
    PronunciationReport,
    PronunciationRule,
    apply_pronunciation,
    parse_pronunciation_rules,
)
from .subtitles import SubtitleCue, parse_plain_text, parse_subtitles
from .text_planner import SpeechPlan, SpeechSegment, plan_speech

__all__ = [
    "AudioAssembly",
    "BatchProductionResult",
    "GeneratedAudio",
    "ModelArtifact",
    "ModelCatalog",
    "ProductionPlan",
    "ProductionResult",
    "ProductionRunner",
    "ProjectSegment",
    "PronunciationReport",
    "PronunciationRule",
    "SpeechPlan",
    "SpeechSegment",
    "SubtitleCue",
    "ValidationReport",
    "VoiceProfile",
    "apply_pronunciation",
    "assemble_speech_audio",
    "build_production_plan",
    "discover_project_files",
    "parse_plain_text",
    "parse_pronunciation_rules",
    "parse_subtitles",
    "parse_voice_profiles",
    "plan_speech",
    "run_batch_production",
    "speech_plan_dict",
    "tokenizer_counter",
]

__version__ = "0.3.0"
