"""Shared integration helpers for the T8 Windows and ComfyUI distributions."""

from .long_text import (
    AudioAssembly,
    assemble_speech_audio,
    speech_plan_dict,
    tokenizer_counter,
)
from .media import (
    AudioFitResult,
    MediaExportResult,
    export_project_media,
    find_ffmpeg,
    fit_audio_to_duration,
    parse_audio_formats,
)
from .model_catalog import ModelArtifact, ModelCatalog, ValidationReport
from .production import (
    BatchControl,
    BatchProductionResult,
    GeneratedAudio,
    ProductionCancelled,
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
from .templates import (
    ProjectTemplate,
    VoiceLibrary,
    parse_project_template,
    parse_voice_library,
    save_project_template,
    save_voice_library,
)
from .text_planner import SpeechPlan, SpeechSegment, plan_speech

__all__ = [
    "AudioAssembly",
    "AudioFitResult",
    "BatchControl",
    "BatchProductionResult",
    "GeneratedAudio",
    "ModelArtifact",
    "ModelCatalog",
    "MediaExportResult",
    "ProductionPlan",
    "ProductionCancelled",
    "ProductionResult",
    "ProductionRunner",
    "ProjectTemplate",
    "ProjectSegment",
    "PronunciationReport",
    "PronunciationRule",
    "SpeechPlan",
    "SpeechSegment",
    "SubtitleCue",
    "ValidationReport",
    "VoiceProfile",
    "VoiceLibrary",
    "apply_pronunciation",
    "assemble_speech_audio",
    "build_production_plan",
    "discover_project_files",
    "export_project_media",
    "find_ffmpeg",
    "fit_audio_to_duration",
    "parse_plain_text",
    "parse_audio_formats",
    "parse_project_template",
    "parse_pronunciation_rules",
    "parse_subtitles",
    "parse_voice_profiles",
    "parse_voice_library",
    "plan_speech",
    "run_batch_production",
    "speech_plan_dict",
    "save_project_template",
    "save_voice_library",
    "tokenizer_counter",
]

__version__ = "0.4.0"
