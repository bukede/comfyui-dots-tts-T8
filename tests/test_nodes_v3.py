from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

NODE_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    try:
        import comfy_api  # noqa: F401
    except ImportError:
        if os.environ.get("T8_REQUIRE_COMFY_API") == "1":
            raise
        pytest.skip("ComfyUI V3 API is not available in this test environment.")
    package_name = "_dots_tts_t8_test"
    package = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    )
    package.__path__ = [str(NODE_ROOT)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.nodes_v3", NODE_ROOT / "nodes_v3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_extension_exposes_stable_node_ids() -> None:
    module = _load_nodes_module()
    extension = asyncio.run(module.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = [node.define_schema() for node in nodes]

    assert [schema.node_id for schema in schemas] == [
        "T8_DotsTTS_ModelLoader",
        "T8_DotsTTS_EditModelLoader",
        "T8_DotsTTS_VoicePrompt",
        "T8_DotsTTS_Sampling",
        "T8_DotsTTS_SpeakerMap",
        "T8_DotsTTS_Pronunciation",
        "T8_DotsTTS_Generate",
        "T8_DotsTTS_Edit",
        "T8_DotsTTS_LongText",
        "T8_DotsTTS_Project",
        "T8_DotsTTS_BatchQueue",
        "T8_DotsTTS_Transcribe",
        "T8_DotsTTS_Timeline",
        "T8_DotsTTS_Environment",
        "T8_DotsTTS_Diagnostics",
        "T8_DotsTTS_Release",
    ]


def test_generate_uses_standard_audio_output() -> None:
    module = _load_nodes_module()
    schema = module.T8DotsTTSGenerate.define_schema()

    assert schema.outputs[0].io_type == "AUDIO"
    assert module.T8DotsTTSGenerate.validate_inputs("hello") is True
    assert isinstance(module.T8DotsTTSGenerate.validate_inputs("   "), str)


def test_edit_nodes_use_dedicated_model_type_and_audio_output() -> None:
    module = _load_nodes_module()
    loader = module.T8DotsTTSEditModelLoader.define_schema()
    edit = module.T8DotsTTSEdit.define_schema()

    assert loader.outputs[0].io_type == "T8_DOTS_TTS_EDIT_MODEL"
    assert edit.inputs[0].io_type == "T8_DOTS_TTS_EDIT_MODEL"
    assert edit.inputs[1].io_type == "AUDIO"
    assert edit.outputs[0].io_type == "AUDIO"
    assert edit.outputs[1].id == "target_text"
    assert (
        module.T8DotsTTSEdit.validate_inputs("<del>旧</del>新", "auto", 1.5, 10, 1.2)
        is True
    )
    assert isinstance(
        module.T8DotsTTSEdit.validate_inputs("", "auto", 1.5, 10, 1.2), str
    )


def test_asr_and_timeline_are_executable_text_output_nodes() -> None:
    module = _load_nodes_module()
    asr_schema = module.T8DotsTTSTranscribe.define_schema()
    timeline_schema = module.T8DotsTTSTimeline.define_schema()

    assert asr_schema.is_output_node is True
    assert timeline_schema.is_output_node is True
    assert [output.io_type for output in asr_schema.outputs] == [
        "STRING",
        "STRING",
        "STRING",
    ]
    assert module.T8DotsTTSTranscribe.validate_inputs(0, 0.72, False) is True
    assert isinstance(module.T8DotsTTSTranscribe.validate_inputs(0, 0.72, True), str)


def test_release_node_is_an_output_sink() -> None:
    module = _load_nodes_module()

    assert module.T8DotsTTSRelease.define_schema().is_output_node is True


def test_long_text_uses_standard_audio_output_and_validates_controls() -> None:
    module = _load_nodes_module()
    schema = module.T8DotsTTSLongText.define_schema()

    assert schema.outputs[0].io_type == "AUDIO"
    assert module.T8DotsTTSLongText.validate_inputs("hello", 120, 200) is True
    assert isinstance(module.T8DotsTTSLongText.validate_inputs("   ", 120, 200), str)
    assert isinstance(module.T8DotsTTSLongText.validate_inputs("hello", 0, 200), str)
    assert isinstance(module.T8DotsTTSLongText.validate_inputs("hello", 120, -1), str)


def test_project_nodes_expose_audio_outputs_and_validate_inputs() -> None:
    module = _load_nodes_module()
    project_schema = module.T8DotsTTSProject.define_schema()
    batch_schema = module.T8DotsTTSBatchQueue.define_schema()

    assert project_schema.outputs[0].io_type == "AUDIO"
    assert batch_schema.outputs[0].io_type == "AUDIO"
    project_inputs = {item.id for item in project_schema.inputs}
    batch_inputs = {item.id for item in batch_schema.inputs}
    assert {
        "subtitle_fit_mode",
        "audio_formats",
        "target_lufs",
        "video_path",
        "ffmpeg_path",
    } <= project_inputs
    assert "retry_failed_only" in batch_inputs
    assert (
        module.T8DotsTTSProject.validate_inputs(
            "旁白：hello", "demo.txt", "text", "", 120, 200, 2
        )
        is True
    )
    assert isinstance(
        module.T8DotsTTSProject.validate_inputs(
            "", "demo.txt", "text", "", 120, 200, 2
        ),
        str,
    )
    assert module.T8DotsTTSBatchQueue.validate_inputs("D:/work", 120, 200, 2) is True


def test_asr_and_timeline_nodes_expose_v3_contracts() -> None:
    module = _load_nodes_module()
    asr_schema = module.T8DotsTTSTranscribe.define_schema()
    timeline_schema = module.T8DotsTTSTimeline.define_schema()
    assert asr_schema.inputs[0].io_type == "AUDIO"
    assert [item.id for item in asr_schema.outputs] == [
        "srt",
        "timeline_json",
        "asr_info",
    ]
    assert module.T8DotsTTSTranscribe.validate_inputs(0, 0.72, False) is True
    assert isinstance(module.T8DotsTTSTranscribe.validate_inputs(0, 0.72, True), str)
    normalized = module.T8DotsTTSTimeline.execute(
        "1\n00:00:00,000 --> 00:00:01,000\n[旁白] 你好。\n",
        "srt",
    )
    assert "[旁白] 你好。" in normalized[0]
    assert timeline_schema.outputs[0].id == "srt"


def test_speaker_map_and_pronunciation_helpers() -> None:
    module = _load_nodes_module()
    voice = module.VoicePrompt(
        audio={"waveform": object(), "sample_rate": 48000},
        transcript="",
        language="ZH",
    )
    output = module.T8DotsTTSSpeakerMap.execute('{"旁白":"A"}', voice_a=voice)
    speaker_map = output[0]
    assert speaker_map.voices["旁白"] is voice
    processed = module.T8DotsTTSPronunciation.execute("2026 年", "2026 => 二零二六")
    assert processed[0] == "二零二六 年"


def test_speaker_map_loads_unlimited_file_voice_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    module = _load_nodes_module()
    monkeypatch.setattr(module, "_comfy_directory", lambda _kind: None)
    voices = {}
    for index in range(6):
        audio_path = tmp_path / f"voice-{index}.wav"
        sf.write(audio_path, np.ones(100, dtype=np.float32) * 0.01, 1000)
        voices[f"角色{index}"] = {
            "audio": str(audio_path),
            "transcript": f"参考文本{index}",
            "language": "ZH",
        }
    output = module.T8DotsTTSSpeakerMap.execute(
        "{}", voice_library=json.dumps({"name": "六角色", "voices": voices})
    )
    speaker_map = output[0]
    info = json.loads(output[1])
    assert len(speaker_map.voices) == 6
    assert info["speaker_count"] == 6


def test_model_selector_keeps_both_types_without_local_weights() -> None:
    module = _load_nodes_module()
    options = module.model_options()

    assert len(options) == 2
    assert options[0].endswith("[soar]")
    assert options[1].endswith("[mf-2steps]")
    edit_options = module.model_options(runtime="edit")
    assert len(edit_options) == 1
    assert edit_options[0].endswith("[edit]")
