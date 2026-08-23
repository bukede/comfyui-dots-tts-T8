from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

NODE_ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    pytest.importorskip("comfy_api")
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
        "T8_DotsTTS_VoicePrompt",
        "T8_DotsTTS_Sampling",
        "T8_DotsTTS_SpeakerMap",
        "T8_DotsTTS_Pronunciation",
        "T8_DotsTTS_Generate",
        "T8_DotsTTS_LongText",
        "T8_DotsTTS_Project",
        "T8_DotsTTS_BatchQueue",
        "T8_DotsTTS_Environment",
        "T8_DotsTTS_Release",
    ]


def test_generate_uses_standard_audio_output() -> None:
    module = _load_nodes_module()
    schema = module.T8DotsTTSGenerate.define_schema()

    assert schema.outputs[0].io_type == "AUDIO"
    assert module.T8DotsTTSGenerate.validate_inputs("hello") is True
    assert isinstance(module.T8DotsTTSGenerate.validate_inputs("   "), str)


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
    processed = module.T8DotsTTSPronunciation.execute(
        "2026 年", "2026 => 二零二六"
    )
    assert processed[0] == "二零二六 年"


def test_model_selector_keeps_both_types_without_local_weights() -> None:
    module = _load_nodes_module()
    options = module.model_options()

    assert len(options) == 2
    assert options[0].endswith("[soar]")
    assert options[1].endswith("[mf-2steps]")
