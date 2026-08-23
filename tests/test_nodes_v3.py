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
        "T8_DotsTTS_Generate",
        "T8_DotsTTS_Environment",
        "T8_DotsTTS_Release",
    ]


def test_generate_uses_standard_audio_output() -> None:
    module = _load_nodes_module()
    schema = module.T8DotsTTSGenerate.define_schema()

    assert schema.outputs[0].io_type == "AUDIO"
    assert module.T8DotsTTSGenerate.validate_inputs("hello") is True
    assert isinstance(module.T8DotsTTSGenerate.validate_inputs("   "), str)


def test_model_selector_keeps_both_types_without_local_weights() -> None:
    module = _load_nodes_module()
    options = module.model_options()

    assert len(options) == 2
    assert options[0].endswith("[soar]")
    assert options[1].endswith("[mf-2steps]")
