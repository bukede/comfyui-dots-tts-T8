from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NODE_IDS = {
    "T8_DotsTTS_ModelLoader",
    "T8_DotsTTS_VoicePrompt",
    "T8_DotsTTS_Sampling",
    "T8_DotsTTS_Generate",
    "T8_DotsTTS_Environment",
    "T8_DotsTTS_Release",
}
PROTECTED_DEPENDENCIES = {
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "pydantic",
    "numpy",
}


def _dependency_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()


def test_registry_metadata_is_publishable() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    comfy = metadata["tool"]["comfy"]
    assert project["name"] == "dots-tts-t8"
    assert project["version"].count(".") == 2
    assert project["urls"]["Repository"] == (
        "https://github.com/T8mars/comfyui-dots-tts-T8"
    )
    assert comfy["PublisherId"] == "t8star"
    assert not ({_dependency_name(item) for item in project["dependencies"]} & PROTECTED_DEPENDENCIES)


def test_manager_node_list_covers_every_v3_node() -> None:
    node_list = json.loads((ROOT / "node_list.json").read_text(encoding="utf-8"))
    assert set(node_list) == EXPECTED_NODE_IDS
    assert all(isinstance(description, str) and description for description in node_list.values())


def test_registry_package_has_no_installer_script() -> None:
    assert not (ROOT / "install.py").exists()
