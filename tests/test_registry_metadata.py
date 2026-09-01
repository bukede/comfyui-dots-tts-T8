from __future__ import annotations

import ast
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NODE_IDS = {
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
}
PROTECTED_DEPENDENCIES = {
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "pydantic",
    "numpy",
}
PUBLISHED_PYTHON_PATHS = (
    ROOT / "__init__.py",
    ROOT / "nodes_v3.py",
    ROOT / "runtime",
    ROOT / "dots_tts",
    ROOT / "dots_tts_t8",
    ROOT / "scripts",
)


def _production_python_files():
    for path in PUBLISHED_PYTHON_PATHS:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


def _dependency_name(requirement: str) -> str:
    return (
        requirement.split("[", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .split("=", 1)[0]
        .lower()
    )


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
    assert not (
        {_dependency_name(item) for item in project["dependencies"]}
        & PROTECTED_DEPENDENCIES
    )


def test_release_version_is_consistent_across_metadata_and_workflows() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    assert f'__version__ = "{version}"' in (ROOT / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert f'__version__ = "{version}"' in (
        ROOT / "dots_tts_t8" / "__init__.py"
    ).read_text(encoding="utf-8")

    for workflow_path in (ROOT / "example_workflows" / "ui").glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        for node in workflow["nodes"]:
            properties = node.get("properties", {})
            if properties.get("cnr_id") == "dots-tts-t8":
                assert properties.get("ver") == version, workflow_path.name


def test_manager_node_list_covers_every_v3_node() -> None:
    node_list = json.loads((ROOT / "node_list.json").read_text(encoding="utf-8"))
    assert set(node_list) == EXPECTED_NODE_IDS
    assert all(
        isinstance(description, str) and description
        for description in node_list.values()
    )


def test_registry_package_has_no_installer_script() -> None:
    assert not (ROOT / "install.py").exists()


def test_registry_package_does_not_read_process_environment() -> None:
    findings: list[str] = []
    for path in _production_python_files():
        source = path.read_text(encoding="utf-8")
        for forbidden in ("os.environ", "os.getenv"):
            if forbidden in source:
                findings.append(f"{path.relative_to(ROOT)}: {forbidden}")
    assert findings == []


def test_registry_package_has_no_direct_process_or_dynamic_code_execution() -> None:
    forbidden_names = {"eval", "exec"}
    forbidden_attributes = {
        "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
        "os": {"popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "system"},
        "subprocess": {
            "Popen",
            "call",
            "check_call",
            "check_output",
            "run",
        },
    }
    findings: list[str] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name) and function.id in forbidden_names:
                findings.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: {function.id}"
                )
            elif (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.attr
                in forbidden_attributes.get(function.value.id, set())
            ):
                findings.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                    f"{function.value.id}.{function.attr}"
                )
    assert findings == []


def test_critical_vendored_inference_modules_are_present() -> None:
    critical = (
        "dots_tts/models/dots_tts/model.py",
        "dots_tts/models/dots_tts/core.py",
        "dots_tts/models/dots_tts/config.py",
        "dots_tts/modules/vocoder/vocoder_inference.py",
    )
    assert all((ROOT / relative).is_file() for relative in critical)
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/models/" in ignore_lines
    assert "models/" not in ignore_lines
