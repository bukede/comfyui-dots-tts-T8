from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

NODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NODE_ROOT.parent
INTENTIONAL_T8_DIVERGENCES = {Path("asr_sidecar.py")}


def _tree_digest(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(normalized).digest())
    return len(files), digest.hexdigest()


def test_vendored_integrity_matches_release_baseline() -> None:
    baseline = json.loads(
        (NODE_ROOT / "T8_UPSTREAM_BASELINE.json").read_text(encoding="utf-8")
    )
    expected = baseline["vendored_integrity"]
    dots_count, dots_digest = _tree_digest(NODE_ROOT / "dots_tts")
    helpers_count, helpers_digest = _tree_digest(NODE_ROOT / "dots_tts_t8")
    assert dots_count == expected["dots_tts_python_files"]
    assert dots_digest == expected["dots_tts_python_sha256"]
    assert helpers_count == expected["dots_tts_t8_python_files"]
    assert helpers_digest == expected["dots_tts_t8_python_sha256"]


def test_vendored_upstream_matches_pinned_source() -> None:
    baseline = json.loads(
        (NODE_ROOT / "T8_UPSTREAM_BASELINE.json").read_text(encoding="utf-8")
    )
    assert baseline["tag"] == "v0.3.1"
    assert baseline["commit"] == "09235d55005c61867d3b94aeb6e13eede919a0a3"

    source_root = PROJECT_ROOT / "src" / "dots_tts"
    vendored_root = NODE_ROOT / "dots_tts"
    if not source_root.exists():
        pytest.skip(
            "Shared integration source is not present in the standalone node repo."
        )
    source_files = sorted(
        path.relative_to(source_root) for path in source_root.rglob("*.py")
    )
    vendored_files = sorted(
        path.relative_to(vendored_root) for path in vendored_root.rglob("*.py")
    )
    assert vendored_files == source_files
    for relative_path in source_files:
        assert (vendored_root / relative_path).read_bytes() == (
            source_root / relative_path
        ).read_bytes(), relative_path


def test_vendored_t8_helpers_match_shared_source() -> None:
    source_root = PROJECT_ROOT / "src" / "dots_tts_t8"
    vendored_root = NODE_ROOT / "dots_tts_t8"
    if not source_root.exists():
        pytest.skip(
            "Shared integration source is not present in the standalone node repo."
        )
    for source in source_root.rglob("*.py"):
        relative = source.relative_to(source_root)
        if relative in INTENTIONAL_T8_DIVERGENCES:
            continue
        assert (vendored_root / relative).read_bytes() == source.read_bytes(), relative
