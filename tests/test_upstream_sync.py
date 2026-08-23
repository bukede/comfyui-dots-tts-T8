from __future__ import annotations

import json
from pathlib import Path

import pytest

NODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = NODE_ROOT.parent


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
        assert (vendored_root / relative).read_bytes() == source.read_bytes(), relative
