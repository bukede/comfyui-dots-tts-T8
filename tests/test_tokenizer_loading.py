from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from dots_tts.models.dots_tts import model as model_module


def test_tokenizer_loader_preserves_upstream_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    sentinel = object()

    def fake_from_pretrained(path: str, **kwargs):
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(
        model_module.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    loaded = model_module.DotsTtsModel._load_tokenizer(Path("model-root"))

    assert loaded is sentinel
    assert calls == [
        (
            "model-root",
            {"local_files_only": True, "fix_mistral_regex": False},
        )
    ]
