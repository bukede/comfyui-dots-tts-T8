from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _handle(*, release_after_run: bool = False):
    from runtime.types import ModelHandle

    return ModelHandle(
        model_key="soar",
        model_dir=Path("model"),
        precision="bfloat16",
        optimize=False,
        release_after_run=release_after_run,
    )


def test_inherit_without_voice_resolves_to_automatic_language(monkeypatch) -> None:
    from runtime import inference_adapter as adapter

    artifact = SimpleNamespace(
        key="soar",
        runtime="tts",
        sampling=SimpleNamespace(
            ode_method="euler",
            num_steps=10,
            guidance_scale=1.2,
            locked=False,
        ),
    )
    monkeypatch.setattr(
        adapter.ModelCatalog,
        "load",
        classmethod(lambda cls: SimpleNamespace(get=lambda key: artifact)),
    )

    resolved = adapter._resolve_inference(_handle(), "inherit", None, None)

    assert resolved[3] is None
    assert adapter._resolve_segment_language("inherit", None) is None


def test_single_generation_releases_model_after_failure(monkeypatch) -> None:
    from runtime import inference_adapter as adapter
    from runtime.types import SamplingConfig

    class Manager:
        release_calls = 0

        def release(self):
            self.release_calls += 1

    manager = Manager()
    artifact = SimpleNamespace(
        repo_id="test/model",
        key="soar",
        sampling=SimpleNamespace(locked=False),
    )
    monkeypatch.setattr(adapter, "_MANAGER", manager)
    monkeypatch.setattr(
        adapter,
        "_resolve_inference",
        lambda *args: (
            artifact,
            SamplingConfig(),
            SimpleNamespace(),
            None,
            "euler",
            10,
            1.2,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_generate_once",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("generation failed")),
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        adapter.run_inference(
            handle=_handle(release_after_run=True),
            text="hello",
            language="inherit",
            seed=1,
            voice=None,
            sampling=None,
        )

    assert manager.release_calls == 1
