from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_long_text_inference_reuses_runtime_and_releases_after_all_segments(
    monkeypatch,
) -> None:
    from runtime import inference_adapter as adapter
    from runtime.types import ModelHandle, SamplingConfig

    from dots_tts_t8.runtime_manager import RuntimeSpec

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)

    class Manager:
        def __init__(self):
            self.load_calls = 0
            self.generate_calls = []
            self.release_calls = 0
            self.runtime = SimpleNamespace(model=SimpleNamespace(tokenizer=Tokenizer()))

        def load(self, spec):
            self.load_calls += 1
            return self.runtime

        def generate(self, spec, *, seed, **kwargs):
            self.generate_calls.append((seed, kwargs["text"]))
            return {
                "audio": torch.ones((1, 100)),
                "sample_rate": 1000,
            }

        def release(self):
            self.release_calls += 1

    artifact = SimpleNamespace(
        repo_id="test/model",
        key="soar",
        sampling=SimpleNamespace(locked=False),
    )
    selected = SamplingConfig()
    spec = RuntimeSpec("test/model")
    manager = Manager()
    monkeypatch.setattr(adapter, "_MANAGER", manager)
    monkeypatch.setattr(
        adapter,
        "_resolve_inference",
        lambda *args: (artifact, selected, spec, None, "euler", 10, 1.2),
    )

    audio, info = adapter.run_long_text_inference(
        handle=ModelHandle(
            model_key="soar",
            model_dir=Path("."),
            precision="bfloat16",
            optimize=False,
            release_after_run=True,
        ),
        text="A。<pause=250ms>B。",
        language="none",
        seed=7,
        voice=None,
        sampling=None,
        max_segment_tokens=2,
        segment_gap_ms=200,
    )

    assert manager.load_calls == 1
    assert manager.generate_calls == [(7, "A。"), (8, "B。")]
    assert manager.release_calls == 1
    assert tuple(audio["waveform"].shape) == (1, 1, 450)
    assert info["segment_seeds"] == [7, 8]
    assert info["assembly"]["silence_samples"] == 250
