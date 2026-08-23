from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_project_inference_maps_speakers_retries_and_resumes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from runtime import inference_adapter as adapter
    from runtime.types import ModelHandle, SamplingConfig, SpeakerVoiceMap, VoicePrompt

    from dots_tts_t8.runtime_manager import RuntimeSpec

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)

    class Manager:
        def __init__(self):
            self.runtime = SimpleNamespace(model=SimpleNamespace(tokenizer=Tokenizer()))
            self.generate_calls = []
            self.release_calls = 0

        def load(self, spec):
            return self.runtime

        def generate(self, spec, *, seed, **kwargs):
            self.generate_calls.append((seed, kwargs["text"], kwargs["prompt_text"]))
            return {"audio": torch.ones((1, 100)) * 0.1, "sample_rate": 1000}

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
    voice_a = VoicePrompt(
        audio={"waveform": torch.ones((1, 1, 100)), "sample_rate": 1000},
        transcript="voice A",
        language="ZH",
    )
    voice_b = VoicePrompt(
        audio={"waveform": torch.zeros((1, 1, 100)), "sample_rate": 1000},
        transcript="voice B",
        language="ZH",
    )
    speaker_map = SpeakerVoiceMap({"A": voice_a, "B": voice_b})
    handle = ModelHandle(
        model_key="soar",
        model_dir=tmp_path,
        precision="bfloat16",
        optimize=False,
        release_after_run=True,
    )

    audio, info = adapter.run_project_inference(
        handle=handle,
        source_name="scene.txt",
        source_text="A：第一句。\nB：第二句。",
        project_kind="text",
        language="inherit",
        seed=10,
        speaker_map=speaker_map,
        sampling=None,
        pronunciation="第一 => 第壹",
        max_segment_tokens=120,
        segment_gap_ms=200,
        max_retries=1,
        resume=True,
        output_root=tmp_path / "projects",
    )

    assert manager.generate_calls == [
        (10, "第壹句。", "voice A"),
        (11, "第二句。", "voice B"),
    ]
    assert tuple(audio["waveform"].shape) == (1, 1, 400)
    assert info["project"]["succeeded"] == 2
    assert manager.release_calls == 1

    _audio, resumed_info = adapter.run_project_inference(
        handle=handle,
        source_name="scene.txt",
        source_text="A：第一句。\nB：第二句。",
        project_kind="text",
        language="inherit",
        seed=10,
        speaker_map=speaker_map,
        sampling=None,
        pronunciation="第一 => 第壹",
        max_segment_tokens=120,
        segment_gap_ms=200,
        max_retries=1,
        resume=True,
        output_root=tmp_path / "projects",
    )
    assert len(manager.generate_calls) == 2
    assert resumed_info["project"]["resumed"] == 2
    assert manager.release_calls == 2
