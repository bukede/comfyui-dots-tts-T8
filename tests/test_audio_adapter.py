from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_normalize_output_uses_comfy_audio_shape() -> None:
    from runtime.audio_adapter import normalize_output_audio

    result = normalize_output_audio(torch.zeros(480), 48000)

    assert tuple(result["waveform"].shape) == (1, 1, 480)
    assert result["sample_rate"] == 48000


def test_prompt_audio_writes_mono_temp_file() -> None:
    from runtime.audio_adapter import prompt_audio_file

    audio = {"waveform": torch.zeros((1, 2, 480)), "sample_rate": 48000}
    with prompt_audio_file(audio) as path:
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 44
    assert not Path(path).exists()
