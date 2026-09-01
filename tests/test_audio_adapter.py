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


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_normalize_output_rejects_invalid_sample_rate(sample_rate: int) -> None:
    from runtime.audio_adapter import normalize_output_audio

    with pytest.raises(ValueError, match="sample_rate"):
        normalize_output_audio(torch.zeros(10), sample_rate)


@pytest.mark.parametrize(
    "waveform",
    [torch.empty(0), torch.tensor([float("nan")]), torch.tensor([float("inf")])],
)
def test_normalize_output_rejects_invalid_waveform(waveform) -> None:
    from runtime.audio_adapter import normalize_output_audio

    with pytest.raises(ValueError):
        normalize_output_audio(waveform, 48000)


def test_audio_file_rejects_multi_batch_input() -> None:
    from runtime.audio_adapter import audio_file

    audio = {"waveform": torch.zeros((2, 1, 10)), "sample_rate": 48000}
    with pytest.raises(ValueError, match="batch size"):
        with audio_file(audio):
            pass
