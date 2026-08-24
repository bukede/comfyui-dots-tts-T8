from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path


def normalize_output_audio(audio, sample_rate: int) -> dict:
    waveform = audio.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Unexpected dots.tts waveform shape: {tuple(waveform.shape)}")
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


@contextmanager
def audio_file(audio: dict, *, subtype: str = "PCM_16", prefix: str = "dots-tts-t8-"):
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        raise ValueError("ComfyUI AUDIO must contain waveform and sample_rate.")
    data = waveform.detach().float().cpu()
    if data.ndim == 3:
        data = data[0]
    if data.ndim == 2:
        data = data.mean(dim=0)
    if data.ndim != 1:
        raise ValueError(f"Unsupported ComfyUI AUDIO shape: {tuple(waveform.shape)}")

    import soundfile as sf

    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        path = Path(temp_dir) / "audio.wav"
        sf.write(path, data.numpy(), sample_rate, subtype=subtype)
        yield path


def prompt_audio_file(audio: dict):
    return audio_file(audio, subtype="PCM_16", prefix="dots-tts-t8-prompt-")


def edit_source_audio_file(audio: dict):
    return audio_file(audio, subtype="FLOAT", prefix="dots-tts-t8-edit-")
