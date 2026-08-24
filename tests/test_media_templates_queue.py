from __future__ import annotations

# ruff: noqa: E402,I001 -- optional audio dependencies skip this module in lean CI.

import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from dots_tts_t8.media import fit_audio_to_duration, parse_audio_formats
from dots_tts_t8.production import BatchControl, ProductionCancelled
from dots_tts_t8.templates import (
    parse_project_template,
    parse_voice_library,
    save_project_template,
    save_voice_library,
)


def test_subtitle_fit_and_export_format_validation() -> None:
    audio = np.ones(2000, dtype=np.float32) * 0.1
    fitted = fit_audio_to_duration(audio, 1000, 1000, mode="shrink")
    assert fitted.output_samples == 1000
    assert fitted.time_stretch_rate == pytest.approx(2.0)
    assert parse_audio_formats("wav, mp3 flac") == ("wav", "mp3", "flac")
    with pytest.raises(ValueError):
        parse_audio_formats("ogg")


def test_voice_library_and_template_roundtrip(tmp_path: Path) -> None:
    voices = {"旁白": {"audio": "narrator.wav", "transcript": "参考文本"}}
    library_path = save_voice_library(tmp_path, "角色库", voices)
    assert parse_voice_library(library_path).voices == voices
    template_path = save_project_template(
        tmp_path,
        "模板",
        voices=library_path,
        settings={"subtitle_fit_mode": "shrink"},
    )
    assert (
        parse_project_template(template_path).settings["subtitle_fit_mode"] == "shrink"
    )


def test_batch_control_is_thread_safe() -> None:
    control = BatchControl()
    control.pause()
    completed = threading.Event()
    worker = threading.Thread(target=lambda: (control.checkpoint(), completed.set()))
    worker.start()
    time.sleep(0.05)
    assert not completed.is_set()
    control.resume()
    worker.join(timeout=2)
    assert completed.is_set()
    control.cancel()
    with pytest.raises(ProductionCancelled):
        control.checkpoint()
