from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from dots_tts_t8.timeline import (  # noqa: E402
    cues_to_rows,
    load_timeline,
    render_timeline_html,
    rows_to_cues,
    save_timeline,
)
from dots_tts_t8.transcription import (  # noqa: E402
    QwenAsrEngine,
    TimedToken,
    TranscriptionResult,
    TranscriptionSegment,
    cluster_speakers,
    group_timed_tokens,
)


def test_editable_timeline_round_trip_and_waveform_preview(tmp_path: Path) -> None:
    rows = [
        [2, 2.0, 3.1, "角色B", "第二句，逗号保留。"],
        [1, 0.1, 1.4, "角色A", "第一句。"],
    ]
    cues = rows_to_cues(rows)
    assert [cue.speaker for cue in cues] == ["角色A", "角色B"]
    assert cues_to_rows(cues)[0][1:3] == [0.1, 1.4]

    sample_rate = 8_000
    audio = np.sin(2 * np.pi * 220 * np.arange(sample_rate * 4) / sample_rate)
    audio_path = tmp_path / "dialog.wav"
    sf.write(audio_path, audio.astype("float32") * 0.1, sample_rate)

    srt = save_timeline(tmp_path, "dialog", rows, "srt")
    vtt = save_timeline(tmp_path, "dialog", rows, "vtt")
    structured = save_timeline(tmp_path, "dialog", rows, "json")
    assert load_timeline(srt) == cues
    assert load_timeline(vtt) == cues
    assert load_timeline(structured) == cues
    assert "第二句，逗号保留。" in vtt.read_text(encoding="utf-8")
    preview = render_timeline_html(audio_path, rows)
    assert "<svg" in preview and "角色A" in preview and "第一句" in preview


def test_timestamp_grouping_and_deterministic_speaker_clustering() -> None:
    segments = group_timed_tokens(
        [
            TimedToken("Hello", 0, 400),
            TimedToken("world!", 450, 1_500),
            TimedToken("你好。", 2_400, 3_500),
        ]
    )
    assert [(item.text, item.start_ms, item.end_ms) for item in segments] == [
        ("Hello world!", 0, 1_500),
        ("你好。", 2_400, 3_500),
    ]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]],
        dtype=np.float32,
    )
    assert cluster_speakers(embeddings, speaker_count=2) == [0, 0, 1, 1]
    assert cluster_speakers(embeddings, similarity_threshold=0.9) == [0, 0, 1, 1]


def test_qwen_adapter_converts_forced_alignment_without_loading_real_model(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "speech.wav"
    sf.write(audio_path, np.zeros(16_000, dtype=np.float32), 16_000)
    fake_result = SimpleNamespace(
        text="Hello world!",
        language="English",
        time_stamps=[
            SimpleNamespace(text="Hello", start_time=0.0, end_time=0.4),
            SimpleNamespace(text="world!", start_time=0.45, end_time=0.9),
        ],
    )

    class FakeModel:
        @staticmethod
        def transcribe(**kwargs):
            assert kwargs["return_time_stamps"] is True
            return [fake_result]

    engine = QwenAsrEngine(
        "local-asr",
        model_revision=None,
        aligner_name_or_path="local-aligner",
        aligner_revision=None,
        device="cpu",
    )
    engine._model = FakeModel()  # noqa: SLF001
    result = engine.transcribe(audio_path)
    assert result.language == "English"
    assert result.segments[0].text == "Hello world!"
    assert "00:00:00,000 --> 00:00:00,900" in result.as_dict()["srt"]


def test_transcription_result_json_contract() -> None:
    result = TranscriptionResult(
        text="你好",
        language="Chinese",
        segments=(TranscriptionSegment(0, 0, 800, "你好", "说话人1"),),
        model="asr",
        model_revision="revision",
        aligner="aligner",
        aligner_revision="aligner-revision",
        diarized=True,
    )
    payload = json.loads(json.dumps(result.as_dict(), ensure_ascii=False))
    assert payload["segments"][0]["speaker"] == "说话人1"
    assert payload["segments"][0]["duration_ms"] == 800
