from __future__ import annotations

# ruff: noqa: E402,I001 -- optional audio dependencies skip this module in lean CI.

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from dots_tts_t8.production import (
    GeneratedAudio,
    ProductionRunner,
    VoiceProfile,
    build_production_plan,
    discover_project_files,
    parse_voice_profiles,
)
from dots_tts_t8.pronunciation import (
    apply_pronunciation,
    parse_pronunciation_rules,
)
from dots_tts_t8.subtitles import parse_plain_text, parse_subtitles


SRT = """1
00:00:00,000 --> 00:00:01,000
[旁白] 2026 年，T8 发布。

2
00:00:01,500 --> 00:00:02,500
小明：你好。
"""


def test_subtitle_and_plain_text_speaker_parsing() -> None:
    cues = parse_subtitles(SRT)
    assert [(cue.speaker, cue.text) for cue in cues] == [
        ("旁白", "2026 年，T8 发布。"),
        ("小明", "你好。"),
    ]
    assert cues[1].start_ms == 1500

    vtt = parse_subtitles("WEBVTT\n\n00:01.000 --> 00:02.500\n<v Alex>Hello")
    assert (vtt[0].start_ms, vtt[0].speaker, vtt[0].text) == (1000, "Alex", "Hello")

    segments = parse_plain_text("旁白：第一句。\n小明: Second line.")
    assert [(item.speaker, item.text) for item in segments] == [
        ("旁白", "第一句。"),
        ("小明", "Second line."),
    ]


def test_pronunciation_json_and_line_rules_are_deterministic() -> None:
    rules = parse_pronunciation_rules(
        '[{"match":"2026","replacement":"二零二六","priority":5},'
        '{"match":"T8","replacement":"T 八"}]'
    )
    report = apply_pronunciation("2026 T8 T8", rules)
    assert report.text == "二零二六 T 八 T 八"
    assert report.replacements == 3

    line_rules = parse_pronunciation_rules("张三 => 章三\n# comment\n100\t一百")
    assert apply_pronunciation("张三有100元", line_rules).text == "章三有一百元"


def test_voice_profile_shorthand_and_validation() -> None:
    profiles = parse_voice_profiles(
        '{"旁白":"narrator","小明":{"audio":"voice.wav",'
        '"transcript":"你好","language":"ZH"}}'
    )
    assert profiles["旁白"].preset == "narrator"
    assert profiles["小明"].prompt_audio_path == "voice.wav"
    with pytest.raises(ValueError, match="no audio path or preset"):
        parse_voice_profiles('{"坏配置":{"transcript":"text"}}')


def test_project_runner_retries_and_resumes_each_segment(tmp_path: Path) -> None:
    plan = build_production_plan(
        "demo.srt",
        SRT,
        pronunciation='{"2026":"二零二六","T8":"T 八"}',
    )
    assert plan.kind == "subtitle"
    assert plan.segments[0].text == "二零二六 年，T 八 发布。"

    calls: list[tuple[int, str | None, int]] = []
    attempts: dict[int, int] = {}
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def generate(segment, voice, seed):
        calls.append((segment.index, voice.speaker if voice else None, seed))
        attempts[segment.index] = attempts.get(segment.index, 0) + 1
        if segment.index == 0 and attempts[segment.index] == 1:
            raise RuntimeError("transient")
        output = scratch / f"{segment.index}-{attempts[segment.index]}.wav"
        sf.write(output, np.ones(500, dtype=np.float32) * 0.1, 1000)
        return GeneratedAudio(str(output), {"seed": seed})

    profiles = {
        "旁白": VoiceProfile("旁白", preset="narrator"),
        "小明": VoiceProfile("小明", preset="boy"),
    }
    runner = ProductionRunner(tmp_path / "projects")
    result = runner.run(
        plan,
        generate,
        voice_profiles=profiles,
        seed=7,
        max_retries=1,
        resume=True,
    )
    assert result.succeeded == 2
    assert result.failed == 0
    assert calls == [(0, "旁白", 7), (0, "旁白", 7), (1, "小明", 8)]
    audio, sample_rate = sf.read(result.audio_path)
    assert sample_rate == 1000
    assert audio.shape[0] == 2500
    checkpoint = json.loads(Path(result.checkpoint_path).read_text(encoding="utf-8"))
    assert checkpoint["segments"]["0"]["attempts"] == 2
    assert Path(result.subtitle_path).read_text(encoding="utf-8").startswith(
        "1\n00:00:00,000 --> 00:00:01,000"
    )

    def should_not_run(*args):
        raise AssertionError("resume should reuse successful segment WAVs")

    resumed = runner.run(
        plan,
        should_not_run,
        voice_profiles=profiles,
        seed=7,
        max_retries=1,
        resume=True,
    )
    assert resumed.resumed == 2


def test_discover_project_files_recurses_and_sorts(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("ignored", encoding="utf-8")
    assert [path.name for path in discover_project_files([tmp_path])] == [
        "a.txt",
        "b.srt",
    ]
