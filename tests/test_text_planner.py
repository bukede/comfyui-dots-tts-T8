from __future__ import annotations

from dots_tts_t8.long_text import speech_plan_dict, tokenizer_counter
from dots_tts_t8.text_planner import plan_speech


def test_long_text_plan_uses_model_tokenizer_and_explicit_pause() -> None:
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(text)

    plan = plan_speech(
        "第一段。<pause=350ms>第二段。",
        max_tokens=8,
        segment_gap_ms=200,
        count_tokens=tokenizer_counter(Tokenizer()),
    )
    metadata = speech_plan_dict(plan)

    assert [segment["text"] for segment in metadata["segments"]] == [
        "第一段。",
        "第二段。",
    ]
    assert metadata["segments"][1]["silence_before_ms"] == 350
    assert metadata["total_silence_ms"] == 350


def test_pause_accumulation_is_bounded_to_sixty_seconds() -> None:
    plan = plan_speech("A。<pause=45s><pause=999999999999999999999999s>B。")

    assert plan.segments[1].silence_before_ms == 60_000
