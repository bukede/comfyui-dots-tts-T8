from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

from dots_tts_t8.long_text import (
    assemble_speech_audio,
    speech_plan_dict,
    tokenizer_counter,
)
from dots_tts_t8.model_catalog import ModelCatalog
from dots_tts_t8.runtime_manager import RuntimeManager, RuntimeSpec
from dots_tts_t8.text_planner import plan_speech

from .audio_adapter import normalize_output_audio, prompt_audio_file
from .types import ModelHandle, SamplingConfig, VoicePrompt

_MANAGER = RuntimeManager()
_LOCK = threading.RLock()


def _resolve_inference(
    handle: ModelHandle,
    language: str,
    voice: VoicePrompt | None,
    sampling: SamplingConfig | None,
) -> tuple[Any, SamplingConfig, RuntimeSpec, str | None, str, int, float]:
    artifact = ModelCatalog.load().get(handle.model_key)
    selected = sampling or SamplingConfig(
        ode_method=artifact.sampling.ode_method,
        num_steps=artifact.sampling.num_steps,
        guidance_scale=artifact.sampling.guidance_scale,
    )
    if artifact.sampling.locked:
        ode_method = artifact.sampling.ode_method
        num_steps = artifact.sampling.num_steps
        guidance_scale = artifact.sampling.guidance_scale
    else:
        ode_method = selected.ode_method
        num_steps = selected.num_steps
        guidance_scale = selected.guidance_scale

    spec = RuntimeSpec(
        model_name_or_path=str(handle.model_dir),
        precision=handle.precision,
        optimize=handle.optimize,
        max_generate_length=selected.max_generate_length,
        warmup_on_optimize=handle.optimize,
    )
    resolved_language = language
    if resolved_language == "inherit" and voice is not None:
        resolved_language = voice.language
    if resolved_language == "none":
        resolved_language = None

    return (
        artifact,
        selected,
        spec,
        resolved_language,
        ode_method,
        num_steps,
        guidance_scale,
    )


def _generate_once(
    *,
    spec: RuntimeSpec,
    text: str,
    language: str | None,
    seed: int,
    prompt_audio_path: str | None,
    prompt_text: str | None,
    selected: SamplingConfig,
    ode_method: str,
    num_steps: int,
    guidance_scale: float,
) -> dict:
    kwargs: dict[str, Any] = {
        "text": text,
        "language": language,
        "ode_method": ode_method,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "speaker_scale": selected.speaker_scale,
        "normalize_text": selected.normalize_text,
    }
    if prompt_audio_path is not None:
        kwargs["prompt_audio_path"] = prompt_audio_path
        kwargs["prompt_text"] = prompt_text
    return _MANAGER.generate(spec, seed=seed, **kwargs)


def _mode(voice: VoicePrompt | None) -> str:
    if voice is None:
        return "text_only"
    return "continuation_clone" if voice.transcript.strip() else "xvector_clone"


def run_inference(
    *,
    handle: ModelHandle,
    text: str,
    language: str,
    seed: int,
    voice: VoicePrompt | None,
    sampling: SamplingConfig | None,
) -> tuple[dict, dict]:
    (
        artifact,
        selected,
        spec,
        resolved_language,
        ode_method,
        num_steps,
        guidance_scale,
    ) = _resolve_inference(handle, language, voice, sampling)

    with _LOCK:
        prompt_context = (
            nullcontext(None) if voice is None else prompt_audio_file(voice.audio)
        )
        with prompt_context as path:
            result = _generate_once(
                spec=spec,
                text=text,
                language=resolved_language,
                seed=seed,
                prompt_audio_path=str(path) if path is not None else None,
                prompt_text=voice.transcript or None if voice is not None else None,
                selected=selected,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )

        audio = normalize_output_audio(result["audio"], result["sample_rate"])
        info = {
            "model": artifact.repo_id,
            "model_key": artifact.key,
            "seed": int(seed),
            "language": resolved_language,
            "mode": _mode(voice),
            "sample_rate": int(result["sample_rate"]),
            "samples": int(audio["waveform"].shape[-1]),
            "sampling_locked": artifact.sampling.locked,
            "ode_method": ode_method,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
        }
        if handle.release_after_run:
            _MANAGER.release()
        return audio, info


def run_long_text_inference(
    *,
    handle: ModelHandle,
    text: str,
    language: str,
    seed: int,
    voice: VoicePrompt | None,
    sampling: SamplingConfig | None,
    max_segment_tokens: int,
    segment_gap_ms: int,
) -> tuple[dict, dict]:
    (
        artifact,
        selected,
        spec,
        resolved_language,
        ode_method,
        num_steps,
        guidance_scale,
    ) = _resolve_inference(handle, language, voice, sampling)

    with _LOCK:
        runtime = _MANAGER.load(spec)
        try:
            plan = plan_speech(
                text,
                max_tokens=int(max_segment_tokens),
                segment_gap_ms=int(segment_gap_ms),
                count_tokens=tokenizer_counter(runtime.model.tokenizer),
            )
            if not plan.segments:
                raise ValueError("待合成文本不包含可朗读内容。")

            prompt_context = (
                nullcontext(None) if voice is None else prompt_audio_file(voice.audio)
            )
            results: list[dict] = []
            with prompt_context as path:
                for segment in plan.segments:
                    segment_seed = (int(seed) + segment.index) & 0xFFFFFFFF
                    results.append(
                        _generate_once(
                            spec=spec,
                            text=segment.text,
                            language=resolved_language,
                            seed=segment_seed,
                            prompt_audio_path=(str(path) if path is not None else None),
                            prompt_text=(
                                voice.transcript or None if voice is not None else None
                            ),
                            selected=selected,
                            ode_method=ode_method,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                        )
                    )

            sample_rates = {int(result["sample_rate"]) for result in results}
            if len(sample_rates) != 1:
                raise ValueError(f"长文本分段采样率不一致：{sorted(sample_rates)}")
            sample_rate = sample_rates.pop()
            assembly = assemble_speech_audio(
                plan,
                [result["audio"] for result in results],
                sample_rate=sample_rate,
            )
            audio = normalize_output_audio(assembly.waveform, sample_rate)
            info = {
                "model": artifact.repo_id,
                "model_key": artifact.key,
                "seed": int(seed),
                "segment_seeds": [
                    (int(seed) + segment.index) & 0xFFFFFFFF
                    for segment in plan.segments
                ],
                "language": resolved_language,
                "mode": _mode(voice),
                "sample_rate": sample_rate,
                "samples": assembly.total_samples,
                "sampling_locked": artifact.sampling.locked,
                "ode_method": ode_method,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "long_text": True,
                "plan": speech_plan_dict(plan),
                "assembly": assembly.as_dict(),
            }
            return audio, info
        finally:
            if handle.release_after_run:
                _MANAGER.release()


def release_model() -> dict:
    before = _MANAGER.info()
    _MANAGER.release()
    return {"released": bool(before["loaded"]), "before": before}


def runtime_info() -> dict:
    return _MANAGER.info()
