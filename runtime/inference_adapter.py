from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from dots_tts_t8.long_text import (
    assemble_speech_audio,
    speech_plan_dict,
    tokenizer_counter,
)
from dots_tts_t8.model_catalog import ModelCatalog
from dots_tts_t8.production import (
    GeneratedAudio,
    ProductionRunner,
    VoiceProfile,
    build_production_plan,
    run_batch_production,
)
from dots_tts_t8.runtime_manager import RuntimeManager, RuntimeSpec
from dots_tts_t8.text_planner import plan_speech

from .audio_adapter import (
    edit_source_audio_file,
    normalize_output_audio,
    prompt_audio_file,
)
from .types import ModelHandle, SamplingConfig, SpeakerVoiceMap, VoicePrompt

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
        runtime_kind=artifact.runtime,
        content_fingerprint=handle.content_fingerprint,
    )
    resolved_language = language
    if resolved_language == "inherit":
        resolved_language = voice.language if voice is not None else None
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
        try:
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
            return audio, info
        finally:
            if handle.release_after_run:
                _MANAGER.release()


def run_edit_inference(
    *,
    handle: ModelHandle,
    source_audio: dict,
    instruction: str,
    source_text: str,
    target_text: str,
    use_xvector: str,
    seed: int,
    speaker_scale: float,
    num_steps: int,
    guidance_scale: float,
    ode_method: str = "euler",
    max_generate_length: int = 500,
) -> tuple[dict, dict]:
    artifact = ModelCatalog.load().get(handle.model_key)
    if artifact.runtime != "edit":
        raise ValueError("语音编辑节点必须使用 dots.tts.edit 模型。")
    mode: bool | str
    if use_xvector == "on":
        mode = True
    elif use_xvector == "off":
        mode = False
    elif use_xvector == "auto":
        mode = "auto"
    else:
        raise ValueError("use_xvector 必须为 auto、on 或 off。")
    spec = RuntimeSpec(
        model_name_or_path=str(handle.model_dir),
        precision=handle.precision,
        optimize=handle.optimize,
        max_generate_length=int(max_generate_length),
        warmup_on_optimize=handle.optimize,
        runtime_kind="edit",
        content_fingerprint=handle.content_fingerprint,
    )
    with _LOCK, edit_source_audio_file(source_audio) as source_path:
        try:
            result = _MANAGER.generate_edit(
                spec,
                seed=int(seed),
                source_audio_path=str(source_path),
                instruction=instruction.strip(),
                source_text=source_text.strip() or None,
                target_text=target_text.strip() or None,
                use_xvector=mode,
                speaker_scale=float(speaker_scale),
                ode_method=ode_method,
                num_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
            )
            audio = normalize_output_audio(result["audio"], result["sample_rate"])
            from dots_tts.data.edit_instruction import resolve_edit_use_xvector

            effective_xvector = resolve_edit_use_xvector(mode, instruction.strip())
            info = {
                "model": artifact.repo_id,
                "model_key": artifact.key,
                "seed": int(seed),
                "sample_rate": int(result["sample_rate"]),
                "samples": int(audio["waveform"].shape[-1]),
                "instruction": instruction.strip(),
                "source_text": result.get("source_text"),
                "target_text": result.get("target_text"),
                "source_text_source": result.get("source_text_source"),
                "target_text_source": result.get("target_text_source"),
                "use_xvector_requested": use_xvector,
                "use_xvector_effective": effective_xvector,
                "speaker_scale": float(speaker_scale),
                "ode_method": ode_method,
                "num_steps": int(num_steps),
                "guidance_scale": float(guidance_scale),
                "max_generate_length": int(max_generate_length),
                "duration_seconds": result.get("duration_seconds"),
                "time_used": result.get("time_used"),
                "rtf": result.get("rtf"),
            }
            return audio, info
        finally:
            if handle.release_after_run:
                _MANAGER.release()


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


def _voice_fingerprint(voice: VoicePrompt | None) -> dict[str, Any] | None:
    if voice is None:
        return None
    waveform = voice.audio.get("waveform")
    sample_rate = int(voice.audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        raise ValueError("ComfyUI speaker audio is invalid.")
    data = waveform.detach().float().cpu().contiguous().numpy()
    return {
        "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "shape": list(data.shape),
        "sample_rate": sample_rate,
        "transcript": voice.transcript,
        "language": voice.language,
    }


def _project_audio(path: str | Path) -> dict:
    import soundfile as sf
    import torch

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(waveform.T.copy())
    return normalize_output_audio(tensor, int(sample_rate))


def _resolve_segment_language(language: str, voice: VoicePrompt | None) -> str | None:
    resolved: str | None = language
    if resolved == "inherit":
        resolved = voice.language if voice is not None else None
    if resolved == "none":
        return None
    return resolved


@contextmanager
def _production_generator(
    *,
    handle: ModelHandle,
    language: str,
    seed: int,
    speaker_map: SpeakerVoiceMap | None,
    sampling: SamplingConfig | None,
    max_segment_tokens: int,
    segment_gap_ms: int,
):
    voices = dict(speaker_map.voices) if speaker_map is not None else {}
    fallback = speaker_map.fallback if speaker_map is not None else None
    reference_voice = fallback or next(iter(voices.values()), None)
    (
        artifact,
        selected,
        spec,
        _resolved_language,
        ode_method,
        num_steps,
        guidance_scale,
    ) = _resolve_inference(handle, language, reference_voice, sampling)

    with _LOCK:
        runtime_holder: list[Any | None] = [None]
        with (
            ExitStack() as stack,
            tempfile.TemporaryDirectory(prefix="dots-tts-t8-project-") as temporary,
        ):
            prompt_paths: dict[int, str] = {}
            for voice in [*voices.values(), fallback]:
                if voice is not None and id(voice) not in prompt_paths:
                    path = stack.enter_context(prompt_audio_file(voice.audio))
                    prompt_paths[id(voice)] = str(path)
            temporary_root = Path(temporary)

            def generate(segment, _profile, segment_seed):
                runtime = runtime_holder[0]
                if runtime is None:
                    runtime = _MANAGER.load(spec)
                    runtime_holder[0] = runtime
                voice = voices.get(segment.speaker or "", fallback)
                speech_plan = plan_speech(
                    segment.text,
                    max_tokens=int(max_segment_tokens),
                    segment_gap_ms=int(segment_gap_ms),
                    count_tokens=tokenizer_counter(runtime.model.tokenizer),
                )
                if not speech_plan.segments:
                    raise ValueError("工程段落不包含可朗读内容。")
                results = []
                for subsegment in speech_plan.segments:
                    results.append(
                        _generate_once(
                            spec=spec,
                            text=subsegment.text,
                            language=_resolve_segment_language(language, voice),
                            seed=(int(segment_seed) + subsegment.index) & 0xFFFFFFFF,
                            prompt_audio_path=(
                                prompt_paths[id(voice)] if voice is not None else None
                            ),
                            prompt_text=(voice.transcript or None if voice else None),
                            selected=selected,
                            ode_method=ode_method,
                            num_steps=num_steps,
                            guidance_scale=guidance_scale,
                        )
                    )
                sample_rates = {int(result["sample_rate"]) for result in results}
                if len(sample_rates) != 1:
                    raise ValueError(f"工程子段采样率不一致：{sorted(sample_rates)}")
                sample_rate = sample_rates.pop()
                assembly = assemble_speech_audio(
                    speech_plan,
                    [result["audio"] for result in results],
                    sample_rate=sample_rate,
                )
                import soundfile as sf

                output = temporary_root / f"segment-{segment.index:06d}.wav"
                sf.write(output, assembly.waveform.squeeze(0).numpy(), sample_rate)
                return GeneratedAudio(
                    str(output),
                    {
                        "model": artifact.repo_id,
                        "mode": _mode(voice),
                        "speaker": segment.speaker,
                        "seed": int(segment_seed),
                        "plan": speech_plan_dict(speech_plan),
                        "assembly": assembly.as_dict(),
                    },
                )

            context = {
                "model": artifact.repo_id,
                "model_key": artifact.key,
                "sampling": {
                    "ode_method": ode_method,
                    "num_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "speaker_scale": selected.speaker_scale,
                    "normalize_text": selected.normalize_text,
                },
                "language": language,
                "base_seed": int(seed),
                "voices": {
                    name: _voice_fingerprint(voice) for name, voice in voices.items()
                },
                "fallback": _voice_fingerprint(fallback),
            }
            placeholder_profiles = {name: VoiceProfile(speaker=name) for name in voices}
            fallback_profile = VoiceProfile("__fallback__") if fallback else None
            try:
                yield generate, context, placeholder_profiles, fallback_profile
            finally:
                if handle.release_after_run:
                    _MANAGER.release()


def run_project_inference(
    *,
    handle: ModelHandle,
    source_name: str,
    source_text: str,
    project_kind: str,
    language: str,
    seed: int,
    speaker_map: SpeakerVoiceMap | None,
    sampling: SamplingConfig | None,
    pronunciation: str,
    max_segment_tokens: int,
    segment_gap_ms: int,
    max_retries: int,
    resume: bool,
    output_root: str | Path,
    subtitle_fit_mode: str = "shrink",
    audio_formats: str = "wav",
    target_lufs: float | None = None,
    video_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
) -> tuple[dict, dict]:
    plan = build_production_plan(
        source_name,
        source_text,
        kind=project_kind,  # type: ignore[arg-type]
        pronunciation=pronunciation,
    )
    with _production_generator(
        handle=handle,
        language=language,
        seed=seed,
        speaker_map=speaker_map,
        sampling=sampling,
        max_segment_tokens=max_segment_tokens,
        segment_gap_ms=segment_gap_ms,
    ) as (generator, context, profiles, fallback_profile):
        result = ProductionRunner(output_root).run(
            plan,
            generator,
            voice_profiles=profiles,
            fallback_voice=fallback_profile,
            seed=seed,
            max_retries=max_retries,
            resume=resume,
            segment_gap_ms=segment_gap_ms,
            fingerprint_context=context,
            subtitle_fit_mode=subtitle_fit_mode,  # type: ignore[arg-type]
            audio_formats=audio_formats,
            target_lufs=target_lufs,
            video_path=video_path,
            ffmpeg_path=ffmpeg_path,
        )
    timeline = json.loads(Path(result.timeline_path).read_text(encoding="utf-8"))
    return _project_audio(result.audio_path), {
        "project": result.as_dict(),
        "timeline": timeline,
    }


def run_batch_inference(
    *,
    handle: ModelHandle,
    input_paths: list[str],
    language: str,
    seed: int,
    speaker_map: SpeakerVoiceMap | None,
    sampling: SamplingConfig | None,
    pronunciation: str,
    max_segment_tokens: int,
    segment_gap_ms: int,
    max_retries: int,
    resume: bool,
    output_root: str | Path,
    retry_failed_only: bool = False,
    subtitle_fit_mode: str = "shrink",
    audio_formats: str = "wav",
    target_lufs: float | None = None,
    video_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
) -> tuple[dict, dict]:
    with _production_generator(
        handle=replace(handle, release_after_run=False),
        language=language,
        seed=seed,
        speaker_map=speaker_map,
        sampling=sampling,
        max_segment_tokens=max_segment_tokens,
        segment_gap_ms=segment_gap_ms,
    ) as (generator, context, profiles, fallback_profile):
        try:
            result = run_batch_production(
                input_paths,
                output_root,
                generator,
                pronunciation=pronunciation,
                voice_profiles=profiles,
                fallback_voice=fallback_profile,
                seed=seed,
                max_retries=max_retries,
                resume=resume,
                segment_gap_ms=segment_gap_ms,
                fingerprint_context=context,
                retry_failed_only=retry_failed_only,
                subtitle_fit_mode=subtitle_fit_mode,  # type: ignore[arg-type]
                audio_formats=audio_formats,
                target_lufs=target_lufs,
                video_path=video_path,
                ffmpeg_path=ffmpeg_path,
            )
        finally:
            if handle.release_after_run:
                _MANAGER.release()
    preview = result.projects[0].audio_path
    return _project_audio(preview), result.as_dict()


def release_model() -> dict:
    from .transcription_adapter import release_transcription

    before = _MANAGER.info()
    _MANAGER.release()
    asr_released = release_transcription()
    return {
        "released": bool(before["loaded"]) or asr_released,
        "tts_released": bool(before["loaded"]),
        "asr_released": asr_released,
        "before": before,
    }


def runtime_info() -> dict:
    return _MANAGER.info()
