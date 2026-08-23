from __future__ import annotations

import threading

from dots_tts_t8.model_catalog import ModelCatalog
from dots_tts_t8.runtime_manager import RuntimeManager, RuntimeSpec

from .audio_adapter import normalize_output_audio, prompt_audio_file
from .types import ModelHandle, SamplingConfig, VoicePrompt


_MANAGER = RuntimeManager()
_LOCK = threading.RLock()


def run_inference(
    *,
    handle: ModelHandle,
    text: str,
    language: str,
    seed: int,
    voice: VoicePrompt | None,
    sampling: SamplingConfig | None,
) -> tuple[dict, dict]:
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

    with _LOCK:
        if voice is None:
            result = _MANAGER.generate(
                spec,
                seed=seed,
                text=text,
                language=resolved_language,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=selected.speaker_scale,
                normalize_text=selected.normalize_text,
            )
        else:
            with prompt_audio_file(voice.audio) as path:
                result = _MANAGER.generate(
                    spec,
                    seed=seed,
                    text=text,
                    prompt_audio_path=str(path),
                    prompt_text=voice.transcript or None,
                    language=resolved_language,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=selected.speaker_scale,
                    normalize_text=selected.normalize_text,
                )

        audio = normalize_output_audio(result["audio"], result["sample_rate"])
        info = {
            "model": artifact.repo_id,
            "model_key": artifact.key,
            "seed": int(seed),
            "language": resolved_language,
            "mode": (
                "text_only"
                if voice is None
                else "continuation_clone"
                if voice.transcript.strip()
                else "xvector_clone"
            ),
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


def release_model() -> dict:
    before = _MANAGER.info()
    _MANAGER.release()
    return {"released": bool(before["loaded"]), "before": before}


def runtime_info() -> dict:
    return _MANAGER.info()
