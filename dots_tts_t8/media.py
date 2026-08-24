"""Audio fitting, loudness normalization and media export helpers."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SubtitleFitMode = Literal["none", "shrink", "fit", "trim"]
SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac")


@dataclass(frozen=True)
class AudioFitResult:
    audio: Any
    mode: SubtitleFitMode
    original_samples: int
    output_samples: int
    target_samples: int
    time_stretch_rate: float
    trimmed_samples: int
    padded_samples: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("audio")
        return value


@dataclass(frozen=True)
class MediaExportResult:
    primary_audio_path: str
    export_paths: tuple[str, ...]
    normalized_audio_path: str | None
    video_path: str | None
    target_lufs: float | None
    loudness: dict[str, Any] | None
    ffmpeg_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_audio_formats(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("wav",)
    raw = re.split(r"[,;\s]+", value) if isinstance(value, str) else value
    formats: list[str] = []
    for item in raw:
        name = str(item).strip().lower().lstrip(".")
        if not name:
            continue
        if name not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio export format {name!r}; expected wav/mp3/flac."
            )
        if name not in formats:
            formats.append(name)
    return tuple(formats or ["wav"])


def fit_audio_to_duration(
    audio: Any,
    sample_rate: int,
    target_ms: int,
    *,
    mode: SubtitleFitMode = "none",
) -> AudioFitResult:
    """Fit mono audio to a subtitle slot while preserving pitch when stretching."""

    import numpy as np

    if mode not in {"none", "shrink", "fit", "trim"}:
        raise ValueError(f"Unsupported subtitle fit mode: {mode}")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive.")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    original_samples = int(waveform.shape[0])
    target_samples = max(1, int(round(int(target_ms) * int(sample_rate) / 1000.0)))
    if original_samples <= 0:
        raise ValueError("Cannot fit empty audio.")

    should_stretch = mode == "fit" or (
        mode == "shrink" and original_samples > target_samples
    )
    rate = 1.0
    trimmed = 0
    padded = 0
    output = waveform
    if mode == "trim" and original_samples > target_samples:
        output = waveform[:target_samples]
        trimmed = original_samples - target_samples
    elif should_stretch and original_samples != target_samples:
        rate = original_samples / target_samples
        import librosa

        output = librosa.effects.time_stretch(waveform, rate=rate).astype(
            np.float32, copy=False
        )
        if output.shape[0] > target_samples:
            trimmed = int(output.shape[0] - target_samples)
            output = output[:target_samples]
        elif output.shape[0] < target_samples:
            padded = int(target_samples - output.shape[0])
            output = np.pad(output, (0, padded))
    return AudioFitResult(
        audio=np.asarray(output, dtype=np.float32),
        mode=mode,
        original_samples=original_samples,
        output_samples=int(output.shape[0]),
        target_samples=target_samples,
        time_stretch_rate=float(rate),
        trimmed_samples=trimmed,
        padded_samples=padded,
    )


def find_ffmpeg(explicit: str | Path | None = None) -> Path:
    executable = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    module_path = Path(__file__).resolve()
    candidates.extend(
        [
            Path(sys.executable).resolve().parent.parent
            / "ffmpeg"
            / "bin"
            / executable,
            module_path.parents[2] / "ffmpeg" / "bin" / executable,
            module_path.parents[2]
            / "desktop"
            / ".vendor"
            / "ffmpeg"
            / "bin"
            / executable,
        ]
    )
    discovered = shutil.which(executable) or shutil.which("ffmpeg")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "FFmpeg was not found. Install FFmpeg, connect an explicit path, or use the "
        "Windows integration package that bundles it."
    )


def _run_ffmpeg(
    ffmpeg: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: int = 1800,
) -> subprocess.CompletedProcess[str]:
    command = [str(ffmpeg), "-hide_banner", "-nostdin", *map(str, arguments)]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"FFmpeg timed out after {timeout_seconds} seconds."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise RuntimeError(
            f"FFmpeg failed with exit code {result.returncode}: {detail}"
        )
    return result


def _loudnorm_filter(
    target_lufs: float, target_peak: float, loudness_range: float
) -> str:
    return f"loudnorm=I={target_lufs}:TP={target_peak}:LRA={loudness_range}"


def measure_loudness(
    source: str | Path,
    *,
    target_lufs: float,
    target_peak: float = -1.5,
    loudness_range: float = 11.0,
    ffmpeg_path: str | Path | None = None,
) -> dict[str, Any]:
    ffmpeg = find_ffmpeg(ffmpeg_path)
    result = _run_ffmpeg(
        ffmpeg,
        [
            "-i",
            str(Path(source).resolve()),
            "-af",
            _loudnorm_filter(target_lufs, target_peak, loudness_range)
            + ":print_format=json",
            "-f",
            "null",
            "NUL" if sys.platform == "win32" else "/dev/null",
        ],
    )
    matches = re.findall(r"\{\s*\"input_i\".*?\}", result.stderr, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("FFmpeg did not return a loudnorm measurement.")
    measured = json.loads(matches[-1])
    for key, value in list(measured.items()):
        try:
            numeric = float(value)
            measured[key] = numeric if math.isfinite(numeric) else value
        except (TypeError, ValueError):
            pass
    return measured


def normalize_loudness(
    source: str | Path,
    destination: str | Path,
    *,
    target_lufs: float,
    target_peak: float = -1.5,
    loudness_range: float = 11.0,
    ffmpeg_path: str | Path | None = None,
) -> dict[str, Any]:
    if not -36.0 <= float(target_lufs) <= -5.0:
        raise ValueError("target_lufs must be between -36 and -5.")
    ffmpeg = find_ffmpeg(ffmpeg_path)
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    source_sample_rate = int(sf.info(source_path).samplerate)
    measured = measure_loudness(
        source_path,
        target_lufs=target_lufs,
        target_peak=target_peak,
        loudness_range=loudness_range,
        ffmpeg_path=ffmpeg,
    )
    required = {
        "input_i": "measured_I",
        "input_tp": "measured_TP",
        "input_lra": "measured_LRA",
        "input_thresh": "measured_thresh",
        "target_offset": "offset",
    }
    measured_values = ":".join(
        f"{argument}={measured[key]}" for key, argument in required.items()
    )
    audio_filter = (
        _loudnorm_filter(target_lufs, target_peak, loudness_range)
        + f":{measured_values}:linear=true:print_format=summary"
    )
    _run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-i",
            str(source_path),
            "-af",
            audio_filter,
            "-c:a",
            "pcm_s24le",
            "-ar",
            str(source_sample_rate),
            str(destination_path),
        ],
    )
    return measured


def export_project_media(
    source_wav: str | Path,
    *,
    audio_formats: str | Sequence[str] | None = None,
    target_lufs: float | None = None,
    video_path: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
) -> MediaExportResult:
    source = Path(source_wav).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Master audio does not exist: {source}")
    formats = parse_audio_formats(audio_formats)
    needs_ffmpeg = (
        target_lufs is not None
        or any(item in {"mp3", "flac"} for item in formats)
        or bool(video_path)
    )
    ffmpeg = find_ffmpeg(ffmpeg_path) if needs_ffmpeg else None
    primary = source
    normalized: Path | None = None
    measured: dict[str, Any] | None = None
    exports: list[Path] = [source]
    if target_lufs is not None:
        normalized = source.with_name(f"{source.stem}-lufs.wav")
        measured = normalize_loudness(
            source,
            normalized,
            target_lufs=float(target_lufs),
            ffmpeg_path=ffmpeg,
        )
        primary = normalized
        exports.append(normalized)
    if ffmpeg is not None:
        for audio_format in formats:
            if audio_format == "wav":
                continue
            destination = source.with_suffix(f".{audio_format}")
            codec = (
                ["-c:a", "libmp3lame", "-q:a", "2"]
                if audio_format == "mp3"
                else ["-c:a", "flac", "-compression_level", "8"]
            )
            _run_ffmpeg(
                ffmpeg,
                ["-y", "-i", str(primary), *codec, str(destination)],
            )
            exports.append(destination)
    dubbed_video: Path | None = None
    if video_path:
        source_video = Path(video_path).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(f"Video input does not exist: {source_video}")
        suffix = source_video.suffix.lower() or ".mp4"
        dubbed_video = source.parent / f"{source_video.stem}-dubbed{suffix}"
        _run_ffmpeg(
            ffmpeg or find_ffmpeg(ffmpeg_path),
            [
                "-y",
                "-i",
                str(source_video),
                "-i",
                str(primary),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "0:s?",
                "-map_metadata",
                "0",
                "-c:v",
                "copy",
                "-c:s",
                "copy",
                "-af",
                "apad",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(dubbed_video),
            ],
        )
        exports.append(dubbed_video)
    unique_exports = tuple(dict.fromkeys(str(path) for path in exports))
    return MediaExportResult(
        primary_audio_path=str(primary),
        export_paths=unique_exports,
        normalized_audio_path=str(normalized) if normalized else None,
        video_path=str(dubbed_video) if dubbed_video else None,
        target_lufs=float(target_lufs) if target_lufs is not None else None,
        loudness=measured,
        ffmpeg_path=str(ffmpeg) if ffmpeg else None,
    )


__all__ = [
    "AudioFitResult",
    "MediaExportResult",
    "SUPPORTED_AUDIO_FORMATS",
    "SubtitleFitMode",
    "export_project_media",
    "find_ffmpeg",
    "fit_audio_to_duration",
    "measure_loudness",
    "normalize_loudness",
    "parse_audio_formats",
]
