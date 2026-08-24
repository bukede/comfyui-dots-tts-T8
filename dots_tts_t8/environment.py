"""Non-mutating environment diagnostics for desktop and ComfyUI."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any

RUNTIME_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "numpy",
    "pydantic",
    "safetensors",
    "huggingface-hub",
    "soundfile",
    "librosa",
    "torchdiffeq",
    "einops",
    "loguru",
    "langcodes",
    "lingua-language-detector",
    "qwen-asr",
)

REQUIRED_RUNTIME_PACKAGES = (
    "torch",
    "torchaudio",
    "transformers",
    "numpy",
    "pydantic",
    "safetensors",
    "soundfile",
    "torchdiffeq",
    "einops",
    "loguru",
    "langcodes",
    "lingua-language-detector",
)


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    executable: str
    os: str
    machine: str
    packages: dict[str, str | None]
    torch_cuda_available: bool
    torch_cuda_version: str | None
    gpu_name: str | None
    gpu_count: int
    gpu_devices: tuple[dict[str, Any], ...]
    cuda_bf16_supported: bool
    cudnn_version: int | None
    vram_recommendation: dict[str, Any]
    supported_python: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def recommend_vram(total_bytes: int, free_bytes: int | None = None) -> dict[str, Any]:
    gib = 1024**3
    total_gib = round(max(0, int(total_bytes)) / gib, 2)
    free_gib = (
        round(max(0, int(free_bytes)) / gib, 2) if free_bytes is not None else None
    )
    if total_gib < 8:
        profile = "limited"
        model = "mf-2steps"
        max_segment_tokens = 96
    elif total_gib < 12:
        profile = "low"
        model = "mf-2steps"
        max_segment_tokens = 192
    elif total_gib < 16:
        profile = "balanced"
        model = "soar-or-mf-2steps"
        max_segment_tokens = 384
    else:
        profile = "quality"
        model = "soar-or-edit"
        max_segment_tokens = 500
    warnings: list[str] = []
    if free_gib is not None and total_gib > 0:
        if free_gib < 6 or free_gib / total_gib < 0.35:
            warnings.append("当前空闲显存偏低；请释放其他模型，并开启生成后释放模型。")
        elif free_gib / total_gib < 0.6:
            warnings.append(
                "当前已有较多显存占用，建议避免 TTS、Edit 与 ASR 同时驻留。"
            )
    return {
        "profile": profile,
        "total_gib": total_gib,
        "free_gib": free_gib,
        "recommended_model": model,
        "max_segment_tokens": max_segment_tokens,
        "precision": "bfloat16",
        "release_after_run": profile in {"limited", "low"},
        "heuristic": True,
        "warnings": warnings,
    }


def inspect_environment(*, import_torch: bool = True) -> EnvironmentReport:
    packages = {name: _package_version(name) for name in RUNTIME_PACKAGES}
    python_version = sys.version_info[:3]
    supported_python = (3, 10) <= python_version < (3, 13)
    warnings: list[str] = []
    missing = [name for name in REQUIRED_RUNTIME_PACKAGES if packages[name] is None]
    if missing:
        warnings.append(
            "Missing required runtime packages: " + ", ".join(missing) + "."
        )
    if not supported_python:
        warnings.append(
            "dots.tts v0.3.1 officially supports Python 3.10-3.12; "
            f"detected {platform.python_version()}."
        )

    torch_cuda_available = False
    torch_cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_count = 0
    gpu_devices: list[dict[str, Any]] = []
    cuda_bf16_supported = False
    cudnn_version: int | None = None
    vram_recommendation = recommend_vram(0)
    if import_torch and packages["torch"] is not None:
        try:
            import torch

            torch_cuda_available = bool(torch.cuda.is_available())
            torch_cuda_version = torch.version.cuda
            if torch_cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = int(torch.cuda.device_count())
                cuda_bf16_supported = bool(torch.cuda.is_bf16_supported())
                cudnn_version = torch.backends.cudnn.version()
                for index in range(gpu_count):
                    properties = torch.cuda.get_device_properties(index)
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    gpu_devices.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "capability": [properties.major, properties.minor],
                            "total_bytes": int(total_bytes),
                            "free_bytes": int(free_bytes),
                            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                        }
                    )
                primary = gpu_devices[0]
                vram_recommendation = recommend_vram(
                    primary["total_bytes"], primary["free_bytes"]
                )
            else:
                warnings.append(
                    "CUDA is unavailable. Desktop BF16 inference requires a "
                    "CUDA-enabled PyTorch build and NVIDIA GPU."
                )
        except Exception as exc:  # diagnostics must not break ComfyUI startup
            warnings.append(f"PyTorch diagnostics failed: {exc}")

    if packages["transformers"] is not None:
        major_minor = packages["transformers"].split(".")[:2]
        try:
            transformers_tuple = tuple(int(part) for part in major_minor)
        except ValueError:
            transformers_tuple = ()
        if transformers_tuple and transformers_tuple < (4, 57):
            warnings.append(
                "dots.tts v0.3.1 requires transformers 4.57 or newer; "
                f"detected {packages['transformers']}."
            )

    return EnvironmentReport(
        python=platform.python_version(),
        executable=sys.executable,
        os=platform.platform(),
        machine=platform.machine(),
        packages=packages,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_devices=tuple(gpu_devices),
        cuda_bf16_supported=cuda_bf16_supported,
        cudnn_version=cudnn_version,
        vram_recommendation=vram_recommendation,
        supported_python=supported_python,
        warnings=tuple(warnings),
    )
