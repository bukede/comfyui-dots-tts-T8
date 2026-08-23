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
    supported_python: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment(*, import_torch: bool = True) -> EnvironmentReport:
    packages = {name: _package_version(name) for name in RUNTIME_PACKAGES}
    python_version = sys.version_info[:3]
    supported_python = (3, 10) <= python_version < (3, 13)
    warnings: list[str] = []
    missing = [name for name in REQUIRED_RUNTIME_PACKAGES if packages[name] is None]
    if missing:
        warnings.append("Missing required runtime packages: " + ", ".join(missing) + ".")
    if not supported_python:
        warnings.append(
            "dots.tts v0.3.1 officially supports Python 3.10-3.12; "
            f"detected {platform.python_version()}."
        )

    torch_cuda_available = False
    torch_cuda_version: str | None = None
    gpu_name: str | None = None
    if import_torch and packages["torch"] is not None:
        try:
            import torch

            torch_cuda_available = bool(torch.cuda.is_available())
            torch_cuda_version = torch.version.cuda
            if torch_cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
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
        supported_python=supported_python,
        warnings=tuple(warnings),
    )
