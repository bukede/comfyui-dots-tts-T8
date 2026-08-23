"""Thread-safe lazy dots.tts runtime lifecycle management."""

from __future__ import annotations

import gc
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeSpec:
    model_name_or_path: str
    revision: str | None = None
    cache_dir: str | None = None
    precision: str = "bfloat16"
    optimize: bool = False
    max_generate_length: int = 500
    max_sequence_length: int = 2048
    warmup_on_optimize: bool = True

    def normalized(self) -> "RuntimeSpec":
        model = self.model_name_or_path.strip()
        if not model:
            raise ValueError("model_name_or_path cannot be empty.")
        cache_dir = str(Path(self.cache_dir).resolve()) if self.cache_dir else None
        return RuntimeSpec(
            model_name_or_path=model,
            revision=self.revision or None,
            cache_dir=cache_dir,
            precision=self.precision.strip() or "bfloat16",
            optimize=bool(self.optimize),
            max_generate_length=int(self.max_generate_length),
            max_sequence_length=int(self.max_sequence_length),
            warmup_on_optimize=bool(self.warmup_on_optimize),
        )


class RuntimeManager:
    """Own exactly one heavyweight model per process and serialize inference."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runtime: Any | None = None
        self._spec: RuntimeSpec | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    @property
    def spec(self) -> RuntimeSpec | None:
        return self._spec

    def load(self, spec: RuntimeSpec):
        normalized = spec.normalized()
        if normalized.max_generate_length <= 0:
            raise ValueError("max_generate_length must be positive.")
        if normalized.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")

        with self._lock:
            if self._runtime is not None and self._spec == normalized:
                return self._runtime
            self._release_locked()

            from dots_tts.runtime import DotsTtsRuntime

            runtime = DotsTtsRuntime.from_pretrained(
                normalized.model_name_or_path,
                revision=normalized.revision,
                cache_dir=normalized.cache_dir,
                precision=normalized.precision,
                optimize=normalized.optimize,
                max_generate_length=normalized.max_generate_length,
                max_sequence_length=normalized.max_sequence_length,
                warmup_on_optimize=normalized.warmup_on_optimize,
            )
            self._runtime = runtime
            self._spec = normalized
            return runtime

    def generate(self, spec: RuntimeSpec, *, seed: int, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            runtime = self.load(spec)
            from dots_tts.utils.util import seed_everything

            seed_everything(int(seed))
            return runtime.generate(**kwargs)

    def release(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        if self._runtime is None:
            self._spec = None
            return
        runtime = self._runtime
        self._runtime = None
        self._spec = None
        del runtime
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if ipc_collect is not None:
                    ipc_collect()
        except Exception:
            # Releasing a model is best-effort and must remain safe during app exit.
            pass

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": self.loaded,
                "spec": asdict(self._spec) if self._spec is not None else None,
            }
