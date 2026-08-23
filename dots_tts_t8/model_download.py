"""Pinned Hugging Face model download used by the desktop launcher."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .model_catalog import ModelCatalog, ValidationReport


ProgressCallback = Callable[[dict], None]


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _emit(callback: ProgressCallback | None, event: str, **payload) -> None:
    if callback is not None:
        callback({"event": event, **payload})


def download_model(
    model_key: str,
    target_root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    max_attempts: int = 4,
    retry_delay_seconds: float = 2.0,
    endpoint: str | None = None,
    verify_hashes: bool = True,
) -> tuple[Path, ValidationReport]:
    catalog = ModelCatalog.load(catalog_path)
    artifact = catalog.get(model_key)
    root = Path(target_root).expanduser().resolve()
    model_dir = root / artifact.repo_id.rsplit("/", 1)[-1]
    model_dir.mkdir(parents=True, exist_ok=True)

    _emit(
        progress,
        "start",
        model_key=artifact.key,
        repo_id=artifact.repo_id,
        revision=artifact.revision,
        directory=str(model_dir),
    )

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    downloaded = None
    for attempt in range(1, max_attempts + 1):
        try:
            downloaded = _snapshot_download(
                repo_id=artifact.repo_id,
                revision=artifact.revision,
                local_dir=model_dir,
                cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
                endpoint=endpoint,
            )
            break
        except Exception as exc:
            _emit(
                progress,
                "retry" if attempt < max_attempts else "error",
                attempt=attempt,
                max_attempts=max_attempts,
                error=f"{type(exc).__name__}: {exc}",
            )
            if attempt >= max_attempts:
                raise
            time.sleep(retry_delay_seconds * (2 ** (attempt - 1)))

    assert downloaded is not None
    resolved = Path(downloaded).resolve()
    _emit(progress, "verify", check_hashes=verify_hashes)
    report = catalog.validate_directory(
        artifact.key, resolved, check_hashes=verify_hashes
    )
    _emit(progress, "complete", report=report.as_dict())
    if not report.valid:
        raise RuntimeError(
            "Downloaded model failed validation: "
            + json.dumps(report.as_dict(), ensure_ascii=False)
        )
    return resolved, report
