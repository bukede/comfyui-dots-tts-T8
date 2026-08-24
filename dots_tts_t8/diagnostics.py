"""Privacy-bounded support reports for desktop and ComfyUI distributions."""

from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .environment import inspect_environment
from .model_catalog import ModelCatalog

MAX_LOG_BYTES = 2 * 1024 * 1024
_TOKEN = re.compile(r"(?i)(token|password|secret|authorization)([\s:=]+)([^\s]+)")
_HOME = re.compile(re.escape(str(Path.home())), re.IGNORECASE)


def _sanitize_log(text: str) -> str:
    value = _HOME.sub("<HOME>", text)
    return _TOKEN.sub(r"\1\2<REDACTED>", value)


def diagnostic_report(
    *, app: str, version: str, runtime: dict[str, Any] | None = None
) -> dict[str, Any]:
    catalog = ModelCatalog.load()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application": {"name": app, "version": version},
        "environment": inspect_environment(import_torch=True).as_dict(),
        "runtime": runtime or {},
        "model_catalog": [
            {
                "key": artifact.key,
                "repo_id": artifact.repo_id,
                "revision": artifact.revision,
                "runtime": artifact.runtime,
                "expected_size": artifact.expected_size,
            }
            for artifact in catalog.values()
        ],
        "privacy": {
            "included": ["versions", "gpu", "memory", "model manifest", "log tail"],
            "excluded": [
                "environment variables",
                "tokens",
                "audio",
                "outputs",
                "model weights",
                "workflows",
            ],
        },
    }


def write_diagnostic_bundle(
    output_path: str | Path,
    *,
    app: str,
    version: str,
    runtime: dict[str, Any] | None = None,
    log_files: Iterable[str | Path] = (),
) -> tuple[Path, dict[str, Any]]:
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("诊断包输出必须使用 .zip 扩展名。")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = diagnostic_report(app=app, version=version, runtime=runtime)
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "diagnostic-report.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        for index, raw in enumerate(log_files, start=1):
            path = Path(raw)
            if not path.is_file() or path.is_symlink():
                continue
            with path.open("rb") as stream:
                size = path.stat().st_size
                if size > MAX_LOG_BYTES:
                    stream.seek(size - MAX_LOG_BYTES)
                content = stream.read(MAX_LOG_BYTES).decode("utf-8", errors="replace")
            archive.writestr(
                f"logs/log-{index:02d}-{path.name}", _sanitize_log(content)
            )
    temporary.replace(output)
    return output, report


__all__ = ["MAX_LOG_BYTES", "diagnostic_report", "write_diagnostic_bundle"]
