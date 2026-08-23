"""Pinned model catalog and local-directory validation.

The desktop bundle and ComfyUI node share this file so model names, sampling
contracts, revisions, and validation behavior cannot silently diverge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _default_catalog_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = (
        module_path.parent.parent / "manifests" / "dots_tts_models.json",
        module_path.parents[2] / "manifests" / "dots_tts_models.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


@dataclass(frozen=True)
class ModelFile:
    relative_path: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class SamplingContract:
    locked: bool
    ode_method: str
    num_steps: int
    guidance_scale: float


@dataclass(frozen=True)
class ModelArtifact:
    key: str
    display_name: str
    repo_id: str
    revision: str
    runtime: str
    recommended: bool
    description: str
    sampling: SamplingContract
    files: tuple[ModelFile, ...]

    @property
    def expected_size(self) -> int:
        return sum(item.size for item in self.files)


@dataclass(frozen=True)
class ValidationIssue:
    relative_path: str
    reason: str
    expected: str
    actual: str


@dataclass(frozen=True)
class ValidationReport:
    model_key: str
    directory: Path
    issues: tuple[ValidationIssue, ...]
    checked_hashes: bool

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "directory": str(self.directory),
            "valid": self.valid,
            "checked_hashes": self.checked_hashes,
            "issues": [
                {
                    "relative_path": issue.relative_path,
                    "reason": issue.reason,
                    "expected": issue.expected,
                    "actual": issue.actual,
                }
                for issue in self.issues
            ],
        }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelCatalog:
    def __init__(self, artifacts: Iterable[ModelArtifact], *, source: Path):
        self.source = source
        self._artifacts = {artifact.key: artifact for artifact in artifacts}
        if not self._artifacts:
            raise ValueError("Model catalog cannot be empty.")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ModelCatalog":
        source = Path(path) if path is not None else _default_catalog_path()
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported model catalog schema: {data.get('schema_version')!r}"
            )

        artifacts: list[ModelArtifact] = []
        for key, raw in data["models"].items():
            sampling = raw["sampling"]
            files = tuple(
                ModelFile(
                    relative_path=relative_path,
                    size=int(metadata["size"]),
                    sha256=metadata.get("sha256"),
                )
                for relative_path, metadata in raw["files"].items()
            )
            artifacts.append(
                ModelArtifact(
                    key=key,
                    display_name=raw["display_name"],
                    repo_id=raw["repo_id"],
                    revision=raw["revision"],
                    runtime=raw["runtime"],
                    recommended=bool(raw.get("recommended", False)),
                    description=raw.get("description", ""),
                    sampling=SamplingContract(
                        locked=bool(sampling["locked"]),
                        ode_method=str(sampling["ode_method"]),
                        num_steps=int(sampling["num_steps"]),
                        guidance_scale=float(sampling["guidance_scale"]),
                    ),
                    files=files,
                )
            )
        return cls(artifacts, source=source)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._artifacts)

    def values(self) -> tuple[ModelArtifact, ...]:
        return tuple(self._artifacts.values())

    def get(self, key_or_repo_id: str) -> ModelArtifact:
        artifact = self._artifacts.get(key_or_repo_id)
        if artifact is not None:
            return artifact
        for candidate in self._artifacts.values():
            if candidate.repo_id == key_or_repo_id:
                return candidate
        raise KeyError(f"Unknown dots.tts model: {key_or_repo_id}")

    def validate_directory(
        self,
        key_or_repo_id: str,
        directory: str | Path,
        *,
        check_hashes: bool = False,
    ) -> ValidationReport:
        artifact = self.get(key_or_repo_id)
        root = Path(directory)
        issues: list[ValidationIssue] = []

        if not root.is_dir():
            issues.append(
                ValidationIssue(".", "missing_directory", "directory", "missing")
            )
            return ValidationReport(artifact.key, root, tuple(issues), check_hashes)

        for item in artifact.files:
            local_path = root / Path(item.relative_path)
            if not local_path.is_file():
                issues.append(
                    ValidationIssue(
                        item.relative_path, "missing_file", str(item.size), "missing"
                    )
                )
                continue

            actual_size = local_path.stat().st_size
            if actual_size != item.size:
                issues.append(
                    ValidationIssue(
                        item.relative_path,
                        "size_mismatch",
                        str(item.size),
                        str(actual_size),
                    )
                )
                continue

            if check_hashes and item.sha256:
                actual_hash = _sha256(local_path)
                if actual_hash.lower() != item.sha256.lower():
                    issues.append(
                        ValidationIssue(
                            item.relative_path,
                            "sha256_mismatch",
                            item.sha256,
                            actual_hash,
                        )
                    )

        return ValidationReport(artifact.key, root, tuple(issues), check_hashes)
