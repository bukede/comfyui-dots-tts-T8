from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dots_tts_t8.model_catalog import ModelCatalog, ValidationReport


MISSING_MODEL_OPTION = "未找到模型 · Run download_models.py"


def _comfy_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        import folder_paths

        return Path(folder_paths.base_path).resolve()
    except Exception:
        return Path.cwd().resolve()


def register_model_paths() -> None:
    try:
        import folder_paths

        model_path = Path(folder_paths.models_dir) / "TTS"
        model_path.mkdir(parents=True, exist_ok=True)
        folder_paths.add_model_folder_path("TTS", str(model_path), is_default=True)
    except Exception:
        return


def model_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    try:
        import folder_paths

        roots.extend(Path(path) for path in folder_paths.get_folder_paths("TTS"))
        roots.append(Path(folder_paths.models_dir) / "dotstts")
    except Exception:
        root = _comfy_root() / "models"
        roots.extend((root / "TTS", root / "dotstts"))

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return tuple(unique)


def candidate_directories() -> dict[str, Path]:
    catalog = ModelCatalog.load()
    candidates: dict[str, Path] = {}
    for artifact in catalog.values():
        folder_name = artifact.repo_id.rsplit("/", 1)[-1]
        for root in model_roots():
            paths = (
                root / "dots.tts" / folder_name,
                root / folder_name,
            )
            for path in paths:
                if path.is_dir():
                    candidates[artifact.key] = path
                    break
            if artifact.key in candidates:
                break
    return candidates


def model_options() -> list[str]:
    catalog = ModelCatalog.load()
    return [
        f"{artifact.display_name} [{artifact.key}]"
        for artifact in catalog.values()
    ]


def _key_from_option(option: str) -> str:
    if option.endswith("]") and "[" in option:
        return option.rsplit("[", 1)[-1][:-1]
    return option


def resolve_model(option: str, custom_model_path: str = "") -> tuple[str, Path]:
    catalog = ModelCatalog.load()
    if custom_model_path.strip():
        key = _key_from_option(option)
        if key == MISSING_MODEL_OPTION:
            raise ValueError("使用自定义路径时仍需选择模型类型。")
        catalog.get(key)
        return key, Path(custom_model_path).expanduser().resolve()

    key = _key_from_option(option)
    path = candidate_directories().get(key)
    if path is None:
        raise FileNotFoundError(
            "未找到 dots.tts 模型。标准位置：ComfyUI/models/TTS/dots.tts/"
            "dots.tts-soar 或 dots.tts-mf-2steps。"
        )
    return key, path


def validate_model(
    model_key: str, model_dir: Path, *, check_hashes: bool = False
) -> ValidationReport:
    return ModelCatalog.load().validate_directory(
        model_key, model_dir, check_hashes=check_hashes
    )


def model_fingerprint(model_key: str, model_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(model_key.encode())
    for name in ("config.json", "model.safetensors", "vocoder.safetensors"):
        path = model_dir / name
        if path.exists():
            stat = path.stat()
            digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()
