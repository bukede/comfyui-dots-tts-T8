"""Client for the authenticated T8 desktop ASR sidecar."""

from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_VERSION = 1
MAX_AUDIO_BYTES = 200 * 1024 * 1024
DISCOVERY_PATH = Path(tempfile.gettempdir()) / "dots-tts-T8" / "asr-sidecar.json"


def _validate_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ValueError("ASR sidecar endpoint must be loopback HTTP.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("ASR sidecar endpoint contains unsupported URL components.")
    return endpoint.rstrip("/")


def read_discovery(path: str | Path = DISCOVERY_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("api_version", 0)) != API_VERSION:
        raise RuntimeError("ASR sidecar API version mismatch.")
    if float(payload.get("expires_at", 0)) <= time.time():
        raise RuntimeError("ASR sidecar discovery has expired.")
    payload["endpoint"] = _validate_endpoint(str(payload.get("endpoint", "")))
    if len(str(payload.get("token", ""))) < 32:
        raise RuntimeError("ASR sidecar discovery token is invalid.")
    return payload


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirect refused", headers, fp
        )


def _request(
    discovery: dict[str, Any],
    method: str,
    route: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    request = urllib.request.Request(
        _validate_endpoint(str(discovery["endpoint"])) + route,
        data=data,
        headers={
            "X-T8-ASR-Token": str(discovery["token"]),
            **(headers or {}),
        },
        method=method,
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(
            request, timeout=timeout
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ASR sidecar HTTP {exc.code}: {detail}") from exc


def sidecar_health(path: str | Path = DISCOVERY_PATH) -> dict[str, Any]:
    return _request(read_discovery(path), "GET", "/v1/health", timeout=5)


def sidecar_transcribe(
    audio_path: str | Path,
    *,
    language: str | None = None,
    discovery_path: str | Path = DISCOVERY_PATH,
) -> dict[str, Any]:
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    if size <= 0 or size > MAX_AUDIO_BYTES:
        raise ValueError("ASR sidecar audio size is outside the supported range.")
    return _request(
        read_discovery(discovery_path),
        "POST",
        "/v1/transcribe",
        data=source.read_bytes(),
        headers={
            "Content-Type": "audio/wav",
            "Content-Length": str(size),
            "X-T8-ASR-Language": language or "",
        },
    )


def sidecar_release(path: str | Path = DISCOVERY_PATH) -> bool:
    return bool(
        _request(read_discovery(path), "POST", "/v1/release", data=b"").get("released")
    )


__all__ = [
    "API_VERSION",
    "DISCOVERY_PATH",
    "MAX_AUDIO_BYTES",
    "read_discovery",
    "sidecar_health",
    "sidecar_release",
    "sidecar_transcribe",
]
