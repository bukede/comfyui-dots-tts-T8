"""dots.tts package."""

from __future__ import annotations


def _check_torch_install() -> None:
    """Fail fast on torch/torchaudio install problems.

    Runs at package import so users get an actionable message instead of a
    cryptic ``libcudart.so.X: cannot open shared object file`` from
    torchaudio's C extension loader, or a downstream dtype mismatch after a
    silent CPU fallback. Reads versions via ``importlib.metadata`` to avoid
    triggering the broken library load itself.
    """
    import importlib.metadata as _md

    try:
        torch_version = _md.version("torch")
        torchaudio_version = _md.version("torchaudio")
    except _md.PackageNotFoundError:
        return

    _install_hint = (
        "Each torch minor release only ships wheels for a fixed set of CUDA "
        "versions. Pick a torch + torchaudio combo that matches your system "
        "CUDA from the PyTorch install matrix and reinstall both together: "
        "https://pytorch.org/get-started/locally/\n"
        "e.g. for CUDA 12.8:\n"
        "  pip install torch==2.8.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )

    torch_minor = tuple(torch_version.split("+")[0].split(".")[:2])
    torchaudio_minor = tuple(torchaudio_version.split("+")[0].split(".")[:2])
    # comfyui-h3 patch 001: torchaudio stopped at 2.11 (no 2.12/2.13 wheels); the
    # production lock pairs torch 2.13 with torchaudio 2.11 and dots.tts only uses
    # its pure-torch ops (functional.resample, transforms.Resample, compliance.kaldi).
    # Allow an older torchaudio at minor >= 11 instead of refusing to import.
    if torch_minor != torchaudio_minor and not (
        torch_minor[0] == torchaudio_minor[0] == "2"
        and int(torchaudio_minor[1]) >= 11
        and int(torch_minor[1]) > int(torchaudio_minor[1])
    ):
        raise RuntimeError(
            f"torch ({torch_version}) and torchaudio ({torchaudio_version}) "
            f"minor versions do not match.\n{_install_hint}"
        )

    try:
        import torchaudio  # noqa: F401
    except OSError as exc:
        if "libcudart" in str(exc) or "libcuda" in str(exc):
            import torch  # deferred; only reached if torchaudio failed

            raise RuntimeError(
                f"torchaudio failed to load its native extension: {exc}\n"
                f"Your torch/torchaudio wheels were built for CUDA "
                f"{torch.version.cuda!r}, but the required runtime library is "
                f"not present on this machine.\n{_install_hint}"
            ) from exc
        raise


_check_torch_install()
del _check_torch_install
