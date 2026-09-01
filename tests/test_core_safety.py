from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dots_tts.models.dots_tts.core import IOHelper  # noqa: E402
from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: E402
from dots_tts.runtime import DotsTtsRuntime  # noqa: E402


class _UnsafePayload:
    def __reduce__(self):
        return (print, ("unsafe-payload-executed",))


def test_latent_stats_accepts_numpy_and_tensor_payloads(tmp_path: Path) -> None:
    for index, payload in enumerate(
        [
            {
                "mean": np.zeros(8, dtype=np.float32),
                "var": np.ones(8, dtype=np.float32),
            },
            {
                "mean": torch.zeros(8),
                "var": torch.ones(8),
            },
        ]
    ):
        path = tmp_path / f"stats-{index}.pt"
        torch.save(payload, path)

        helper = IOHelper(path)

        assert tuple(helper.global_mean.shape) == (8,)
        assert tuple(helper.global_var.shape) == (8,)


def test_latent_stats_rejects_unsafe_pickle_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "unsafe.pt"
    torch.save({"mean": _UnsafePayload(), "var": np.ones(8)}, path)

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        IOHelper(path)

    assert "unsafe-payload-executed" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "payload",
    [
        {"mean": torch.zeros(8)},
        {"mean": torch.zeros(8), "var": torch.ones(7)},
        {"mean": torch.full((8,), float("nan")), "var": torch.ones(8)},
        {"mean": torch.zeros(8), "var": torch.zeros(8)},
    ],
)
def test_latent_stats_rejects_invalid_values(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)

    with pytest.raises((TypeError, ValueError)):
        IOHelper(path)


def _runtime_stub(*, optimize: bool = False) -> DotsTtsRuntime:
    runtime = object.__new__(DotsTtsRuntime)
    runtime.model = SimpleNamespace(config=SimpleNamespace(sampling=None))
    runtime.optimize = optimize
    return runtime


def test_runtime_rejects_blank_text_and_invalid_sampling() -> None:
    runtime = _runtime_stub()

    with pytest.raises(ValueError, match="non-whitespace"):
        runtime._process_text("   ")

    for steps in (0, -1, 1001):
        with pytest.raises(ValueError, match="num_steps"):
            runtime.resolve_sampling_options(
                ode_method="euler", num_steps=steps, guidance_scale=1.2
            )
    for guidance in (float("nan"), float("inf"), 101.0):
        with pytest.raises(ValueError, match="guidance_scale"):
            runtime.resolve_sampling_options(
                ode_method="euler", num_steps=10, guidance_scale=guidance
            )
    with pytest.raises(ValueError, match="only ode_method='euler'"):
        _runtime_stub(optimize=True).resolve_sampling_options(
            ode_method="midpoint", num_steps=10, guidance_scale=1.2
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 10.1])
def test_runtime_rejects_invalid_speaker_scale(value: float) -> None:
    with pytest.raises(ValueError, match="speaker_scale"):
        DotsTtsRuntime._validate_speaker_scale(value)


def test_llm_config_rejects_dynamic_kernel_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "llm_config.json"
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "_attn_implementation_internal": "attacker/repository",
            }
        ),
        encoding="utf-8",
    )
    called = False

    def fake_from_dict(payload):
        nonlocal called
        called = True
        return payload

    monkeypatch.setattr(
        "dots_tts.models.dots_tts.model.Qwen2Config.from_dict", fake_from_dict
    )

    with pytest.raises(ValueError, match="reserved"):
        DotsTtsModel._load_llm_config(path)

    assert called is False
