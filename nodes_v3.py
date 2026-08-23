from __future__ import annotations

# ruff: noqa: E402 -- vendored dots_tts must be added to sys.path before imports.
import json
import logging
import sys
from pathlib import Path

try:
    from typing import override
except ImportError:  # Python 3.10-3.11

    def override(method):
        return method


from comfy_api.v0_0_2 import ComfyExtension, io

_NODE_ROOT = Path(__file__).resolve().parent
if str(_NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_NODE_ROOT))

from dots_tts_t8 import __version__
from dots_tts_t8.environment import inspect_environment
from dots_tts_t8.model_catalog import ModelCatalog

from .runtime.inference_adapter import (
    release_model,
    run_inference,
    run_long_text_inference,
    runtime_info,
)
from .runtime.model_store import (
    model_fingerprint,
    model_options,
    register_model_paths,
    resolve_model,
    validate_model,
)
from .runtime.types import ModelHandle, SamplingConfig, VoicePrompt

LOGGER = logging.getLogger("comfyui-dots-tts-T8")
CATEGORY = "T8star-Aix/Audio/dots.tts"
ModelType = io.Custom("T8_DOTS_TTS_MODEL")
VoiceType = io.Custom("T8_DOTS_TTS_VOICE")
SamplingType = io.Custom("T8_DOTS_TTS_SAMPLING")
LANGUAGES = [
    "inherit",
    "auto_detect",
    "none",
    "ZH",
    "EN",
    "YUE",
    "JA",
    "KO",
    "AR",
    "ES",
    "TR",
    "ID",
    "PT",
    "FR",
    "IT",
    "NL",
    "VI",
    "DE",
    "RU",
    "UK",
    "TH",
    "PL",
    "RO",
    "EL",
    "CS",
    "FI",
    "HI",
]


class T8DotsTTSModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options()
        return io.Schema(
            node_id="T8_DotsTTS_ModelLoader",
            display_name="dots.tts 模型加载器 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["dots.tts", "Dots TTS", "T8star-Aix"],
            description="发现并校验 SOAR / MF-2steps 模型；权重在首次生成时惰性加载。",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    display_name="模型",
                    options=options,
                    default=options[0],
                ),
                io.Combo.Input(
                    "precision",
                    display_name="精度",
                    options=["bfloat16", "float32"],
                    default="bfloat16",
                ),
                io.Boolean.Input(
                    "optimize",
                    display_name="Torch Compile 加速",
                    default=False,
                    tooltip="默认关闭；首次编译很慢，失败时请关闭。",
                ),
                io.Boolean.Input(
                    "release_after_run",
                    display_name="生成后释放模型",
                    default=False,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "verify_hashes",
                    display_name="完整 SHA-256 校验",
                    default=False,
                    advanced=True,
                ),
                io.String.Input(
                    "custom_model_path",
                    display_name="自定义模型绝对路径",
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                ModelType.Output("model", display_name="dots.tts 模型"),
                io.String.Output("model_info", display_name="模型信息 JSON"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model_name: str,
        precision: str,
        optimize: bool,
        release_after_run: bool,
        verify_hashes: bool,
        custom_model_path: str = "",
    ) -> str:
        try:
            key, path = resolve_model(model_name, custom_model_path)
            return model_fingerprint(key, path)
        except Exception as exc:
            return f"missing:{model_name}:{custom_model_path}:{exc}"

    @classmethod
    def validate_inputs(
        cls, model_name: str, custom_model_path: str = "", **kwargs
    ) -> bool | str:
        try:
            resolve_model(model_name, custom_model_path)
            return True
        except Exception as exc:
            return str(exc)

    @classmethod
    def execute(
        cls,
        model_name: str,
        precision: str,
        optimize: bool,
        release_after_run: bool,
        verify_hashes: bool,
        custom_model_path: str = "",
    ) -> io.NodeOutput:
        key, model_dir = resolve_model(model_name, custom_model_path)
        report = validate_model(key, model_dir, check_hashes=verify_hashes)
        if not report.valid:
            raise RuntimeError(json.dumps(report.as_dict(), ensure_ascii=False))
        artifact = ModelCatalog.load().get(key)
        handle = ModelHandle(
            model_key=key,
            model_dir=Path(model_dir),
            precision=precision,
            optimize=bool(optimize),
            release_after_run=bool(release_after_run),
        )
        info = {
            "model_key": key,
            "repo_id": artifact.repo_id,
            "revision": artifact.revision,
            "directory": str(model_dir),
            "precision": precision,
            "optimize": bool(optimize),
            "sampling": {
                "locked": artifact.sampling.locked,
                "num_steps": artifact.sampling.num_steps,
                "guidance_scale": artifact.sampling.guidance_scale,
            },
        }
        return io.NodeOutput(handle, json.dumps(info, ensure_ascii=False, indent=2))


class T8DotsTTSVoicePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_VoicePrompt",
            display_name="dots.tts 音色提示 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="封装参考音频、准确转写和语言；转写留空时使用 x-vector-only 克隆。",
            inputs=[
                io.Audio.Input("audio", display_name="参考音频"),
                io.String.Input(
                    "transcript",
                    display_name="参考音频准确转写",
                    multiline=True,
                    default="",
                ),
                io.Combo.Input(
                    "language",
                    display_name="参考语言",
                    options=LANGUAGES[1:],
                    default="auto_detect",
                ),
            ],
            outputs=[
                VoiceType.Output("voice", display_name="dots.tts 音色"),
                io.String.Output("voice_info", display_name="音色信息"),
            ],
        )

    @classmethod
    def execute(cls, audio: dict, transcript: str, language: str) -> io.NodeOutput:
        voice = VoicePrompt(
            audio=audio, transcript=transcript.strip(), language=language
        )
        info = {
            "mode": "continuation_clone" if transcript.strip() else "xvector_clone",
            "language": language,
            "sample_rate": int(audio.get("sample_rate", 0)),
        }
        return io.NodeOutput(voice, json.dumps(info, ensure_ascii=False))


class T8DotsTTSSampling(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Sampling",
            display_name="dots.tts 采样配置 · T8star-Aix",
            category=CATEGORY,
            description="SOAR 使用自定义 NFE/CFG；固定步数模型会自动忽略不兼容覆盖。",
            inputs=[
                io.Combo.Input(
                    "ode_method",
                    display_name="ODE 方法",
                    options=["euler", "midpoint", "rk4"],
                    default="euler",
                ),
                io.Int.Input(
                    "num_steps", display_name="NFE / Steps", default=10, min=1, max=32
                ),
                io.Float.Input(
                    "guidance_scale",
                    display_name="CFG",
                    default=1.2,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                ),
                io.Float.Input(
                    "speaker_scale",
                    display_name="音色强度",
                    default=1.5,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                ),
                io.Boolean.Input(
                    "normalize_text",
                    display_name="文本归一化（需可选依赖）",
                    default=False,
                ),
                io.Int.Input(
                    "max_generate_length",
                    display_name="最大音频 Patch",
                    default=500,
                    min=32,
                    max=1024,
                    advanced=True,
                ),
            ],
            outputs=[SamplingType.Output("sampling", display_name="采样配置")],
        )

    @classmethod
    def execute(
        cls,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float,
        normalize_text: bool,
        max_generate_length: int,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            SamplingConfig(
                ode_method=ode_method,
                num_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                speaker_scale=float(speaker_scale),
                normalize_text=bool(normalize_text),
                max_generate_length=int(max_generate_length),
            )
        )


class T8DotsTTSGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Generate",
            display_name="dots.tts 语音生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["voice clone", "语音克隆", "Dots TTS"],
            description="生成标准 48kHz ComfyUI AUDIO；音色不连接时为 text-only 模式。",
            inputs=[
                ModelType.Input("model", display_name="dots.tts 模型"),
                io.String.Input(
                    "text",
                    display_name="待合成文本",
                    multiline=True,
                    default="欢迎使用 dots.tts，来自 T8star-Aix。",
                    dynamic_prompts=True,
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=LANGUAGES,
                    default="inherit",
                ),
                io.Int.Input(
                    "seed",
                    display_name="seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                VoiceType.Input("voice", display_name="音色提示", optional=True),
                SamplingType.Input("sampling", display_name="采样配置", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="生成音频"),
                io.String.Output("generation_info", display_name="生成信息 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, text: str, **kwargs) -> bool | str:
        if not str(text).strip():
            return "待合成文本不能为空。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        text: str,
        language: str,
        seed: int,
        voice: VoicePrompt | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        audio, info = run_inference(
            handle=model,
            text=text.strip(),
            language=language,
            seed=int(seed),
            voice=voice,
            sampling=sampling,
        )
        return io.NodeOutput(audio, json.dumps(info, ensure_ascii=False, indent=2))


class T8DotsTTSLongText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_LongText",
            display_name="dots.tts 长文本生成 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["long text", "长文本", "pause", "Dots TTS"],
            description=(
                "按模型 token 数智能分段并拼接 48kHz AUDIO；支持 "
                "<pause=800ms> / <pause=1.2s> 显式停顿。"
            ),
            inputs=[
                ModelType.Input("model", display_name="dots.tts 模型"),
                io.String.Input(
                    "text",
                    display_name="长文本",
                    multiline=True,
                    default=(
                        "第一段内容。<pause=800ms>第二段内容，"
                        "会使用同一个音色继续生成。"
                    ),
                    dynamic_prompts=True,
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=LANGUAGES,
                    default="inherit",
                ),
                io.Int.Input(
                    "seed",
                    display_name="起始 seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "max_segment_tokens",
                    display_name="每段最大 Tokens",
                    default=120,
                    min=16,
                    max=512,
                ),
                io.Int.Input(
                    "segment_gap_ms",
                    display_name="自动段间停顿（毫秒）",
                    default=200,
                    min=0,
                    max=10_000,
                ),
                VoiceType.Input("voice", display_name="音色提示", optional=True),
                SamplingType.Input("sampling", display_name="采样配置", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="拼接音频"),
                io.String.Output("generation_info", display_name="分段信息 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str,
        max_segment_tokens: int,
        segment_gap_ms: int,
        **kwargs,
    ) -> bool | str:
        if not str(text).strip():
            return "长文本不能为空。"
        if int(max_segment_tokens) <= 0:
            return "每段最大 Tokens 必须大于 0。"
        if int(segment_gap_ms) < 0:
            return "自动段间停顿不能为负数。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        text: str,
        language: str,
        seed: int,
        max_segment_tokens: int,
        segment_gap_ms: int,
        voice: VoicePrompt | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        audio, info = run_long_text_inference(
            handle=model,
            text=text.strip(),
            language=language,
            seed=int(seed),
            voice=voice,
            sampling=sampling,
            max_segment_tokens=int(max_segment_tokens),
            segment_gap_ms=int(segment_gap_ms),
        )
        return io.NodeOutput(audio, json.dumps(info, ensure_ascii=False, indent=2))


class T8DotsTTSEnvironment(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Environment",
            display_name="dots.tts 环境诊断 · T8star-Aix",
            category=CATEGORY,
            description="只读检查 Python、依赖、CUDA 与当前模型缓存，不安装任何软件。",
            inputs=[],
            outputs=[io.String.Output("environment", display_name="环境报告 JSON")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        report = inspect_environment(import_torch=True).as_dict()
        report["runtime"] = runtime_info()
        return io.NodeOutput(json.dumps(report, ensure_ascii=False, indent=2))


class T8DotsTTSRelease(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Release",
            display_name="dots.tts 释放模型 · T8star-Aix",
            category=CATEGORY,
            description="释放本节点持有的 dots.tts 模型和 CUDA 缓存。",
            inputs=[
                ModelType.Input("model", display_name="dots.tts 模型", optional=True)
            ],
            outputs=[io.String.Output("result", display_name="释放结果 JSON")],
        )

    @classmethod
    def execute(cls, model: ModelHandle | None = None) -> io.NodeOutput:
        return io.NodeOutput(json.dumps(release_model(), ensure_ascii=False, indent=2))


class T8DotsTTSExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        register_model_paths()
        LOGGER.info("Loaded comfyui-dots-tts-T8 v%s (ComfyUI V3)", __version__)

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            T8DotsTTSModelLoader,
            T8DotsTTSVoicePrompt,
            T8DotsTTSSampling,
            T8DotsTTSGenerate,
            T8DotsTTSLongText,
            T8DotsTTSEnvironment,
            T8DotsTTSRelease,
        ]


async def comfy_entrypoint() -> T8DotsTTSExtension:
    return T8DotsTTSExtension()


__all__ = [
    "T8DotsTTSModelLoader",
    "T8DotsTTSVoicePrompt",
    "T8DotsTTSSampling",
    "T8DotsTTSGenerate",
    "T8DotsTTSLongText",
    "T8DotsTTSEnvironment",
    "T8DotsTTSRelease",
    "T8DotsTTSExtension",
    "comfy_entrypoint",
]
