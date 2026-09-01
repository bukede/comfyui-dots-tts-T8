from __future__ import annotations

# ruff: noqa: E402 -- vendored dots_tts must be added to sys.path before imports.
import json
import logging
import sys
import time
from pathlib import Path

try:
    from typing import override
except ImportError:  # Python 3.10-3.11

    def override(method):
        return method


from comfy_api.v0_0_2 import ComfyExtension, io, ui

_NODE_ROOT = Path(__file__).resolve().parent
if str(_NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_NODE_ROOT))

from dots_tts_t8 import __version__
from dots_tts_t8.diagnostics import write_diagnostic_bundle
from dots_tts_t8.environment import inspect_environment
from dots_tts_t8.media import parse_audio_formats
from dots_tts_t8.model_catalog import ModelCatalog
from dots_tts_t8.production import build_production_plan
from dots_tts_t8.pronunciation import (
    apply_pronunciation,
    parse_pronunciation_rules,
)
from dots_tts_t8.subtitles import parse_subtitles
from dots_tts_t8.templates import parse_voice_library
from dots_tts_t8.timeline import rows_to_cues, timeline_document

from .runtime.inference_adapter import (
    release_model,
    run_batch_inference,
    run_edit_inference,
    run_inference,
    run_long_text_inference,
    run_project_inference,
    runtime_info,
)
from .runtime.model_store import (
    model_fingerprint,
    model_options,
    register_model_paths,
    resolve_model,
    validate_model,
)
from .runtime.transcription_adapter import run_transcription
from .runtime.types import ModelHandle, SamplingConfig, SpeakerVoiceMap, VoicePrompt

LOGGER = logging.getLogger("comfyui-dots-tts-T8")
CATEGORY = "T8star-Aix/Audio/dots.tts"
ModelType = io.Custom("T8_DOTS_TTS_MODEL")
EditModelType = io.Custom("T8_DOTS_TTS_EDIT_MODEL")
VoiceType = io.Custom("T8_DOTS_TTS_VOICE")
SamplingType = io.Custom("T8_DOTS_TTS_SAMPLING")
SpeakerMapType = io.Custom("T8_DOTS_TTS_SPEAKER_MAP")
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


def _comfy_directory(kind: str) -> Path | None:
    try:
        import folder_paths

        if kind == "input":
            return Path(folder_paths.get_input_directory()).resolve()
        if kind == "output":
            return Path(folder_paths.get_output_directory()).resolve()
    except (ImportError, AttributeError):
        return None
    raise ValueError(f"Unsupported ComfyUI directory kind: {kind}")


def _require_within(path: Path, root: Path | None, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{label} 必须位于 ComfyUI {root} 内。"
            ) from exc
    return resolved


def _input_path(value: str | Path, *, base: Path | None = None) -> Path:
    root = _comfy_directory("input")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (base or root or Path.cwd()) / candidate
    return _require_within(candidate, root, "输入路径")


def _project_output_root(value: str) -> Path:
    root = _comfy_directory("output")
    if value.strip():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (root or Path.cwd() / "output") / candidate
        return _require_within(candidate, root, "输出目录")
    return (root or Path.cwd() / "output") / "dots_tts_t8_projects"


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
            content_fingerprint=model_fingerprint(key, model_dir),
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


class T8DotsTTSEditModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options(runtime="edit")
        return io.Schema(
            node_id="T8_DotsTTS_EditModelLoader",
            display_name="dots.tts Edit 模型加载器 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["dots.tts.edit", "speech edit", "语音编辑"],
            description="发现并校验官方 dots.tts.edit；权重在首次编辑时惰性加载。",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    display_name="编辑模型",
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
                    display_name="编辑后释放模型",
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
                    display_name="自定义编辑模型绝对路径",
                    default="",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[
                EditModelType.Output("model", display_name="dots.tts Edit 模型"),
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
            key, _path = resolve_model(model_name, custom_model_path)
            if ModelCatalog.load().get(key).runtime != "edit":
                return "请选择 dots.tts.edit 模型。"
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
        artifact = ModelCatalog.load().get(key)
        if artifact.runtime != "edit":
            raise ValueError("请选择 dots.tts.edit 模型。")
        report = validate_model(key, model_dir, check_hashes=verify_hashes)
        if not report.valid:
            raise RuntimeError(json.dumps(report.as_dict(), ensure_ascii=False))
        handle = ModelHandle(
            model_key=key,
            model_dir=Path(model_dir),
            precision=precision,
            optimize=bool(optimize),
            release_after_run=bool(release_after_run),
            content_fingerprint=model_fingerprint(key, model_dir),
        )
        info = {
            "model_key": key,
            "repo_id": artifact.repo_id,
            "revision": artifact.revision,
            "runtime": artifact.runtime,
            "directory": str(model_dir),
            "precision": precision,
            "optimize": bool(optimize),
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


class T8DotsTTSSpeakerMap(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_SpeakerMap",
            display_name="dots.tts 多角色音色映射 · T8star-Aix",
            category=CATEGORY,
            description=(
                "把字幕角色映射到 A-D 槽位，或从 JSON 音色库加载任意数量的"
                "本地参考音频；未命中角色使用默认音色。"
            ),
            inputs=[
                io.String.Input(
                    "mapping_json",
                    display_name="角色到槽位 JSON",
                    multiline=True,
                    default='{"旁白":"A","小明":"B"}',
                ),
                io.String.Input(
                    "voice_library",
                    display_name="音色库 JSON / 文件路径（可选）",
                    multiline=True,
                    default="",
                    advanced=True,
                ),
                VoiceType.Input("voice_a", display_name="音色 A", optional=True),
                VoiceType.Input("voice_b", display_name="音色 B", optional=True),
                VoiceType.Input("voice_c", display_name="音色 C", optional=True),
                VoiceType.Input("voice_d", display_name="音色 D", optional=True),
                VoiceType.Input(
                    "fallback_voice", display_name="默认音色", optional=True
                ),
            ],
            outputs=[
                SpeakerMapType.Output("speaker_map", display_name="多角色音色映射"),
                io.String.Output("mapping_info", display_name="映射信息 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls, mapping_json: str, voice_library: str = "", **kwargs
    ) -> bool | str:
        try:
            value = json.loads(mapping_json or "{}")
            if not isinstance(value, dict):
                return "角色到槽位 JSON 必须是对象。"
            for speaker, slot in value.items():
                selected = slot.get("slot") if isinstance(slot, dict) else slot
                has_audio = isinstance(slot, dict) and bool(
                    slot.get("audio", slot.get("prompt_audio_path"))
                )
                if not has_audio and str(selected).upper() not in {"A", "B", "C", "D"}:
                    return f"角色 {speaker!r} 必须使用 A-D 槽位或本地 audio。"
            if voice_library.strip():
                parse_voice_library(voice_library)
            return True
        except Exception as exc:
            return f"角色映射 JSON 无效：{exc}"

    @classmethod
    def execute(
        cls,
        mapping_json: str,
        voice_library: str = "",
        voice_a: VoicePrompt | None = None,
        voice_b: VoicePrompt | None = None,
        voice_c: VoicePrompt | None = None,
        voice_d: VoicePrompt | None = None,
        fallback_voice: VoicePrompt | None = None,
    ) -> io.NodeOutput:
        value: dict = {}
        library_base = Path.cwd()
        if voice_library.strip():
            library_value: str | Path = voice_library
            stripped_library = voice_library.strip()
            if not stripped_library.startswith("{"):
                library_value = _input_path(stripped_library)
            library = parse_voice_library(library_value)
            value.update(library.voices)
            if isinstance(library_value, Path):
                library_base = library_value.resolve().parent
        value.update(json.loads(mapping_json or "{}"))
        slots = {"A": voice_a, "B": voice_b, "C": voice_c, "D": voice_d}
        voices: dict[str, VoicePrompt] = {}
        sources: dict[str, str] = {}
        for raw_speaker, raw_slot in value.items():
            speaker = str(raw_speaker).strip()
            selected = raw_slot.get("slot") if isinstance(raw_slot, dict) else raw_slot
            slot = str(selected).upper()
            if not speaker:
                raise ValueError("角色名不能为空。")
            audio_value = (
                raw_slot.get("audio", raw_slot.get("prompt_audio_path"))
                if isinstance(raw_slot, dict)
                else None
            )
            if audio_value:
                audio_path = _input_path(str(audio_value), base=library_base)
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"角色 {speaker!r} 的参考音频不存在：{audio_path}"
                    )
                import soundfile as sf
                import torch

                waveform, sample_rate = sf.read(
                    audio_path, dtype="float32", always_2d=True
                )
                voice = VoicePrompt(
                    audio={
                        "waveform": torch.from_numpy(waveform.T.copy()).unsqueeze(0),
                        "sample_rate": int(sample_rate),
                    },
                    transcript=str(
                        raw_slot.get("transcript", raw_slot.get("prompt_text", ""))
                    ),
                    language=str(raw_slot.get("language", "ZH")) or "ZH",
                )
                sources[speaker] = str(audio_path)
            else:
                voice = slots.get(slot)
                if voice is None:
                    raise ValueError(f"角色 {speaker!r} 使用的音色槽 {slot} 未连接。")
                sources[speaker] = slot
            voices[speaker] = voice
        speaker_map = SpeakerVoiceMap(voices=voices, fallback=fallback_voice)
        info = {
            "speakers": {speaker: sources[speaker] for speaker in voices},
            "speaker_count": len(voices),
            "has_fallback": fallback_voice is not None,
        }
        return io.NodeOutput(
            speaker_map, json.dumps(info, ensure_ascii=False, indent=2)
        )


class T8DotsTTSPronunciation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Pronunciation",
            display_name="dots.tts 发音词典 · T8star-Aix",
            category=CATEGORY,
            description="覆盖数字、人名和专有词读音；支持 JSON 或 `原文 => 读法`。",
            inputs=[
                io.String.Input(
                    "text",
                    display_name="原文本",
                    multiline=True,
                    default="2026 年，T8 发布。",
                ),
                io.String.Input(
                    "dictionary",
                    display_name="发音词典",
                    multiline=True,
                    default="2026 => 二零二六\nT8 => T 八",
                ),
            ],
            outputs=[
                io.String.Output("processed_text", display_name="替换后文本"),
                io.String.Output("report", display_name="替换报告 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, text: str, dictionary: str, **kwargs) -> bool | str:
        if not str(text).strip():
            return "原文本不能为空。"
        try:
            parse_pronunciation_rules(dictionary)
            return True
        except Exception as exc:
            return f"发音词典无效：{exc}"

    @classmethod
    def execute(cls, text: str, dictionary: str) -> io.NodeOutput:
        report = apply_pronunciation(text, parse_pronunciation_rules(dictionary))
        return io.NodeOutput(
            report.text, json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
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


class T8DotsTTSEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Edit",
            display_name="dots.tts 语音编辑 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["speech edit", "replace speech", "语音编辑"],
            description=(
                "按官方标签指令编辑局部语音，支持替换、插入、删除、情绪、"
                "音高、语速、停顿、增强和背景音操作。"
            ),
            inputs=[
                EditModelType.Input("model", display_name="dots.tts Edit 模型"),
                io.Audio.Input("source_audio", display_name="原始音频"),
                io.String.Input(
                    "instruction",
                    display_name="编辑指令",
                    multiline=True,
                    default='你好，<sub targ="欢迎">很高兴</sub>使用 dots.tts。',
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "source_text",
                    display_name="原始转写（可选）",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                io.String.Input(
                    "target_text",
                    display_name="目标转写（可选）",
                    multiline=True,
                    default="",
                    optional=True,
                ),
                io.Combo.Input(
                    "use_xvector",
                    display_name="原说话人引导",
                    options=["auto", "on", "off"],
                    default="auto",
                ),
                io.Int.Input(
                    "seed",
                    display_name="seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "speaker_scale",
                    display_name="说话人引导强度",
                    default=1.5,
                    min=0.0,
                    max=5.0,
                    step=0.05,
                ),
                io.Int.Input(
                    "num_steps",
                    display_name="采样步数",
                    default=10,
                    min=1,
                    max=64,
                ),
                io.Float.Input(
                    "guidance_scale",
                    display_name="CFG",
                    default=1.2,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                ),
                SamplingType.Input(
                    "sampling",
                    display_name="采样配置（可选，连接后覆盖上方参数）",
                    optional=True,
                ),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="编辑后音频"),
                io.String.Output("target_text", display_name="目标转写"),
                io.String.Output("edit_info", display_name="编辑信息 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        instruction: str,
        use_xvector: str,
        speaker_scale: float,
        num_steps: int,
        guidance_scale: float,
        **kwargs,
    ) -> bool | str:
        if not str(instruction).strip():
            return "编辑指令不能为空。"
        try:
            from dots_tts.data.edit_instruction import (
                render_source_text,
                render_target_text,
            )

            if not render_source_text(instruction).strip():
                return "编辑指令推导出的原始转写不能为空。"
            if not render_target_text(instruction).strip():
                return "编辑指令推导出的目标转写不能为空。"
        except Exception as exc:
            return f"编辑指令格式无效：{exc}"
        if use_xvector not in {"auto", "on", "off"}:
            return "原说话人引导必须为 auto、on 或 off。"
        if not 0.0 <= float(speaker_scale) <= 5.0:
            return "说话人引导强度必须在 0–5。"
        if not 1 <= int(num_steps) <= 64:
            return "采样步数必须在 1–64。"
        if not 0.0 <= float(guidance_scale) <= 10.0:
            return "CFG 必须在 0–10。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        source_audio: dict,
        instruction: str,
        source_text: str,
        target_text: str,
        use_xvector: str,
        seed: int,
        speaker_scale: float,
        num_steps: int,
        guidance_scale: float,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        selected = sampling or SamplingConfig(
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            speaker_scale=float(speaker_scale),
        )
        audio, info = run_edit_inference(
            handle=model,
            source_audio=source_audio,
            instruction=instruction,
            source_text=source_text,
            target_text=target_text,
            use_xvector=use_xvector,
            seed=int(seed),
            speaker_scale=float(selected.speaker_scale),
            num_steps=int(selected.num_steps),
            guidance_scale=float(selected.guidance_scale),
            ode_method=selected.ode_method,
            max_generate_length=int(selected.max_generate_length),
        )
        return io.NodeOutput(
            audio,
            str(info.get("target_text") or ""),
            json.dumps(info, ensure_ascii=False, indent=2),
        )


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


class T8DotsTTSProject(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Project",
            display_name="dots.tts 字幕与多角色工程 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["SRT", "subtitle", "多角色", "resume", "retry"],
            description=(
                "渲染 TXT/SRT/VTT 工程；逐段重试并持久化 checkpoint，"
                "支持时间轴、多角色和发音词典。"
            ),
            inputs=[
                ModelType.Input("model", display_name="dots.tts 模型"),
                io.String.Input(
                    "source_text",
                    display_name="TXT / SRT / VTT 内容",
                    multiline=True,
                    default=(
                        "1\n00:00:00,000 --> 00:00:02,000\n[旁白] 2026 年，T8 发布。"
                    ),
                    dynamic_prompts=True,
                ),
                io.String.Input(
                    "project_name", display_name="工程名", default="subtitle.srt"
                ),
                io.Combo.Input(
                    "project_kind",
                    display_name="工程类型",
                    options=["auto", "subtitle", "text"],
                    default="auto",
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
                io.String.Input(
                    "pronunciation",
                    display_name="发音词典",
                    multiline=True,
                    default="2026 => 二零二六\nT8 => T 八",
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
                    display_name="文本段间停顿（毫秒）",
                    default=200,
                    min=0,
                    max=10_000,
                ),
                io.Int.Input(
                    "max_retries",
                    display_name="单段重试次数",
                    default=2,
                    min=0,
                    max=10,
                ),
                io.Boolean.Input("resume", display_name="断点续做", default=True),
                io.Combo.Input(
                    "subtitle_fit_mode",
                    display_name="字幕时长适配",
                    options=["shrink", "fit", "none", "trim"],
                    default="shrink",
                ),
                io.String.Input(
                    "audio_formats",
                    display_name="导出格式",
                    default="wav",
                    tooltip="逗号分隔：wav,mp3,flac",
                ),
                io.Float.Input(
                    "target_lufs",
                    display_name="目标 LUFS（0=关闭）",
                    default=0.0,
                    min=-36.0,
                    max=0.0,
                    step=0.5,
                ),
                io.String.Input(
                    "video_path",
                    display_name="可选视频路径",
                    default="",
                    advanced=True,
                ),
                io.String.Input(
                    "ffmpeg_path",
                    display_name="FFmpeg 路径（留空自动）",
                    default="",
                    advanced=True,
                ),
                io.String.Input(
                    "output_dir",
                    display_name="工程输出目录（留空自动）",
                    default="",
                    advanced=True,
                ),
                SpeakerMapType.Input(
                    "speaker_map", display_name="多角色音色映射", optional=True
                ),
                SamplingType.Input("sampling", display_name="采样配置", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="工程音频"),
                io.String.Output("project_info", display_name="时间轴与报告 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        source_text: str,
        project_name: str,
        project_kind: str,
        pronunciation: str,
        max_segment_tokens: int,
        segment_gap_ms: int,
        max_retries: int,
        subtitle_fit_mode: str = "shrink",
        audio_formats: str = "wav",
        target_lufs: float = 0.0,
        **kwargs,
    ) -> bool | str:
        if not str(source_text).strip():
            return "工程内容不能为空。"
        if int(max_segment_tokens) <= 0:
            return "每段最大 Tokens 必须大于 0。"
        if int(segment_gap_ms) < 0 or int(max_retries) < 0:
            return "段间停顿和重试次数不能为负数。"
        try:
            parse_audio_formats(audio_formats)
            if subtitle_fit_mode not in {"none", "shrink", "fit", "trim"}:
                return "字幕时长适配模式无效。"
            if float(target_lufs) != 0.0 and not -36.0 <= float(target_lufs) <= -5.0:
                return "目标 LUFS 必须为 0（关闭）或 -36 至 -5。"
            build_production_plan(
                project_name or "project.txt",
                source_text,
                kind=project_kind,
                pronunciation=pronunciation,
            )
            return True
        except Exception as exc:
            return f"工程内容无效：{exc}"

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        source_text: str,
        project_name: str,
        project_kind: str,
        language: str,
        seed: int,
        pronunciation: str,
        max_segment_tokens: int,
        segment_gap_ms: int,
        max_retries: int,
        resume: bool,
        output_dir: str,
        subtitle_fit_mode: str = "shrink",
        audio_formats: str = "wav",
        target_lufs: float = 0.0,
        video_path: str = "",
        ffmpeg_path: str = "",
        speaker_map: SpeakerVoiceMap | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        resolved_video_path = (
            str(_input_path(video_path.strip())) if video_path.strip() else None
        )
        audio, info = run_project_inference(
            handle=model,
            source_name=project_name.strip() or "project.txt",
            source_text=source_text,
            project_kind=project_kind,
            language=language,
            seed=int(seed),
            speaker_map=speaker_map,
            sampling=sampling,
            pronunciation=pronunciation,
            max_segment_tokens=int(max_segment_tokens),
            segment_gap_ms=int(segment_gap_ms),
            max_retries=int(max_retries),
            resume=bool(resume),
            output_root=_project_output_root(output_dir),
            subtitle_fit_mode=subtitle_fit_mode,
            audio_formats=audio_formats,
            target_lufs=float(target_lufs) if float(target_lufs) <= -5.0 else None,
            video_path=resolved_video_path,
            ffmpeg_path=ffmpeg_path.strip() or None,
        )
        return io.NodeOutput(audio, json.dumps(info, ensure_ascii=False, indent=2))


class T8DotsTTSBatchQueue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_BatchQueue",
            display_name="dots.tts 批量任务队列 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["batch", "folder", "批处理", "任务队列"],
            description=(
                "递归处理本地 TXT/SRT/VTT 文件或文件夹；输出首个工程预览、"
                "批量 ZIP 和可恢复任务报告。"
            ),
            inputs=[
                ModelType.Input("model", display_name="dots.tts 模型"),
                io.String.Input(
                    "input_paths",
                    display_name="文件/文件夹路径（每行一个）",
                    multiline=True,
                    default="D:/dubbing-project/subtitles",
                ),
                io.Combo.Input(
                    "language",
                    display_name="语言",
                    options=LANGUAGES,
                    default="inherit",
                ),
                io.Int.Input(
                    "seed", display_name="起始 seed", default=42, min=0, max=0xFFFFFFFF
                ),
                io.String.Input(
                    "pronunciation",
                    display_name="发音词典",
                    multiline=True,
                    default="",
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
                    display_name="文本段间停顿（毫秒）",
                    default=200,
                    min=0,
                    max=10_000,
                ),
                io.Int.Input(
                    "max_retries",
                    display_name="单段重试次数",
                    default=2,
                    min=0,
                    max=10,
                ),
                io.Boolean.Input("resume", display_name="断点续做", default=True),
                io.Boolean.Input(
                    "retry_failed_only", display_name="只重跑失败项", default=False
                ),
                io.Combo.Input(
                    "subtitle_fit_mode",
                    display_name="字幕时长适配",
                    options=["shrink", "fit", "none", "trim"],
                    default="shrink",
                ),
                io.String.Input(
                    "audio_formats",
                    display_name="导出格式",
                    default="wav",
                    tooltip="逗号分隔：wav,mp3,flac",
                ),
                io.Float.Input(
                    "target_lufs",
                    display_name="目标 LUFS（0=关闭）",
                    default=0.0,
                    min=-36.0,
                    max=0.0,
                    step=0.5,
                ),
                io.String.Input(
                    "video_path",
                    display_name="单工程可选视频路径",
                    default="",
                    advanced=True,
                ),
                io.String.Input(
                    "ffmpeg_path",
                    display_name="FFmpeg 路径（留空自动）",
                    default="",
                    advanced=True,
                ),
                io.String.Input(
                    "output_dir",
                    display_name="工程输出目录（留空自动）",
                    default="",
                    advanced=True,
                ),
                SpeakerMapType.Input(
                    "speaker_map", display_name="多角色音色映射", optional=True
                ),
                SamplingType.Input("sampling", display_name="采样配置", optional=True),
            ],
            outputs=[
                io.Audio.Output("preview_audio", display_name="首个工程预览"),
                io.String.Output("archive_path", display_name="批量 ZIP 路径"),
                io.String.Output("batch_info", display_name="队列报告 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        input_paths: str,
        max_segment_tokens: int,
        segment_gap_ms: int,
        max_retries: int,
        subtitle_fit_mode: str = "shrink",
        audio_formats: str = "wav",
        target_lufs: float = 0.0,
        **kwargs,
    ) -> bool | str:
        if not any(line.strip() for line in str(input_paths).splitlines()):
            return "至少填写一个文件或文件夹路径。"
        if int(max_segment_tokens) <= 0:
            return "每段最大 Tokens 必须大于 0。"
        if int(segment_gap_ms) < 0 or int(max_retries) < 0:
            return "段间停顿和重试次数不能为负数。"
        try:
            parse_audio_formats(audio_formats)
        except Exception as exc:
            return f"导出格式无效：{exc}"
        if subtitle_fit_mode not in {"none", "shrink", "fit", "trim"}:
            return "字幕时长适配模式无效。"
        if float(target_lufs) != 0.0 and not -36.0 <= float(target_lufs) <= -5.0:
            return "目标 LUFS 必须为 0（关闭）或 -36 至 -5。"
        return True

    @classmethod
    def execute(
        cls,
        model: ModelHandle,
        input_paths: str,
        language: str,
        seed: int,
        pronunciation: str,
        max_segment_tokens: int,
        segment_gap_ms: int,
        max_retries: int,
        resume: bool,
        output_dir: str,
        retry_failed_only: bool = False,
        subtitle_fit_mode: str = "shrink",
        audio_formats: str = "wav",
        target_lufs: float = 0.0,
        video_path: str = "",
        ffmpeg_path: str = "",
        speaker_map: SpeakerVoiceMap | None = None,
        sampling: SamplingConfig | None = None,
    ) -> io.NodeOutput:
        paths = [
            str(_input_path(line.strip()))
            for line in input_paths.splitlines()
            if line.strip()
        ]
        resolved_video_path = (
            str(_input_path(video_path.strip())) if video_path.strip() else None
        )
        audio, info = run_batch_inference(
            handle=model,
            input_paths=paths,
            language=language,
            seed=int(seed),
            speaker_map=speaker_map,
            sampling=sampling,
            pronunciation=pronunciation,
            max_segment_tokens=int(max_segment_tokens),
            segment_gap_ms=int(segment_gap_ms),
            max_retries=int(max_retries),
            resume=bool(resume),
            output_root=_project_output_root(output_dir),
            retry_failed_only=bool(retry_failed_only),
            subtitle_fit_mode=subtitle_fit_mode,
            audio_formats=audio_formats,
            target_lufs=float(target_lufs) if float(target_lufs) <= -5.0 else None,
            video_path=resolved_video_path,
            ffmpeg_path=ffmpeg_path.strip() or None,
        )
        return io.NodeOutput(
            audio,
            str(info["archive_path"]),
            json.dumps(info, ensure_ascii=False, indent=2),
        )


class T8DotsTTSTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Transcribe",
            display_name="dots.tts ASR 转写与说话人分离 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            search_aliases=["ASR", "transcribe", "diarization", "自动字幕"],
            description=(
                "使用固定版本 Qwen3-ASR + ForcedAligner 生成时间戳；"
                "可连接 dots.tts 模型，用 CAM++ 进行本地说话人分离。"
            ),
            inputs=[
                io.Audio.Input("audio", display_name="待转写音频"),
                io.Combo.Input(
                    "language",
                    display_name="识别语言",
                    options=[
                        "auto",
                        "Chinese",
                        "English",
                        "Cantonese",
                        "Japanese",
                        "Korean",
                    ],
                    default="auto",
                ),
                io.Boolean.Input("diarize", display_name="说话人分离", default=False),
                io.Int.Input(
                    "speaker_count",
                    display_name="说话人数（0=自动）",
                    default=0,
                    min=0,
                    max=32,
                ),
                io.Float.Input(
                    "similarity_threshold",
                    display_name="同一说话人相似度阈值",
                    default=0.72,
                    min=0.4,
                    max=0.95,
                    step=0.01,
                ),
                io.Combo.Input(
                    "asr_model",
                    display_name="固定 ASR 模型",
                    options=["Qwen/Qwen3-ASR-0.6B-hf"],
                    default="Qwen/Qwen3-ASR-0.6B-hf",
                    advanced=True,
                ),
                io.Combo.Input(
                    "aligner_model",
                    display_name="固定 ForcedAligner 模型",
                    options=["Qwen/Qwen3-ForcedAligner-0.6B-hf"],
                    default="Qwen/Qwen3-ForcedAligner-0.6B-hf",
                    advanced=True,
                ),
                io.Combo.Input(
                    "transport",
                    display_name="ASR 隔离方式",
                    options=["auto", "sidecar", "in_process"],
                    default="auto",
                    tooltip="auto 优先连接桌面隔离服务；不可用时兼容进程内 ASR。",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "release_after_run",
                    display_name="转写后释放 ASR 显存",
                    default=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "device",
                    display_name="ASR 设备",
                    options=["cuda:0", "cpu"],
                    default="cuda:0",
                    advanced=True,
                ),
                ModelType.Input(
                    "speaker_model",
                    display_name="说话人编码模型",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output("srt", display_name="SRT 字幕"),
                io.String.Output("timeline_json", display_name="时间轴 JSON"),
                io.String.Output("asr_info", display_name="ASR 报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(
        cls,
        speaker_count: int,
        similarity_threshold: float,
        diarize: bool,
        speaker_model: ModelHandle | None = None,
        transport: str = "auto",
        asr_model: str = "Qwen/Qwen3-ASR-0.6B-hf",
        aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        **kwargs,
    ) -> bool | str:
        if not 0 <= int(speaker_count) <= 32:
            return "说话人数必须为 0（自动）或 1–32。"
        if not 0.4 <= float(similarity_threshold) <= 0.95:
            return "说话人相似度阈值必须在 0.4–0.95。"
        if diarize and speaker_model is None:
            return "启用说话人分离时必须连接 dots.tts 模型。"
        if transport not in {"auto", "sidecar", "in_process", "in_process_legacy"}:
            return "ASR 隔离方式无效。"
        if asr_model != "Qwen/Qwen3-ASR-0.6B-hf":
            return "ASR 模型必须使用固定清单版本。"
        if aligner_model != "Qwen/Qwen3-ForcedAligner-0.6B-hf":
            return "ForcedAligner 必须使用固定清单版本。"
        return True

    @classmethod
    def execute(
        cls,
        audio: dict,
        language: str,
        diarize: bool,
        speaker_count: int,
        similarity_threshold: float,
        asr_model: str,
        aligner_model: str,
        transport: str,
        release_after_run: bool,
        device: str,
        speaker_model: ModelHandle | None = None,
    ) -> io.NodeOutput:
        report = run_transcription(
            audio,
            asr_model=asr_model,
            aligner_model=aligner_model,
            language=None if language == "auto" else language,
            diarize=bool(diarize),
            speaker_count=int(speaker_count),
            similarity_threshold=float(similarity_threshold),
            device=device,
            speaker_model=speaker_model,
            transport=transport,
            release_after_run=bool(release_after_run),
        )
        timeline_json = json.dumps(
            {
                "schema_version": 1,
                "kind": "dots-tts-timeline",
                "cues": report["segments"],
            },
            ensure_ascii=False,
            indent=2,
        )
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        return io.NodeOutput(
            report["srt"],
            timeline_json,
            report_json,
            ui=ui.PreviewText(report["srt"]),
        )


class T8DotsTTSTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Timeline",
            display_name="dots.tts 字幕时间轴标准化 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["subtitle", "timeline", "SRT", "VTT"],
            description="校验并标准化 SRT、WebVTT 或 dots.tts 时间轴 JSON。",
            inputs=[
                io.String.Input(
                    "timeline",
                    display_name="字幕 / 时间轴",
                    multiline=True,
                    default="",
                ),
                io.Combo.Input(
                    "input_format",
                    display_name="输入格式",
                    options=["srt", "vtt", "json"],
                    default="srt",
                ),
            ],
            outputs=[
                io.String.Output("srt", display_name="标准 SRT"),
                io.String.Output("timeline_json", display_name="标准时间轴 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, timeline: str, input_format: str) -> io.NodeOutput:
        if input_format == "json":
            value = json.loads(timeline)
            raw_cues = value.get("cues", value) if isinstance(value, dict) else value
            cues = rows_to_cues(raw_cues)
        else:
            cues = parse_subtitles(timeline)
        srt = timeline_document(cues, "srt")
        return io.NodeOutput(
            srt,
            timeline_document(cues, "json"),
            ui=ui.PreviewText(srt),
        )


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


class T8DotsTTSDiagnostics(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_DotsTTS_Diagnostics",
            display_name="dots.tts 导出诊断包 · T8star-Aix",
            category=CATEGORY,
            description=("导出不含环境变量、令牌、音频、工作流和模型权重的支持 ZIP。"),
            inputs=[
                io.String.Input(
                    "output_dir",
                    display_name="输出目录（留空使用 ComfyUI output）",
                    default="",
                    optional=True,
                    advanced=True,
                )
            ],
            outputs=[
                io.String.Output("bundle_path", display_name="诊断包路径"),
                io.String.Output("report", display_name="诊断报告 JSON"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, output_dir: str = "") -> io.NodeOutput:
        root = (
            _project_output_root(output_dir)
            if output_dir.strip()
            else _project_output_root("").parent / "dots_tts_t8_diagnostics"
        )
        destination = root / f"dots-tts-T8-diagnostics-{int(time.time())}.zip"
        path, report = write_diagnostic_bundle(
            destination,
            app="comfyui-dots-tts-T8",
            version=__version__,
            runtime=runtime_info(),
        )
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        return io.NodeOutput(str(path), report_json, ui=ui.PreviewText(report_json))


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
            is_output_node=True,
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
            T8DotsTTSEditModelLoader,
            T8DotsTTSVoicePrompt,
            T8DotsTTSSampling,
            T8DotsTTSSpeakerMap,
            T8DotsTTSPronunciation,
            T8DotsTTSGenerate,
            T8DotsTTSEdit,
            T8DotsTTSLongText,
            T8DotsTTSProject,
            T8DotsTTSBatchQueue,
            T8DotsTTSTranscribe,
            T8DotsTTSTimeline,
            T8DotsTTSEnvironment,
            T8DotsTTSDiagnostics,
            T8DotsTTSRelease,
        ]


async def comfy_entrypoint() -> T8DotsTTSExtension:
    return T8DotsTTSExtension()


__all__ = [
    "T8DotsTTSModelLoader",
    "T8DotsTTSEditModelLoader",
    "T8DotsTTSVoicePrompt",
    "T8DotsTTSSampling",
    "T8DotsTTSSpeakerMap",
    "T8DotsTTSPronunciation",
    "T8DotsTTSGenerate",
    "T8DotsTTSEdit",
    "T8DotsTTSLongText",
    "T8DotsTTSProject",
    "T8DotsTTSBatchQueue",
    "T8DotsTTSTranscribe",
    "T8DotsTTSTimeline",
    "T8DotsTTSEnvironment",
    "T8DotsTTSDiagnostics",
    "T8DotsTTSRelease",
    "T8DotsTTSExtension",
    "comfy_entrypoint",
]
