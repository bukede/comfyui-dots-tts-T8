# comfyui-dots-tts-T8

dots.tts 的 ComfyUI V3 原生节点，作者与发布者：T8star-Aix。

节点菜单：`T8star-Aix / Audio / dots.tts`

## 节点

- dots.tts 模型加载器
- dots.tts Edit 模型加载器
- dots.tts 音色提示
- dots.tts 采样配置
- dots.tts 多角色音色映射（A-D 音色槽 + 无限角色 JSON 音色库 + 默认回退）
- dots.tts 发音词典（数字、人名与专有词覆盖）
- dots.tts 语音生成
- dots.tts 语音编辑（替换、插入、删除、情绪、音高、语速、停顿、增强与背景）
- dots.tts 长文本生成（智能分段、音频拼接、显式停顿）
- dots.tts 字幕与多角色工程（时间轴适配、媒体导出、重试、断点续做）
- dots.tts 批量任务队列（递归处理、失败项重跑与 ZIP 导出）
- dots.tts ASR 转写与说话人分离（Qwen3-ASR + ForcedAligner + 可选 CAM++）
- dots.tts 字幕时间轴标准化（SRT / WebVTT / JSON）
- dots.tts 环境诊断
- dots.tts 诊断包导出
- dots.tts 释放模型

支持官方 SOAR（质量优先）和 MF-2steps（速度优先）模型。生成节点输出标准
ComfyUI `AUDIO`，采样率为 48 kHz。

长文本节点使用当前模型 tokenizer 按 token 数切分，支持中文/英文标点边界，
并可使用 `<pause=800ms>` 或 `<pause=1.2s>` 插入最长 60 秒的显式停顿。各段使用
从起始 seed 确定性派生的 seed，输出 JSON 会记录完整分段与拼接信息。

## 字幕、多角色与批量工程

字幕角色可写成 `[旁白] 文本`、`旁白：文本` 或 WebVTT `<v 旁白>文本`。
“多角色音色映射”节点把角色映射到 A-D 音色槽；没有匹配的角色会使用默认音色。
`voice_library` 还可填写音色库 JSON 或 JSON 文件路径，每个角色使用
`audio`、`transcript`、`language` 描述，因此角色数量不再受四个连接槽限制。示例见
`example_workflows/voice_library.example.json`。相对音频路径以音色库文件目录为准。

发音词典支持简洁行格式：

```text
2026 => 二零二六
张三 => 章三
T8 => T 八
```

也支持 JSON 对象或带 `mode`、`case_sensitive`、`priority` 的规则数组。工程节点把
每个字幕条目或文本段独立保存到 `segments/`，每次尝试后原子更新
`checkpoint.json`。开启“断点续做”后，已成功且音频仍有效的段不会再次推理。

字幕工程按 SRT/VTT 起始时间混音。默认 `shrink` 会对超出槽位的语音做保音高变速；
也可选择全部对齐、保持原时长或明确裁剪。`timeline.json` 会记录原时长、输出时长、
变速率和 `overflow_ms`。工程生成 `master.wav` 与标准化 `timeline.srt`。批量任务队列递归发现
`.txt`、`.srt`、`.vtt`，逐工程隔离失败，并导出包含音频、时间轴和 checkpoint 的
ZIP。“只重跑失败项”会读取原 manifest，并要求配置指纹完全一致。

媒体导出支持 WAV、MP3、FLAC、两遍 EBU R128 LUFS 标准化和视频音轨回封。纯 WAV
不需要 FFmpeg；其他功能要求整合包内置 FFmpeg，或系统 `PATH` 中存在 FFmpeg。
工作流不能启动任意可执行文件，节点也不会下载或安装 FFmpeg。

## ASR 与字幕时间轴

“ASR 转写与说话人分离”节点把 ComfyUI `AUDIO` 转为带时间戳的 SRT、标准时间轴
JSON 和完整识别报告。默认使用 Transformers 5 原生、固定 revision 的
`Qwen/Qwen3-ASR-0.6B-hf` 与 `Qwen/Qwen3-ForcedAligner-0.6B-hf`，首次使用会下载
约 3.4 GB 权重并逐文件校验尺寸和
SHA-256。启用说话人分离时，需要连接任意已发现的 dots.tts 模型；节点只读取其中的
CAM++ `speaker_encoder.safetensors`，不会为分离加载完整 TTS 推理模型。

默认 `auto` 会优先连接 T8 Windows 整合包自动启动的本机 ASR sidecar。sidecar 使用
随机令牌、只绑定 loopback，并在独立 Python 进程中加载 Qwen3-ASR；转写后默认释放
显存。这样 ComfyUI 主环境无需调整 Transformers。

ASR 是可选能力。为避免 ComfyUI Manager 自动升级核心 `transformers`，它不在
`requirements.txt` 中。请先备份 ComfyUI 环境，再使用启动 ComfyUI 的同一个 Python
手动安装安全版本的原生组件：

```powershell
python -m pip install "transformers>=5.5,<6"
```

若其他节点不兼容该 Transformers 版本，可保留桌面整合包中的 ASR，ComfyUI 侧只用
TTS 节点。没有桌面 sidecar 时，可明确选择 `in_process` 并按上面的安全版本
手动安装。“字幕时间轴标准化”节点可校验 SRT、WebVTT 或本项目 JSON，重新输出统一
SRT/JSON；两个节点都可作为队列终点并直接预览字幕文本。

## 通过 ComfyUI Manager 安装（推荐）

在 ComfyUI Manager 中搜索 `dots-tts-t8` 或 `dots.tts · T8star-Aix`，点击安装并
重启 ComfyUI。节点由 Publisher ID `t8star` 通过官方 Comfy Registry 发布：

- GitHub：<https://github.com/T8mars/comfyui-dots-tts-T8>
- Registry：<https://registry.comfy.org/nodes/dots-tts-t8>

Manager 会依据 `requirements.txt` 安装轻量依赖。遵循 ComfyUI Registry 安全规范，
节点本身不会调用 pip，也不会在导入或推理时安装软件。

## 手动安装

把仓库克隆到 `ComfyUI/custom_nodes/`，再使用启动 ComfyUI 的同一个 Python 安装
依赖：

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/T8mars/comfyui-dots-tts-T8.git
python -m pip install -r comfyui-dots-tts-T8\requirements.txt
```

依赖清单不会安装或升级 `torch`、`torchaudio`、`torchvision`、`transformers`、
`pydantic` 或 `numpy`，避免覆盖 ComfyUI 的核心环境。缺少这些包时请修复 ComfyUI
自身环境。

本节点验证范围为 `transformers>=5.5,<6`。若环境诊断报告版本过低，请先备份
ComfyUI 环境并自行评估其他节点兼容性，再用同一个 Python 执行：

```powershell
python -m pip install "transformers>=5.5,<6"
```

## 模型位置

推荐目录：

```text
ComfyUI/models/TTS/dots.tts/dots.tts-soar/
ComfyUI/models/TTS/dots.tts/dots.tts-mf-2steps/
ComfyUI/models/TTS/dots.tts/dots.tts.edit/
```

也支持 `extra_model_paths.yaml` 中登记的 `TTS` 路径，以及已有社区节点使用的
`ComfyUI/models/dotstts/`。节点导入时不会联网或自动下载模型。

下载 SOAR：

```powershell
python scripts/download_models.py --model soar --comfyui-root D:\ComfyUI
```

下载 MF 2-Steps：

```powershell
python scripts/download_models.py --model mf-2steps --comfyui-root D:\ComfyUI
```

下载 dots.tts Edit：

```powershell
python scripts/download_models.py --model edit --comfyui-root D:\ComfyUI
```

下载器固定官方仓库 revision，支持 Hugging Face/Xet 缓存，并在完成后校验文件尺寸
和 SHA-256。节点导入时不会自动联网。

`example_workflows/api/` 可直接用于 ComfyUI API，`example_workflows/ui/` 可拖入
ComfyUI 画布。基础克隆示例中的 `voice_reference.wav` 需要先放到 ComfyUI input
目录，并把参考音频转写改成与录音逐字一致。

Windows 不自动安装 WeTextProcessing：其依赖 pynini 没有受支持的 Windows wheel。
节点默认关闭“文本归一化”，普通合成和音色克隆不受影响；请在输入前自行展开数字、
单位等容易歧义的写法。

## 兼容边界

dots.tts v0.3.1 官方支持 Python 3.10-3.12。本版明确不支持 Python 3.13；请使用
Python 3.10-3.12 的 ComfyUI 环境。

当前版本要求 PyTorch/Torchaudio 2.8 或更新版本，并要求二者 minor 版本一致。实际
推理以 NVIDIA CUDA 为主要支持目标；CPU 回退速度不作为发布验收目标。

## 发布验收

- 官方 ComfyUI V3 API 节点注册与 schema 测试通过。
- SOAR 与 MF 2-Steps 的 text-only、continuation clone、x-vector clone 真实推理通过。
- SOAR 与 MF 2-Steps 权重均通过完整尺寸和 SHA-256 校验。
- 输出为标准 ComfyUI `AUDIO`：`[batch, channel, samples]`，48 kHz，无 NaN/Inf。
- 长文本节点会按模型 token 数切分、复用同一运行时，并精确保留显式停顿。
- 字幕工程的角色映射、发音覆盖、时间轴混音、逐段重试和断点续做测试通过。
- 批量任务队列可递归发现 TXT/SRT/VTT，隔离单工程失败并生成 ZIP。
- 六角色文件音色库、字幕保音高时长适配、失败项确定性重跑通过。
- 真实 FFmpeg 两遍 LUFS、MP3/FLAC 与 MP4 音轨回封通过。
- Qwen3-ASR + ForcedAligner 真实 CUDA 转写、时间戳和 CAM++ 说话人分离通过。
- ASR/时间轴节点可作为 V3 输出节点执行并返回标准 SRT/JSON。

## 许可证

节点与内置的 dots.tts v0.3.1 推理源码遵循 Apache-2.0。模型权重同样为
Apache-2.0。请只使用已获授权的参考音频，并明确标注 AI 生成语音。
