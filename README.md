# comfyui-dots-tts-T8

dots.tts 的 ComfyUI V3 原生节点，作者与发布者：T8star-Aix。

节点菜单：`T8star-Aix / Audio / dots.tts`

## 节点

- dots.tts 模型加载器
- dots.tts 音色提示
- dots.tts 采样配置
- dots.tts 多角色音色映射（四个音色槽 + 默认回退）
- dots.tts 发音词典（数字、人名与专有词覆盖）
- dots.tts 语音生成
- dots.tts 长文本生成（智能分段、音频拼接、显式停顿）
- dots.tts 字幕与多角色工程（TXT/SRT/VTT、时间轴、重试、断点续做）
- dots.tts 批量任务队列（文件/文件夹递归处理与 ZIP 导出）
- dots.tts 环境诊断
- dots.tts 释放模型

支持官方 SOAR（质量优先）和 MF-2steps（速度优先）模型。生成节点输出标准
ComfyUI `AUDIO`，采样率为 48 kHz。

长文本节点使用当前模型 tokenizer 按 token 数切分，支持中文/英文标点边界，
并可使用 `<pause=800ms>` 或 `<pause=1.2s>` 插入最长 60 秒的显式停顿。各段使用
从起始 seed 确定性派生的 seed，输出 JSON 会记录完整分段与拼接信息。

## 字幕、多角色与批量工程

字幕角色可写成 `[旁白] 文本`、`旁白：文本` 或 WebVTT `<v 旁白>文本`。
“多角色音色映射”节点把角色映射到 A-D 音色槽；没有匹配的角色会使用默认音色。

发音词典支持简洁行格式：

```text
2026 => 二零二六
张三 => 章三
T8 => T 八
```

也支持 JSON 对象或带 `mode`、`case_sensitive`、`priority` 的规则数组。工程节点把
每个字幕条目或文本段独立保存到 `segments/`，每次尝试后原子更新
`checkpoint.json`。开启“断点续做”后，已成功且音频仍有效的段不会再次推理。

字幕工程按 SRT/VTT 起始时间混音，生成 `master.wav`、`timeline.json` 与标准化
`timeline.srt`；超出字幕
时长的语音不会被静默截断，报告会写入 `overflow_ms`。批量任务队列递归发现
`.txt`、`.srt`、`.vtt`，逐工程隔离失败，并导出包含音频、时间轴和 checkpoint 的
ZIP。

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

dots.tts v0.3.1 要求 `transformers>=4.57,<5`。若环境诊断报告版本过低，请先备份
ComfyUI 环境并自行评估其他节点兼容性，再用同一个 Python 执行：

```powershell
python -m pip install "transformers>=4.57,<5"
```

## 模型位置

推荐目录：

```text
ComfyUI/models/TTS/dots.tts/dots.tts-soar/
ComfyUI/models/TTS/dots.tts/dots.tts-mf-2steps/
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

## 许可证

节点与内置的 dots.tts v0.3.1 推理源码遵循 Apache-2.0。模型权重同样为
Apache-2.0。请只使用已获授权的参考音频，并明确标注 AI 生成语音。
