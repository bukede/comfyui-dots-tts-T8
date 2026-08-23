# comfyui-dots-tts-T8

dots.tts 的 ComfyUI V3 原生节点，作者与发布者：T8star-Aix。

节点菜单：`T8star-Aix / Audio / dots.tts`

## 首版节点

- dots.tts 模型加载器
- dots.tts 音色提示
- dots.tts 采样配置
- dots.tts 语音生成
- dots.tts 环境诊断
- dots.tts 释放模型

支持官方 SOAR（质量优先）和 MF-2steps（速度优先）模型。生成节点输出标准
ComfyUI `AUDIO`，采样率为 48 kHz。

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

## 许可证

节点与内置的 dots.tts v0.3.1 推理源码遵循 Apache-2.0。模型权重同样为
Apache-2.0。请只使用已获授权的参考音频，并明确标注 AI 生成语音。
