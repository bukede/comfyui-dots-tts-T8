# 示例工作流

- `api/01_soar_voice_clone.json`：SOAR 参考音频克隆。
- `api/02_mf_text_only.json`：MF-2steps 无参考音频快速合成。
- `api/03_soar_long_text.json`：SOAR 长文本分段、显式停顿与音色克隆。
- `api/04_subtitle_project.json`：SRT 时间轴、角色音色、发音词典与断点续做。
- `api/05_batch_queue.json`：递归处理 TXT/SRT/VTT 文件与文件夹。
- `api/06_asr_transcribe.json`：自动转写、时间戳与可选说话人分离。
- `api/07_timeline_normalize.json`：SRT/WebVTT/JSON 时间轴标准化。
- `api/08_speech_edit.json`：用 dots.tts.edit 替换原音频中的局部内容。
- `voice_library.example.json`：六角色本地参考音频库；可继续增加角色。
- `ui/01_soar_voice_clone.json`：可拖入 ComfyUI 画布的基础克隆流程。
- `ui/02_soar_long_text.json`：可拖入画布的长文本音色克隆流程。
- `ui/03_subtitle_project.json`：可拖入画布的字幕时间轴和角色音色流程。
- `ui/04_asr_transcribe.json`：可拖入画布的自动字幕与说话人分离流程。
- `ui/05_timeline_normalize.json`：可拖入画布的字幕格式标准化流程。

模型加载器默认按固定清单从 `ComfyUI/models/TTS/dots.tts/` 查找模型。
克隆示例使用 `voice_reference.wav`，请先把已获授权的参考录音放进 ComfyUI
`input` 目录，并将转写文本改成与录音逐字一致。

长文本节点按模型 tokenizer 自动切分，默认段间停顿 200ms。文本中可使用
`<pause=800ms>` 或 `<pause=1.2s>` 覆盖相邻段落之间的自动停顿。

字幕支持 `[旁白] 文本`、`旁白：文本` 和 WebVTT `<v 旁白>文本` 角色标记。
工程节点每成功一段就更新 `checkpoint.json`，再次执行且“断点续做”开启时只处理
失败或未完成段。批量节点还会导出包含 master WAV、分段 WAV、时间轴和 checkpoint
的 ZIP。

v0.4 默认用 `shrink` 对超出字幕槽位的语音做保音高变速，并可导出 MP3/FLAC、
标准化 LUFS 或回封视频。媒体导出需 FFmpeg；纯 WAV 不需要。批量示例把
`retry_failed_only` 设为 `true` 后，只会重跑原 manifest 中失败的项目，并校验配置
指纹。

当前 ASR 示例使用 Transformers 5 原生实现，需要手动安装
`transformers>=5.5,<6`。首次运行会下载约 3.4 GB 固定 revision 权重并校验大小与
SHA-256；启用说话人分离时还需连接 dots.tts 模型加载器。

v0.6 的语音编辑示例需要把官方 `dots.tts.edit` 放到
`ComfyUI/models/TTS/dots.tts/dots.tts.edit/`。指令支持 `<del>`、`<ins>`、
`<sub targ="...">`、`<emo>`、`<pitch>`、`<rate>`、`<pause/>`、`<enhance>`、
`<bg>` 与 `<spk_transfer/>`；原始/目标转写可留空并由结构化指令推导。
