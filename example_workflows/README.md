# 示例工作流

- `api/01_soar_voice_clone.json`：SOAR 参考音频克隆。
- `api/02_mf_text_only.json`：MF-2steps 无参考音频快速合成。
- `api/03_soar_long_text.json`：SOAR 长文本分段、显式停顿与音色克隆。
- `ui/01_soar_voice_clone.json`：可拖入 ComfyUI 画布的基础克隆流程。
- `ui/02_soar_long_text.json`：可拖入画布的长文本音色克隆流程。

模型加载器默认按固定清单从 `ComfyUI/models/TTS/dots.tts/` 查找模型。
克隆示例使用 `voice_reference.wav`，请先把已获授权的参考录音放进 ComfyUI
`input` 目录，并将转写文本改成与录音逐字一致。

长文本节点按模型 tokenizer 自动切分，默认段间停顿 200ms。文本中可使用
`<pause=800ms>` 或 `<pause=1.2s>` 覆盖相邻段落之间的自动停顿。
