from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parents[1]
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

from dots_tts_t8.model_catalog import ModelCatalog  # noqa: E402
from dots_tts_t8.model_download import download_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned dots.tts models.")
    parser.add_argument("--model", choices=ModelCatalog.load().keys(), default="soar")
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional Hugging Face-compatible mirror endpoint.",
    )
    args = parser.parse_args()
    target = args.comfyui_root / "models" / "TTS" / "dots.tts"
    path, report = download_model(
        args.model,
        target,
        endpoint=args.endpoint,
        progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
    )
    print(
        json.dumps({"path": str(path), "report": report.as_dict()}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
