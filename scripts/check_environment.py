from __future__ import annotations

import json
import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parents[1]
if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

from dots_tts_t8.environment import inspect_environment  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(inspect_environment().as_dict(), ensure_ascii=False, indent=2))
