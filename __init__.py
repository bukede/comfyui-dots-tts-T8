"""
@author: T8star-Aix
@title: comfyui-dots-tts-T8
@nickname: dots.tts · T8star-Aix
@description: Native ComfyUI V3 nodes for dots.tts synthesis and voice cloning.
"""

import sys
from pathlib import Path

_NODE_ROOT = Path(__file__).resolve().parent
__version__ = "0.2.0"
if str(_NODE_ROOT) not in sys.path:
    # The vendored upstream uses absolute ``dots_tts`` imports.
    sys.path.insert(0, str(_NODE_ROOT))

try:
    if __package__:
        from .nodes_v3 import comfy_entrypoint
    else:
        import importlib.util
        import types

        _PACKAGE_NAME = "_comfyui_dots_tts_t8_bootstrap"
        _package = sys.modules.get(_PACKAGE_NAME)
        if _package is None:
            _package = types.ModuleType(_PACKAGE_NAME)
            _package.__path__ = [str(_NODE_ROOT)]
            _package.__package__ = _PACKAGE_NAME
            sys.modules[_PACKAGE_NAME] = _package
        _nodes_name = f"{_PACKAGE_NAME}.nodes_v3"
        _nodes = sys.modules.get(_nodes_name)
        if _nodes is None:
            _spec = importlib.util.spec_from_file_location(
                _nodes_name, _NODE_ROOT / "nodes_v3.py"
            )
            _nodes = importlib.util.module_from_spec(_spec)
            sys.modules[_nodes_name] = _nodes
            _spec.loader.exec_module(_nodes)
        comfy_entrypoint = _nodes.comfy_entrypoint
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("comfy_api"):
        raise
    _COMFY_IMPORT_ERROR = exc

    async def comfy_entrypoint():
        raise RuntimeError(
            "A current ComfyUI build with the V3 API is required."
        ) from _COMFY_IMPORT_ERROR


__all__ = ["comfy_entrypoint", "__version__"]
