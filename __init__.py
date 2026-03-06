"""
FEnodes — ComfyUI custom node pack
Author: FugitiveExpert01
"""

from .nodes_tiling import NODE_CLASS_MAPPINGS as _TILING_CLASSES
from .nodes_tiling import NODE_DISPLAY_NAME_MAPPINGS as _TILING_NAMES
from .nodes_text import NODE_CLASS_MAPPINGS as _TEXT_CLASSES
from .nodes_text import NODE_DISPLAY_NAME_MAPPINGS as _TEXT_NAMES

NODE_CLASS_MAPPINGS = {
    **_TILING_CLASSES,
    **_TEXT_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_TILING_NAMES,
    **_TEXT_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
