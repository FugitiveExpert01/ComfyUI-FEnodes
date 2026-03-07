"""
FEnodes — ComfyUI custom node pack
Author: FugitiveExpert01
"""
import logging
try:
    from .nodes_tiling import NODE_CLASS_MAPPINGS as _TILING_CLASSES
    from .nodes_tiling import NODE_DISPLAY_NAME_MAPPINGS as _TILING_NAMES
except Exception as e:
    logging.warning(f"[FEnodes] Failed to load Tiling nodes: {e}")
    _TILING_CLASSES = {}
    _TILING_NAMES = {}
try:
    from .nodes_text import NODE_CLASS_MAPPINGS as _TEXT_CLASSES
    from .nodes_text import NODE_DISPLAY_NAME_MAPPINGS as _TEXT_NAMES
except Exception as e:
    logging.warning(f"[FEnodes] Failed to load Text nodes: {e}")
    _TEXT_CLASSES = {}
    _TEXT_NAMES = {}
try:
    from .nodes_color import NODE_CLASS_MAPPINGS as _COLOR_CLASSES
    from .nodes_color import NODE_DISPLAY_NAME_MAPPINGS as _COLOR_NAMES
except Exception as e:
    logging.warning(f"[FEnodes] Failed to load Color nodes: {e}")
    _COLOR_CLASSES = {}
    _COLOR_NAMES = {}
NODE_CLASS_MAPPINGS = {
    **_TILING_CLASSES,
    **_TEXT_CLASSES,
    **_COLOR_CLASSES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_TILING_NAMES,
    **_TEXT_NAMES,
    **_COLOR_NAMES,
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
