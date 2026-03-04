"""
FEnodes — ComfyUI custom node pack
Author: FugitiveExpert01
"""

import logging

log = logging.getLogger(__name__)

from .nodes_tiling import NODE_CLASS_MAPPINGS as _TILING_CLASSES
from .nodes_tiling import NODE_DISPLAY_NAME_MAPPINGS as _TILING_NAMES
from .nodes_text import NODE_CLASS_MAPPINGS as _TEXT_CLASSES
from .nodes_text import NODE_DISPLAY_NAME_MAPPINGS as _TEXT_NAMES
try:
    from .vae import NODE_CLASS_MAPPINGS as _VAE_CLASSES
    from .vae import NODE_DISPLAY_NAME_MAPPINGS as _VAE_NAMES
except Exception as e:
    log.warning("Failed to load Radiance VAE nodes: %s", e)
    _VAE_CLASSES = {}
    _VAE_NAMES = {}

NODE_CLASS_MAPPINGS = {
    **_TILING_CLASSES,
    **_TEXT_CLASSES,
    **_VAE_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_TILING_NAMES,
    **_TEXT_NAMES,
    **_VAE_NAMES,
}

log.info("FEnodes loaded — %d nodes registered: %s",
         len(NODE_CLASS_MAPPINGS), ", ".join(NODE_CLASS_MAPPINGS.keys()))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
