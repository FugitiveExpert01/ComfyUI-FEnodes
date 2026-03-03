"""
22DogsNodes — ComfyUI custom node pack
"""

from .nodes_tiling import NODE_CLASS_MAPPINGS as _TILING_CLASSES
from .nodes_tiling import NODE_DISPLAY_NAME_MAPPINGS as _TILING_NAMES
from .nodes_text import NODE_CLASS_MAPPINGS as _TEXT_CLASSES
from .nodes_text import NODE_DISPLAY_NAME_MAPPINGS as _TEXT_NAMES
try:
    from .vae import NODE_CLASS_MAPPINGS as _VAE_CLASSES
    from .vae import NODE_DISPLAY_NAME_MAPPINGS as _VAE_NAMES
except Exception as e:
    import logging
    logging.warning(f"[22DogsNodes] Failed to load Radiance VAE nodes: {e}")
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
