"""
FEnodes — LoRA utility nodes for VFX production pipelines.
Author: FugitiveExpert01

Provides a two-node pattern for LoRA application that mirrors the separation
of concerns in KJNodes' WanVideo workflow:

  FELoraLoad   →  loads a .safetensors LoRA file into a LORA object
  FEApplyLora  →  applies that LORA object to any ComfyUI MODEL (+ optional CLIP)

Both nodes use ComfyUI's native model patcher under the hood, so they are
architecture-agnostic: SD1, SDXL, Flux, WAN 2.1/2.2, HunyuanVideo, etc.
"""

__version__ = "0.0.1"

import logging

import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger("FEnodes")

# ---------------------------------------------------------------------------
# Custom type tag — keeps LORA objects distinct from raw tensors or strings
# ---------------------------------------------------------------------------
LORA_TYPE = "FE_LORA"


class FELoraLoad:
    """
    Loads a LoRA file from the ComfyUI loras directory and outputs it as a
    typed LORA object.  Pass the output directly to FEApplyLora; the actual
    weights are loaded once and cached by ComfyUI's normal node-caching logic.

    Keeps the file-selection concern separate from the application concern so
    the same LoRA can be wired to multiple FEApplyLora nodes (e.g. different
    strengths per tile stream) without re-reading from disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {}),
            }
        }

    RETURN_TYPES = (LORA_TYPE,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Loads a LoRA .safetensors file from ComfyUI/models/loras. "
        "Connect the output to FEApplyLora to apply it to any model."
    )

    @classmethod
    def IS_CHANGED(cls, lora_name):
        """Re-execute only when the file itself changes (hash-based caching)."""
        lora_path = folder_paths.get_full_path("loras", lora_name)
        return comfy.utils.calculate_file_hash(lora_path)

    def load(self, lora_name):
        lora_path = folder_paths.get_full_path_by_key("loras", lora_name)
        logger.info(f"[FEnodes/FELoraLoad] Loading: {lora_name}")
        weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return ({"name": lora_name, "path": lora_path, "weights": weights},)


class FEApplyLora:
    """
    Applies a LORA object (from FELoraLoad) to a MODEL using ComfyUI's native
    model-patcher system.

    Works across all ComfyUI-supported architectures without any model-specific
    key remapping — the patcher handles that internally.

    Inputs
    ------
    model          : MODEL   — any ComfyUI diffusion model
    lora           : FE_LORA — output of FELoraLoad
    strength_model : FLOAT   — LoRA weight for the diffusion model  (default 1.0)
    clip           : CLIP    — optional; pass through and apply LoRA if provided
    strength_clip  : FLOAT   — LoRA weight for CLIP               (default 1.0)

    Outputs
    -------
    model : MODEL — patched model (LoRA applied via ComfyUI model patcher)
    clip  : CLIP  — patched CLIP, or None pass-through if no CLIP was connected
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora": (LORA_TYPE,),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Applies a LoRA loaded by FELoraLoad to a MODEL (and optionally CLIP). "
        "Architecture-agnostic: works with SD1, SDXL, Flux, WAN 2.1/2.2, "
        "HunyuanVideo, and any other ComfyUI-compatible diffusion model. "
        "Negative strengths invert the LoRA effect."
    )

    def apply(
        self,
        model,
        lora,
        strength_model: float,
        clip=None,
        strength_clip: float = 1.0,
    ):
        lora_name = lora.get("name", "unknown")
        lora_weights = lora["weights"]

        clip_info = f"clip_str={strength_clip}" if clip is not None else "clip=None"
        logger.info(
            f"[FEnodes/FEApplyLora] Applying '{lora_name}' — "
            f"model_str={strength_model}, {clip_info}"
        )

        # comfy.sd.load_lora_for_models accepts clip=None gracefully and
        # returns (patched_model, None) in that case.
        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            model, clip, lora_weights, strength_model, strength_clip
        )

        return (patched_model, patched_clip)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "FELoraLoad": FELoraLoad,
    "FEApplyLora": FEApplyLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FELoraLoad": "LoRA Load 🔗",
    "FEApplyLora": "Apply LoRA 🔗",
}
