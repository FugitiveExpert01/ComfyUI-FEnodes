"""
FEnodes -- LoRA nodes
Author: FugitiveExpert01
 
  FELoraLoad             -- multi-LoRA browser UI -> FE_LORA_STACK
  FEApplyLora            -- MODEL + FE_LORA_STACK -> MODEL / CLIP
  FELoraTriggerAnalysis  -- FE_LORA_STACK + CLIP  -> STRING
 
FELoraLoad has a custom JS UI (web/js/fe_power_lora.js).
"""
 
__version__ = "0.1.2"
 
import json
import logging
 
import comfy.sd
import comfy.utils
import folder_paths
 
from .lora_utils import (
    FE_LORA_STACK,
    _lora_weight_cache,
    _cached_file_hash,
    normalize_lora_name,
    merge_lora_weights,
    discover_encoders,
    analyse_encoder,
)
 
logger = logging.getLogger("FEnodes")
 
 
class FELoraLoad:
    """
    Multi-LoRA loader with a custom folder-tree browser UI (fe_power_lora.js).
 
    The JS widget serialises the LoRA list as JSON into the hidden loras_json
    STRING widget. Each entry:
        { "enabled": bool, "lora": "path/to/file.safetensors",
          "strength_model": float, "strength_clip": float }
 
    Disabled and empty rows are skipped. Outputs FE_LORA_STACK for FEApplyLora.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loras_json": ("STRING", {"default": "[]", "multiline": False}),
            }
        }
 
    RETURN_TYPES = (FE_LORA_STACK,)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Multi-LoRA loader with folder-tree browser UI. "
        "Add LoRAs, toggle on/off, set per-LoRA strength. "
        "Connect lora_stack to FEApplyLora."
    )
 
    @classmethod
    def IS_CHANGED(cls, loras_json):
        try:
            entries = json.loads(loras_json)
        except Exception:
            return loras_json
 
        parts = []
        for entry in entries:
            if not entry.get("enabled", True) or not entry.get("lora"):
                continue
            lora_name = normalize_lora_name(entry["lora"])
            path      = folder_paths.get_full_path("loras", lora_name)
            file_hash = _cached_file_hash(path) if path else lora_name
            sm = round(float(entry.get("strength_model", 1.0)), 4)
            sc = round(float(entry.get("strength_clip",  sm)),   4)
            parts.append(f"{lora_name}:{file_hash}:{sm}:{sc}")
        return "|".join(parts)
 
    def load(self, loras_json):
        try:
            entries = json.loads(loras_json)
        except Exception as e:
            logger.warning(f"[FEnodes/FELoraLoad] Failed to parse loras_json: {e}")
            return ([],)
 
        stack = []
        for entry in entries:
            if not entry.get("enabled", True):
                continue
 
            lora_name = normalize_lora_name(entry.get("lora", ""))
            if not lora_name:
                continue
 
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                logger.warning(f"[FEnodes/FELoraLoad] LoRA not found: {lora_name}")
                continue
 
            strength_model = float(entry.get("strength_model", 1.0))
            # strength_clip falls back to strength_model if not explicitly set.
            # The JS widget always writes it; fallback covers older workflows.
            strength_clip = float(entry.get("strength_clip", strength_model))
 
            if lora_path in _lora_weight_cache:
                logger.info(f"[FEnodes/FELoraLoad] Cache hit: '{lora_name}'")
                weights = _lora_weight_cache[lora_path]
            else:
                logger.info(
                    f"[FEnodes/FELoraLoad] Loading '{lora_name}' from disk "
                    f"(model_str={strength_model}, clip_str={strength_clip})"
                )
                weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
                _lora_weight_cache[lora_path] = weights
 
            stack.append({
                "name":           lora_name,
                "weights":        weights,
                "strength_model": strength_model,
                "strength_clip":  strength_clip,
            })
 
        logger.info(f"[FEnodes/FELoraLoad] Stack ready: {len(stack)} LoRA(s).")
        return (stack,)
 
 
class FEApplyLora:
    """
    Applies a FE_LORA_STACK to a MODEL in the chosen application mode.
    Architecture-agnostic: SD1, SDXL, Flux, WAN 2.1/2.2, HunyuanVideo, etc.
 
    Stack -- sequential patching, safe with any combination.
    Merge -- pre-combines all deltas into one patch, best when LoRAs share layers.
    strength_scale multiplies all per-LoRA strengths uniformly.
    lora_stack is optional -- model passes through if unconnected.
    """
 
    APPLICATION_MODES = ["Stack", "Merge"]
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("MODEL",),
                "application_mode": (cls.APPLICATION_MODES, {"default": "Stack"}),
            },
            "optional": {
                "lora_stack":     (FE_LORA_STACK,),
                "clip":           ("CLIP",),
                "strength_scale": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Multiplies all per-LoRA model and clip strengths uniformly.",
                }),
            },
        }
 
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Applies a LoRA stack from FELoraLoad to a MODEL (and optionally CLIP). "
        "strength_scale multiplies all per-LoRA strengths uniformly. "
        "Stack: sequential patching. Merge: pre-combines all deltas into one patch."
    )
 
    @classmethod
    def IS_CHANGED(cls, model, application_mode, lora_stack=None,
                   clip=None, strength_scale=1.0):
        if not lora_stack:
            return f"empty:{application_mode}:{round(float(strength_scale), 4)}"
        parts = [application_mode, str(round(float(strength_scale), 4))]
        for entry in lora_stack:
            sm = round(float(entry.get("strength_model", 1.0)), 4)
            sc = round(float(entry.get("strength_clip",  sm)),   4)
            parts.append(f"{entry.get('name', '')}:{sm}:{sc}")
        return "|".join(parts)
 
    def apply(self, model, application_mode, lora_stack=None,
              clip=None, strength_scale=1.0):
        if not lora_stack:
            logger.info("[FEnodes/FEApplyLora] No lora_stack connected -- model unchanged.")
            return (model, clip)
 
        scale = float(strength_scale)
        if scale != 1.0:
            logger.info(f"[FEnodes/FEApplyLora] strength_scale={scale}")
 
        if application_mode == "Merge":
            logger.info(
                f"[FEnodes/FEApplyLora] Merge: combining {len(lora_stack)} LoRA(s) "
                f"(strength_scale={scale})"
            )
            scaled_stack = [
                {**e,
                 "strength_model": e["strength_model"] * scale,
                 "strength_clip":  e["strength_clip"]  * scale}
                for e in lora_stack
            ]
            merged = merge_lora_weights(scaled_stack)
            model, clip = comfy.sd.load_lora_for_models(model, clip, merged, 1.0, 1.0)
        else:
            for entry in lora_stack:
                sm = entry["strength_model"] * scale
                sc = entry["strength_clip"]  * scale
                logger.info(
                    f"[FEnodes/FEApplyLora] Stack: '{entry['name']}' "
                    f"(model_str={sm:.4f}, clip_str={sc:.4f})"
                )
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, entry["weights"], sm, sc
                )
 
        return (model, clip)
 
 
class FELoraTriggerAnalysis:
    """
    Analyses LoRA weight deltas against every text encoder in the wired CLIP
    to surface candidate trigger words.
    Architecture-agnostic: CLIP-L/G, T5-XXL, LLaMA/Gemma and combinations.
    """
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": (FE_LORA_STACK,),
                "clip":       ("CLIP",),
                "top_k":      ("INT", {"default": 10, "min": 1, "max": 50}),
            }
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("candidate_triggers",)
    OUTPUT_NODE = True
    FUNCTION = "analyse"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Analyses LoRA weight deltas against all text encoders in the wired CLIP "
        "to surface candidate trigger words. Works with CLIP-L/G, T5-XXL, LLaMA/Gemma "
        "and any combination of dual/triple encoders."
    )
 
    def analyse(self, lora_stack, clip, top_k):
        encoders = discover_encoders(clip)
 
        if not encoders:
            logger.warning("[FEnodes/TriggerAnalysis] No token embedding tables found.")
            return ("",)
 
        logger.info(
            f"[FEnodes/TriggerAnalysis] Found {len(encoders)} encoder(s): "
            f"{[e['name'] for e in encoders]}"
        )
 
        sections = []
        multi = len(encoders) > 1
 
        for enc in encoders:
            results, layers_used = analyse_encoder(enc, lora_stack, top_k)
 
            if layers_used == 0:
                logger.info(f"[FEnodes/TriggerAnalysis] '{enc['name']}': no matching layers.")
                continue
 
            logger.info(
                f"[FEnodes/TriggerAnalysis] '{enc['name']}': "
                f"{layers_used} layer(s), top: {[t for t, _ in results[:5]]}"
            )
 
            if not results:
                continue
 
            tokens_str = ", ".join(t for t, _ in results)
            sections.append(f"[{enc['name']}]  {tokens_str}" if multi else tokens_str)
 
        if not sections:
            logger.warning(
                "[FEnodes/TriggerAnalysis] No LoRA layers matched any encoder. "
                "LoRA likely targets the diffusion model only."
            )
            return ("",)
 
        return ("\n".join(sections),)
 
 
NODE_CLASS_MAPPINGS = {
    "FELoraLoad":            FELoraLoad,
    "FEApplyLora":           FEApplyLora,
    "FELoraTriggerAnalysis": FELoraTriggerAnalysis,
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "FELoraLoad":            "LoRA Load ⚡",
    "FEApplyLora":           "Apply LoRA ⚡",
    "FELoraTriggerAnalysis": "LoRA Trigger Analysis 🔍",
}
