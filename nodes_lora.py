"""
FEnodes — LoRA utility nodes for VFX production pipelines.
Author: FugitiveExpert01
 
  FELoraLoad             — multi-LoRA browser UI → FE_LORA_STACK
  FEApplyLora            — MODEL + FE_LORA_STACK → MODEL / CLIP
  FELoraTriggerAnalysis  — FE_LORA_STACK + CLIP  → STRING (candidate trigger words)
 
FELoraLoad has a custom JS UI (web/js/fe_power_lora.js) providing:
  • Folder-tree browser with search
  • Per-row on/off toggle and strength
  • "+ Add LoRA" button
  • Optional separate CLIP strength (right-click node)
"""
 
__version__ = "0.0.8"
 
import json
import logging
 
import torch
import torch.nn as nn
import comfy.sd
import comfy.utils
import folder_paths
 
logger = logging.getLogger("FEnodes")
 
FE_LORA_STACK = "FE_LORA_STACK"
 
# ---------------------------------------------------------------------------
# Module-level weight cache
# Keyed on absolute lora_path so identical files shared across multiple
# FELoraLoad nodes or tile streams are only read from disk once.
# ---------------------------------------------------------------------------
_lora_weight_cache: dict[str, dict] = {}
 
# ---------------------------------------------------------------------------
# IS_CHANGED hash cache
# Maps lora_path -> (mtime, sha256_hex) so IS_CHANGED never reads a file
# twice unless it has actually been modified on disk.
# ---------------------------------------------------------------------------
_hash_cache: dict[str, tuple[float, str]] = {}
 
def _cached_file_hash(path: str) -> str:
    """Return SHA256 hex for path, recomputing only when mtime changes."""
    import os, hashlib
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return path  # file missing — use path as sentinel
 
    cached = _hash_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
 
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(131072), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _hash_cache[path] = (mtime, digest)
    return digest
 
# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
try:
    import hashlib
    import os
    import requests as _requests
    from server import PromptServer
    from aiohttp import web as _web
 
    # ── /fenodes/loras — flat file list for the JS browser ────────────────
    @PromptServer.instance.routes.get("/fenodes/loras")
    async def _fenodes_get_loras(request):
        loras = folder_paths.get_filename_list("loras")
        return _web.json_response({"loras": loras})
 
    # ── /fenodes/lora_info — CivitAI + metadata info for one LoRA ────────
    def _sha256(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(131072), b""):
                h.update(chunk)
        return h.hexdigest()
 
    def _read_safetensors_metadata(file_path: str) -> dict:
        """Read the __metadata__ block from a .safetensors header."""
        try:
            with open(file_path, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little", signed=False)
                if header_size <= 0 or header_size > 100_000_000:
                    return {}
                header = json.loads(f.read(header_size))
                meta = header.get("__metadata__", {})
                # Parse any JSON-string values
                for k, v in meta.items():
                    if isinstance(v, str) and v.startswith("{"):
                        try:
                            meta[k] = json.loads(v)
                        except Exception:
                            pass
                return meta
        except Exception as e:
            logger.warning(f"[FEnodes/lora_info] Failed to read metadata: {e}")
            return {}
 
    @PromptServer.instance.routes.get("/fenodes/lora_info")
    async def _fenodes_lora_info(request):
        lora_name = request.rel_url.query.get("name", "")
        refresh   = request.rel_url.query.get("refresh", "false").lower() == "true"
 
        if not lora_name:
            return _web.json_response({"error": "missing name"}, status=400)
 
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.isfile(lora_path):
            return _web.json_response({"error": "file not found"}, status=404)
 
        cache_path = os.path.splitext(lora_path)[0] + ".fe-info.json"
 
        # Load cache if it exists and refresh not requested
        cached = {}
        if os.path.isfile(cache_path) and not refresh:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                logger.info(f"[FEnodes/lora_info] Cache hit: {lora_name}")
                return _web.json_response(cached)
            except Exception:
                pass
 
        logger.info(f"[FEnodes/lora_info] Building info for: {lora_name}")
 
        info = {"file": lora_name}
 
        # Safetensors header metadata
        if lora_path.endswith(".safetensors"):
            meta = _read_safetensors_metadata(lora_path)
            if meta:
                info["metadata"] = meta
                # Pull out common fields
                for field in ("ss_sd_model_name", "ss_base_model_version",
                              "ss_network_module", "ss_num_train_images",
                              "modelspec.title", "modelspec.architecture"):
                    if field in meta:
                        info[field] = meta[field]
 
        # CivitAI lookup via SHA256
        try:
            file_hash = _sha256(lora_path)
            info["sha256"] = file_hash
            civitai_url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
            logger.info(f"[FEnodes/lora_info] Querying CivitAI: {civitai_url}")
            resp = _requests.get(civitai_url, timeout=10)
            if resp.status_code == 200:
                cdata = resp.json()
                info["civitai"] = {
                    "name":          cdata.get("model", {}).get("name", ""),
                    "versionName":   cdata.get("name", ""),
                    "baseModel":     cdata.get("baseModel", ""),
                    "modelId":       cdata.get("modelId"),
                    "versionId":     cdata.get("id"),
                    "trainedWords":  cdata.get("trainedWords", []),
                    "description":   cdata.get("description", ""),
                    "url": (
                        f"https://civitai.com/models/{cdata.get('modelId')}"
                        f"?modelVersionId={cdata.get('id')}"
                        if cdata.get("modelId") else ""
                    ),
                    "images": [
                        {"url": img.get("url"), "nsfw": img.get("nsfwLevel", 0)}
                        for img in cdata.get("images", [])[:3]
                    ],
                }
            elif resp.status_code == 404:
                info["civitai"] = {"error": "not found on CivitAI"}
            else:
                info["civitai"] = {"error": f"CivitAI returned {resp.status_code}"}
        except Exception as e:
            logger.warning(f"[FEnodes/lora_info] CivitAI fetch failed: {e}")
            info["civitai"] = {"error": str(e)}
 
        # Save cache
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"[FEnodes/lora_info] Could not write cache: {e}")
 
        return _web.json_response(info)
 
    logger.info("[FEnodes] Registered /fenodes/loras and /fenodes/lora_info API routes.")
except Exception as _e:
    logger.warning(f"[FEnodes] Could not register API routes: {_e}")
 
 
# ---------------------------------------------------------------------------
# FELoraLoad — multi-LoRA browser → FE_LORA_STACK
# ---------------------------------------------------------------------------
class FELoraLoad:
    """
    Multi-LoRA loader with a custom folder-tree browser UI (see fe_power_lora.js).
 
    The JS widget serialises the full LoRA list as JSON into the hidden
    `loras_json` STRING widget. Each entry:
        { "enabled": bool, "lora": "path/to/lora.safetensors",
          "strength_model": float, "strength_clip": float }
 
    Disabled entries and empty-lora rows are skipped at execute time.
    Outputs a FE_LORA_STACK (list of dicts) for FEApplyLora.
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
        "Add any number of LoRAs, toggle each on/off, set per-LoRA strength. "
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
            lora_name = entry["lora"]
            path = folder_paths.get_full_path("loras", lora_name)
            file_hash = _cached_file_hash(path) if path else lora_name
            # Normalise strengths to 4dp to avoid float serialisation jitter
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
            lora_name = entry.get("lora", "")
            if not lora_name:
                continue
 
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                logger.warning(f"[FEnodes/FELoraLoad] LoRA not found: {lora_name}")
                continue
 
            strength_model = float(entry.get("strength_model", 1.0))
            # strength_clip is always written explicitly by the JS widget.
            # The fallback to strength_model is intentional: if no separate
            # CLIP strength was set, treat them as equal.
            strength_clip  = float(entry.get("strength_clip", strength_model))
 
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
 
        logger.info(f"[FEnodes/FELoraLoad] Stack ready: {len(stack)} LoRA(s) loaded.")
        return (stack,)
 
 
# ---------------------------------------------------------------------------
# Merge helper — pre-scales and combines all LoRA weight dicts into one,
# so load_lora_for_models is called once rather than once per LoRA.
#
# Keys containing "lora_te" are treated as text-encoder (CLIP) keys and
# scaled by that entry's strength_clip; all others are scaled by strength_model.
# This heuristic covers every major training convention (Kohya SD1/SDXL/Flux,
# SimpleTuner, etc.) without requiring model-specific knowledge.
# ---------------------------------------------------------------------------
def _merge_lora_weights(lora_stack: list) -> dict:
    """
    Merge all LoRAs in the stack into a single synthetic weight dict by
    concatenating lora_down / lora_up tensors along the rank dimension,
    with per-LoRA strength and alpha baked in via sqrt-scaling.
 
    Mathematical guarantee:
        combined_up @ combined_down = Σ strength_i * alpha_scale_i * (up_i @ down_i)
 
    This preserves the lora_down / lora_up format that comfy.sd.load_lora_for_models
    expects, so the merged dict can be applied in a single patcher call at
    strength_model=1.0 / strength_clip=1.0.
 
    Layers that appear in only one LoRA pass through unchanged (no concat overhead).
    Layers where ranks differ (shape mismatch) fall back gracefully with a warning.
    """
    staging: dict = {}  # base_key → {"downs": [], "ups": []}
 
    for entry in lora_stack:
        weights        = entry["weights"]
        strength_model = entry["strength_model"]
        strength_clip  = entry["strength_clip"]
 
        for down_key in (k for k in weights if k.endswith(".lora_down.weight")):
            base_key = down_key[: -len(".lora_down.weight")]
            up_key   = base_key + ".lora_up.weight"
            if up_key not in weights:
                continue
 
            down = weights[down_key].float()
            up   = weights[up_key].float()
 
            # Alpha scale: alpha/rank if present, else 1.0
            alpha_key   = base_key + ".alpha"
            rank        = down.shape[0]
            alpha_scale = float(weights[alpha_key]) / rank if alpha_key in weights else 1.0
 
            # Text-encoder keys use strength_clip; everything else strength_model.
            # Covers all major training conventions (Kohya SD1/SDXL/Flux, SimpleTuner).
            is_te    = any(f in base_key for f in ("lora_te", "lora_te1", "lora_te2"))
            strength = strength_clip if is_te else strength_model
 
            # Bake strength + alpha into the matrices via sqrt-scaling so that
            # (scale*up) @ (scale*down) = strength * alpha_scale * up @ down.
            # Sign carried on `up` so the combined product has the correct sign.
            scale = (abs(strength) * alpha_scale) ** 0.5
            sign  = 1.0 if strength >= 0 else -1.0
 
            if base_key not in staging:
                staging[base_key] = {"downs": [], "ups": []}
            staging[base_key]["downs"].append(down * scale)
            staging[base_key]["ups"].append(up * scale * sign)
 
    combined: dict = {}
    for base_key, parts in staging.items():
        downs, ups = parts["downs"], parts["ups"]
        if len(downs) == 1:
            # Only one LoRA targets this layer — no concatenation needed.
            combined[base_key + ".lora_down.weight"] = downs[0]
            combined[base_key + ".lora_up.weight"]   = ups[0]
        else:
            try:
                # Linear: down=(rank, in_features),  up=(out_features, rank)
                # Conv:   down=(rank, in_ch, kH, kW), up=(out_ch, rank, 1, 1)
                combined[base_key + ".lora_down.weight"] = torch.cat(downs, dim=0)
                combined[base_key + ".lora_up.weight"]   = torch.cat(ups,   dim=1)
            except Exception as e:
                logger.warning(
                    f"[FEnodes/FEApplyLora] Merge: shape mismatch on '{base_key}' — "
                    f"falling back to first LoRA only. ({e})"
                )
                combined[base_key + ".lora_down.weight"] = downs[0]
                combined[base_key + ".lora_up.weight"]   = ups[0]
 
    return combined
 
 
# ---------------------------------------------------------------------------
# FEApplyLora — MODEL + FE_LORA_STACK → patched MODEL (+ optional CLIP)
# ---------------------------------------------------------------------------
class FEApplyLora:
    """
    Applies a FE_LORA_STACK (from FELoraLoad) to a MODEL in order.
    Architecture-agnostic: SD1, SDXL, Flux, WAN 2.1/2.2, HunyuanVideo, etc.
 
    Optional inputs (hidden by default, enabled via right-click menu):
      clip           — CLIP to patch alongside the model
      strength_scale — global multiplier applied to all per-LoRA strengths
 
    Application modes
    -----------------
    Stack — each LoRA applied as a sequential patch via the model patcher.
            Safe with any combination of LoRAs. Default.
    Merge — all LoRA weight deltas are pre-scaled and summed into a single
            combined dict, then one patch is applied. Slightly cleaner at
            inference time; best when LoRAs share many of the same target layers.
    """
 
    APPLICATION_MODES = ["Stack", "Merge"]
 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("MODEL",),
                "lora_stack":       (FE_LORA_STACK,),
                "application_mode": (cls.APPLICATION_MODES, {"default": "Stack"}),
            },
            "optional": {
                "clip":             ("CLIP",),
                "strength_scale":   ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Global multiplier applied to all per-LoRA model and clip strengths."
                }),
                # Hidden state widgets — managed by JS, not shown as normal inputs.
                # Stored as STRING so they survive workflow save/load.
                "_fe_show_clip":     ("STRING", {"default": "false"}),
                "_fe_show_strength": ("STRING", {"default": "false"}),
            },
        }
 
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Applies a LoRA stack from FELoraLoad to a MODEL. "
        "Right-click to enable optional CLIP input and/or global strength scale. "
        "Stack: sequential patching. Merge: pre-combines all deltas into one patch."
    )
 
    @classmethod
    def IS_CHANGED(cls, model, lora_stack, application_mode,
                   clip=None, strength_scale=1.0,
                   _fe_show_clip="false", _fe_show_strength="false"):
        if not lora_stack:
            return f"empty:{application_mode}:{strength_scale}"
        parts = [application_mode, str(round(float(strength_scale), 4))]
        for entry in lora_stack:
            sm = round(float(entry.get("strength_model", 1.0)), 4)
            sc = round(float(entry.get("strength_clip",  sm)),   4)
            parts.append(f"{entry.get('name','')}:{sm}:{sc}")
        return "|".join(parts)
 
    def apply(self, model, lora_stack, application_mode,
              clip=None, strength_scale=1.0,
              _fe_show_clip="false", _fe_show_strength="false"):
        if not lora_stack:
            logger.info("[FEnodes/FEApplyLora] Empty stack — model unchanged.")
            return (model, clip)
 
        scale = float(strength_scale)
        if scale != 1.0:
            logger.info(f"[FEnodes/FEApplyLora] Applying global strength_scale={scale}")
 
        if application_mode == "Merge":
            logger.info(
                f"[FEnodes/FEApplyLora] Merge mode — combining "
                f"{len(lora_stack)} LoRA(s) into one patch "
                f"(strength_scale={scale})."
            )
            # Scale a copy of the stack so the cached originals are untouched
            scaled_stack = [
                {**e,
                 "strength_model": e["strength_model"] * scale,
                 "strength_clip":  e["strength_clip"]  * scale}
                for e in lora_stack
            ]
            merged_weights = _merge_lora_weights(scaled_stack)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, merged_weights, 1.0, 1.0
            )
        else:
            # Stack mode — sequential application
            for entry in lora_stack:
                sm = entry["strength_model"] * scale
                sc = entry["strength_clip"]  * scale
                logger.info(
                    f"[FEnodes/FEApplyLora] Stack: applying '{entry['name']}' "
                    f"(model_str={sm:.4f}, clip_str={sc:.4f})"
                )
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip,
                    entry["weights"],
                    sm, sc,
                )
 
        return (model, clip)
 
 
# ---------------------------------------------------------------------------
# FELoraTriggerAnalysis — encoder-agnostic trigger word analysis
# ---------------------------------------------------------------------------
 
# BPE / SentencePiece tokens that score high universally and are not useful triggers.
_STOP_TOKENS = {
    "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|",
    "}", "~", "a", "an", "the", "of", "in", "is", "it", "to", "and", "or",
    "with", "on", "at", "by", "for", "as", "be", "this", "that", "are", "was",
    "from", "its", "not", "but", "have", "he", "she", "they", "we", "you", "i",
    "</w>", "▁", "<pad>", "</s>", "<s>", "<unk>",
    "<|startoftext|>", "<|endoftext|>", "<0x00>",
}
 
# LoRA key suffix patterns for the input-projection (down/A) half of each pair.
_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
 
 
def _discover_encoders(clip):
    """
    Dynamically discover every text encoder sub-module present in a ComfyUI
    CLIP object, along with its token embedding table and vocabulary.
 
    Returns a list of dicts:
        {
            "name":      str,             # attribute name, e.g. "clip_l", "t5xxl"
            "emb_table": Tensor,          # (vocab_size, embed_dim) float32 CPU
            "vocab":     dict[int, str],  # token_id → raw token string
            "key_frags": list[str],       # LoRA key fragments that target this encoder
        }
 
    Strategy
    --------
    Walk every attribute of `clip.cond_stage_model`. For each nn.Module child,
    recursively search for an nn.Embedding or bare weight tensor whose shape is
    (N, D) with N > 1000 — this is the token embedding table regardless of
    whether it is called `token_embedding`, `shared`, or `embed_tokens`.
    Match the same attribute name in `clip.tokenizer` to get the vocab.
 
    This approach is architecture-agnostic: CLIP-L/G (BPE, 768/1280-dim),
    T5-XXL (SentencePiece, 32128 or 256384 vocab, 4096-dim),
    LLaMA/Gemma/Qwen (SentencePiece, 32000-256384 vocab, 4096-dim) all expose
    their embedding tables as the same structural pattern.
    """
    cond = getattr(clip, "cond_stage_model", None)
    tokenizer_root = getattr(clip, "tokenizer", None)
    if cond is None:
        return []
 
    encoders = []
 
    def find_embedding_weight(module):
        """DFS through an nn.Module to find the first token embedding weight."""
        for name, child in module.named_modules():
            if isinstance(child, nn.Embedding):
                w = child.weight
                if w.dim() == 2 and w.shape[0] > 1000:
                    return w.detach().float().cpu()
            # Some encoders store the embedding as a raw Parameter, not nn.Embedding
            for pname, param in child.named_parameters(recurse=False):
                if param.dim() == 2 and param.shape[0] > 1000 and pname in (
                    "weight", "shared_weight", "embed_weight"
                ):
                    return param.detach().float().cpu()
        return None
 
    def get_vocab_for(tokenizer_child):
        """
        Extract a {token_id: token_string} vocab from whatever tokenizer object
        ComfyUI has wired up for this encoder.  Four strategies tried in order:
 
        1. HuggingFace get_vocab() — covers CLIP BPE, Qwen2/2.5/3, LLaMA/Gemma,
           T5 via HF wrapper, and most modern tokenizers.
        2. Wrapped HF tokenizer at .tokenizer — ComfyUI sometimes double-wraps.
        3. SentencePiece sp_model.id_to_piece() — raw T5, some LLaMA variants.
        4. tiktoken _mergeable_ranks — legacy Qwen v1 only; tiktoken Encoding
           objects are identified by their unique signature attributes rather than
           a hardcoded attribute path, so this works regardless of how deeply
           the Encoding is nested inside ComfyUI's wrapper.
        """
        if tokenizer_child is None:
            return {}
 
        # ── Strategy 1: HuggingFace get_vocab() ──────────────────────────
        if hasattr(tokenizer_child, "get_vocab"):
            try:
                v = tokenizer_child.get_vocab()
                return {idx: tok for tok, idx in v.items()}
            except Exception:
                pass
 
        # ── Strategy 2: HF tokenizer wrapped at .tokenizer ───────────────
        inner = getattr(tokenizer_child, "tokenizer", None)
        if inner is not None and hasattr(inner, "get_vocab"):
            try:
                v = inner.get_vocab()
                return {idx: tok for tok, idx in v.items()}
            except Exception:
                pass
 
        # ── Strategy 3: SentencePiece sp_model ───────────────────────────
        for candidate in (tokenizer_child, inner):
            if candidate is None:
                continue
            sp = getattr(candidate, "sp_model", None)
            if sp is not None:
                try:
                    return {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}
                except Exception:
                    pass
 
        # ── Strategy 4: tiktoken (legacy Qwen v1) ────────────────────────
        # tiktoken Encoding objects have both `_mergeable_ranks` ({bytes: int})
        # and `_special_tokens` ({str: int}).  We search recursively by these
        # signature attributes rather than a fixed attribute path.
        def _find_tiktoken(obj, _seen=None, _depth=0):
            if _depth > 5:
                return None
            if _seen is None:
                _seen = set()
            obj_id = id(obj)
            if obj_id in _seen:
                return None
            _seen.add(obj_id)
            if (hasattr(obj, "_mergeable_ranks") and
                    hasattr(obj, "_special_tokens") and
                    isinstance(getattr(obj, "_mergeable_ranks", None), dict)):
                return obj
            for attr in ("tokenizer", "enc", "encoding", "bpe", "inner", "_tokenizer"):
                child = getattr(obj, attr, None)
                if child is not None:
                    result = _find_tiktoken(child, _seen, _depth + 1)
                    if result is not None:
                        return result
            return None
 
        tik_enc = _find_tiktoken(tokenizer_child)
        if tik_enc is not None:
            try:
                vocab = {}
                # _mergeable_ranks: {token_bytes: token_id}
                for tok_bytes, tok_id in tik_enc._mergeable_ranks.items():
                    vocab[tok_id] = tok_bytes.decode("utf-8", errors="replace")
                # _special_tokens: {token_str: token_id}
                for tok_str, tok_id in tik_enc._special_tokens.items():
                    vocab[tok_id] = tok_str
                logger.info(
                    f"[FEnodes/TriggerAnalysis] tiktoken vocab extracted: "
                    f"{len(vocab)} tokens"
                )
                return vocab
            except Exception as e:
                logger.warning(
                    f"[FEnodes/TriggerAnalysis] tiktoken extraction failed: {e}"
                )
 
        logger.warning(
            "[FEnodes/TriggerAnalysis] Could not extract vocab for a discovered "
            "encoder — token IDs will appear as <token_N> placeholders."
        )
        return {}
 
    # Walk the top-level children of cond_stage_model
    for enc_name, enc_module in cond.named_children():
        emb = find_embedding_weight(enc_module)
        if emb is None:
            continue  # This child has no usable embedding table
 
        # Match tokenizer sub-object by the same attribute name
        tok_child = getattr(tokenizer_root, enc_name, None)
        vocab = get_vocab_for(tok_child)
 
        # Build LoRA key fragments: keys in the weights dict that target this encoder.
        # Common conventions: lora_te1_, lora_te2_, lora_te_, lora_{enc_name}_
        key_frags = [
            f"lora_te_{enc_name}_",   # e.g. lora_te_clip_l_
            f"lora_{enc_name}_",       # e.g. lora_clip_l_
            f"lora_te1_",              # Kohya SD1 / SDXL first encoder
            f"lora_te2_",              # Kohya SDXL second encoder
            f"lora_te_",               # generic fallback
            enc_name,                  # bare name fragment
        ]
 
        logger.info(
            f"[FEnodes/TriggerAnalysis] Discovered encoder '{enc_name}': "
            f"vocab={emb.shape[0]}, embed_dim={emb.shape[1]}"
        )
        encoders.append({
            "name":      enc_name,
            "emb_table": emb,
            "vocab":     vocab,
            "key_frags": key_frags,
        })
 
    return encoders
 
 
def _clean_token(tok):
    """Strip BPE/SentencePiece boundary markers and normalise to lowercase."""
    return (
        tok.replace("</w>", "")
           .replace("▁", " ")
           .replace("Ġ", " ")
           .strip()
           .lower()
    )
 
 
def _analyse_encoder(enc_info, all_lora_weights, top_k):
    """
    Run trigger analysis for one encoder against all LoRA weights.
 
    For each lora_down key whose in_features matches this encoder's embed_dim,
    project every token embedding through the layer's input subspace and
    accumulate L2 activation norms.  Returns a ranked list of (token, score)
    tuples, filtered for stop-tokens.
    """
    emb_table = enc_info["emb_table"]      # (vocab_size, embed_dim)
    vocab     = enc_info["vocab"]          # {id: raw_string}
    embed_dim = emb_table.shape[1]
    vocab_size = emb_table.shape[0]
    key_frags  = enc_info["key_frags"]
 
    scores = torch.zeros(vocab_size, dtype=torch.float32)
    layers_used = 0
 
    for lora_entry in all_lora_weights:
        weights = lora_entry["weights"]
 
        for key, tensor in weights.items():
            # Must be a lora_down / lora_A weight
            if not any(key.endswith(s) for s in _DOWN_SUFFIXES):
                continue
            # Must plausibly target this encoder
            if not any(frag in key for frag in key_frags):
                continue
 
            t = tensor.detach().float().cpu()
            # Shape must be (rank, embed_dim) — 2D and inner dim matches
            if t.dim() != 2 or t.shape[1] != embed_dim:
                continue
 
            # (vocab_size, embed_dim) @ (embed_dim, rank) → (vocab_size, rank)
            # Row-wise L2 norm → (vocab_size,) activation score
            scores += (emb_table @ t.T).norm(dim=1)
            layers_used += 1
 
    if layers_used == 0:
        return [], 0
 
    # Decode top tokens, skip stop-tokens and single chars
    sorted_ids  = scores.argsort(descending=True).tolist()
    results = []
    for idx in sorted_ids:
        if len(results) >= top_k:
            break
        raw     = vocab.get(idx, f"<token_{idx}>")
        cleaned = _clean_token(raw)
        if not cleaned or cleaned in _STOP_TOKENS or len(cleaned) < 2:
            continue
        results.append((cleaned, float(scores[idx])))
 
    return results, layers_used
 
 
class FELoraTriggerAnalysis:
    """
    Analyses LoRA weight deltas against every text encoder present in the
    wired CLIP object to surface vocabulary tokens most likely to function
    as trigger words.
 
    Architecture-agnostic
    ---------------------
    Encoders are discovered dynamically by walking cond_stage_model's children
    looking for token embedding tables (any weight shaped vocab_size × embed_dim
    with vocab_size > 1000).  Tokenizers are matched by the same attribute name.
    This covers, without any hardcoded paths:
 
        CLIP-L      (SD1.5, SDXL, Flux, WAN)   — BPE, 49408 × 768
        CLIP-G      (SDXL, SD3)                 — BPE, 49408 × 1280
        T5-XXL      (SD3, Flux, WAN, PixArt)    — SentencePiece, 32128/256384 × 4096
        LLaMA/Gemma (HunyuanVideo, LTX, Lumina) — SentencePiece, 32000+ × 4096
        Dual/triple encoders handled automatically — results labelled per encoder.
 
    Method
    ------
    For each text-encoder layer in the LoRA whose in_features matches a
    discovered encoder's embed_dim, project the full token embedding table
    through the lora_down input subspace and accumulate L2 activation norms.
    High-scoring tokens are the ones most aligned with what the LoRA was
    trained to respond to.
 
    Output
    ------
    A single STRING listing candidates from all encoders, formatted as:
        [clip_l]  token_a, token_b, token_c ...
        [t5xxl]   token_d, token_e, token_f ...
 
    If only one encoder is present the label is omitted.
    Treat the output as a ranked shortlist, not a guaranteed match.
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
        encoders = _discover_encoders(clip)
 
        if not encoders:
            logger.warning(
                "[FEnodes/TriggerAnalysis] No token embedding tables found in this CLIP object. "
                "Cannot perform analysis."
            )
            return ("",)
 
        logger.info(
            f"[FEnodes/TriggerAnalysis] Found {len(encoders)} encoder(s): "
            f"{[e['name'] for e in encoders]}"
        )
 
        output_sections = []
        multi = len(encoders) > 1
 
        for enc in encoders:
            results, layers_used = _analyse_encoder(enc, lora_stack, top_k)
 
            if layers_used == 0:
                logger.info(
                    f"[FEnodes/TriggerAnalysis] '{enc['name']}': "
                    "no matching LoRA layers found for this encoder — skipping."
                )
                continue
 
            logger.info(
                f"[FEnodes/TriggerAnalysis] '{enc['name']}': "
                f"aggregated {layers_used} layer(s), top candidates: "
                f"{[t for t, _ in results[:5]]}"
            )
 
            if not results:
                continue
 
            tokens_str = ", ".join(t for t, _ in results)
            if multi:
                output_sections.append(f"[{enc['name']}]  {tokens_str}")
            else:
                output_sections.append(tokens_str)
 
        if not output_sections:
            logger.warning(
                "[FEnodes/TriggerAnalysis] No LoRA layers matched any discovered encoder. "
                "The LoRA may only target the diffusion model (UNet/DiT) rather than the "
                "text encoder, in which case trigger analysis does not apply."
            )
            return ("",)
 
        output = "\n".join(output_sections)
        return (output,)
 
 
# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
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
