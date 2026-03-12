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
 
__version__ = "0.0.5"
 
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
# API route — supplies lora file list to the JS frontend
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web as _web
 
    @PromptServer.instance.routes.get("/fenodes/loras")
    async def _fenodes_get_loras(request):
        loras = folder_paths.get_filename_list("loras")
        return _web.json_response({"loras": loras})
 
    logger.info("[FEnodes] Registered /fenodes/loras API route.")
except Exception as _e:
    logger.warning(f"[FEnodes] Could not register /fenodes/loras route: {_e}")
 
 
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
 
        hashes = []
        for entry in entries:
            if not entry.get("enabled", True) or not entry.get("lora"):
                continue
            path = folder_paths.get_full_path("loras", entry["lora"])
            if path:
                hashes.append(comfy.utils.calculate_file_hash(path))
            else:
                hashes.append(entry["lora"])
        return str(hashes) + loras_json
 
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
    merged: dict = {}
    for entry in lora_stack:
        sm = entry["strength_model"]
        sc = entry["strength_clip"]
        for key, tensor in entry["weights"].items():
            strength = sc if "lora_te" in key else sm
            scaled = tensor.float() * strength
            if key in merged:
                merged[key] = merged[key] + scaled
            else:
                merged[key] = scaled
    return merged
 
 
# ---------------------------------------------------------------------------
# FEApplyLora — MODEL + FE_LORA_STACK → patched MODEL (+ optional CLIP)
# ---------------------------------------------------------------------------
class FEApplyLora:
    """
    Applies a FE_LORA_STACK (from FELoraLoad) to a MODEL in order.
    Architecture-agnostic: SD1, SDXL, Flux, WAN 2.1/2.2, HunyuanVideo, etc.
    CLIP is optional.
 
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
                "clip": ("CLIP",),
            },
        }
 
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Applies a LoRA stack from FELoraLoad to a MODEL (and optionally CLIP). "
        "Stack: sequential patching, safe with any LoRA combination. "
        "Merge: pre-combines all deltas into one patch — best when LoRAs share target layers."
    )
 
    def apply(self, model, lora_stack, application_mode, clip=None):
        if not lora_stack:
            logger.info("[FEnodes/FEApplyLora] Empty stack — model unchanged.")
            return (model, clip)
 
        if application_mode == "Merge":
            logger.info(
                f"[FEnodes/FEApplyLora] Merge mode — combining "
                f"{len(lora_stack)} LoRA(s) into one patch."
            )
            merged_weights = _merge_lora_weights(lora_stack)
            # Strengths are already baked into merged_weights; apply at 1.0.
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, merged_weights, 1.0, 1.0
            )
        else:
            # Stack mode — sequential application
            for entry in lora_stack:
                logger.info(
                    f"[FEnodes/FEApplyLora] Stack: applying '{entry['name']}' "
                    f"(model_str={entry['strength_model']}, clip_str={entry['strength_clip']})"
                )
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip,
                    entry["weights"],
                    entry["strength_model"],
                    entry["strength_clip"],
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
