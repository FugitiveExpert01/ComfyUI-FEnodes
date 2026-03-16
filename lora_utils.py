"""
FEnodes -- LoRA shared utilities
Author: FugitiveExpert01
 
Contains all non-node, non-route logic:
  - Type tag
  - Hash cache (in-memory, IS_CHANGED only)
  - File utilities (path normalisation, hash, safetensors metadata)
  - CivitAI fetch (disk cache in system temp directory)
  - Merge logic
  - Trigger analysis helpers
"""
 
import hashlib
import json
import logging
import os
import tempfile
 
import torch
import torch.nn as nn
 
import comfy.utils
import folder_paths
 
logger = logging.getLogger("FEnodes")
 
# ---------------------------------------------------------------------------
# Type tag
# ---------------------------------------------------------------------------
FE_LORA_STACK = "FE_LORA_STACK"
 
# ---------------------------------------------------------------------------
# requests -- optional but required for CivitAI lookups
# ---------------------------------------------------------------------------
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False
    logger.warning(
        "[FEnodes/lora_utils] 'requests' is not installed. "
        "CivitAI lookups will be unavailable. "
        "Install with: pip install requests"
    )
 
# ---------------------------------------------------------------------------
# In-memory hash cache
# Maps lora_path -> (mtime, sha256_hex).
# Used exclusively by IS_CHANGED so large files are not re-read on every
# graph evaluation. Lives in memory only -- no disk writes.
# ---------------------------------------------------------------------------
_hash_cache: dict[str, tuple[float, str]] = {}
 
# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------
 
def normalize_lora_name(lora_name: str) -> str:
    """
    Normalise a LoRA name to forward slashes so folder_paths.get_full_path
    works correctly on both Windows and Linux.
 
    Windows may serialise subfolder paths with backslashes into the workflow
    JSON. folder_paths expects forward slashes on all platforms.
    """
    return lora_name.replace("\\", "/")
 
 
# ---------------------------------------------------------------------------
# File hash utilities
# ---------------------------------------------------------------------------
 
def _cached_file_hash(path: str) -> str:
    """
    Return SHA256 hex for path, recomputing only when mtime changes.
    Used by IS_CHANGED to avoid reading large files on every graph evaluation.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        logger.warning(f"[FEnodes/lora_utils] Cannot stat file for hash: {path}")
        return path  # use path as sentinel so IS_CHANGED stays stable
 
    cached = _hash_cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
 
    logger.info(f"[FEnodes/lora_utils] Computing hash for: {os.path.basename(path)}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(131072), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _hash_cache[path] = (mtime, digest)
    return digest
 
 
def _sha256_full(path: str) -> str:
    """
    Compute full SHA256 of a file unconditionally.
    Used for CivitAI lookups where we need the definitive hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(131072), b""):
            h.update(chunk)
    return h.hexdigest()
 
 
# ---------------------------------------------------------------------------
# Safetensors metadata reader
# ---------------------------------------------------------------------------
 
def read_safetensors_metadata(file_path: str) -> dict:
    """
    Read the __metadata__ block from a .safetensors file header without
    loading any tensors. Returns an empty dict on any failure.
    """
    try:
        with open(file_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little", signed=False)
            if header_size <= 0 or header_size > 100_000_000:
                logger.warning(
                    f"[FEnodes/lora_utils] Implausible header size "
                    f"({header_size}) in {os.path.basename(file_path)} -- skipping metadata."
                )
                return {}
            header = json.loads(f.read(header_size))
            meta = header.get("__metadata__", {})
            # Some tools serialise nested JSON as strings -- parse those too.
            for k, v in meta.items():
                if isinstance(v, str) and v.startswith("{"):
                    try:
                        meta[k] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            return meta
    except FileNotFoundError:
        logger.warning(f"[FEnodes/lora_utils] File not found when reading metadata: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"[FEnodes/lora_utils] JSON decode error in metadata header: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[FEnodes/lora_utils] Unexpected error reading metadata from {file_path}: {e}")
        return {}
 
 
# ---------------------------------------------------------------------------
# CivitAI fetch
# ---------------------------------------------------------------------------
 
_CIVITAI_BASE = "https://civitai.com/api/v1/model-versions/by-hash"
 
# CivitAI info cache lives in the system temp directory, never in the
# user's models or loras folders.
_TEMP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fenodes_lora_cache")
os.makedirs(_TEMP_CACHE_DIR, exist_ok=True)
 
 
def _info_cache_path(file_hash: str) -> str:
    # Cache file keyed on SHA256 so it is stable across renames/moves.
    return os.path.join(_TEMP_CACHE_DIR, f"{file_hash}.fe-info.json")
 
 
def fetch_civitai_info(file_path: str) -> dict:
    """
    Query CivitAI for model info by SHA256 hash of the file.
 
    Returns a dict with keys:
        name, versionName, baseModel, modelId, versionId,
        trainedWords, description, url, images
 
    On any failure returns {"error": "<reason>"} so callers can surface the
    specific problem rather than getting a silent empty result.
    """
    if not _REQUESTS_AVAILABLE:
        return {"error": "requests library not installed"}
 
    try:
        file_hash = _sha256_full(file_path)
    except OSError as e:
        return {"error": f"Could not read file for hashing: {e}"}
 
    url = f"{_CIVITAI_BASE}/{file_hash}"
    logger.info(f"[FEnodes/civitai] Querying: {url}")
 
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to CivitAI -- check internet connection"}
    except requests.exceptions.Timeout:
        return {"error": "CivitAI request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"CivitAI request failed: {e}"}
 
    if resp.status_code == 404:
        return {"error": "Model not found on CivitAI"}
    if resp.status_code != 200:
        return {"error": f"CivitAI returned HTTP {resp.status_code}"}
 
    try:
        data = resp.json()
    except ValueError:
        return {"error": "CivitAI returned invalid JSON"}
 
    model_id   = data.get("modelId")
    version_id = data.get("id")
 
    return {
        "sha256":       file_hash,
        "name":         data.get("model", {}).get("name", ""),
        "versionName":  data.get("name", ""),
        "baseModel":    data.get("baseModel", ""),
        "modelId":      model_id,
        "versionId":    version_id,
        "trainedWords": data.get("trainedWords", []),
        "description":  data.get("description", ""),
        "url": (
            f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
            if model_id else ""
        ),
        "images": [
            {"url": img.get("url"), "nsfw": img.get("nsfwLevel", 0)}
            for img in data.get("images", [])[:3]
        ],
    }
 
 
# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------
 
# Supported (down_suffix, up_suffix) pairs.
# Kohya: lora_down/lora_up. Diffusers: lora_A/lora_B (SD Turbo, SDXL Turbo, etc.)
_LORA_PAIRS = [
    (".lora_down.weight", ".lora_up.weight"),
    (".lora_A.weight",    ".lora_B.weight"),
]
 
 
def merge_lora_weights(lora_stack: list) -> dict:
    """
    Merge all LoRAs in the stack into a single synthetic weight dict by
    concatenating lora_down / lora_up tensors along the rank dimension,
    with per-LoRA strength and alpha baked in via sqrt-scaling.
 
    Mathematical guarantee:
        combined_up @ combined_down = sum_i( strength_i * alpha_scale_i * (up_i @ down_i) )
 
    Handles both naming conventions transparently (Kohya lora_down/up and
    Diffusers lora_A/B). Output is always in lora_down/up format which
    comfy.sd.load_lora_for_models accepts for all architectures.
 
    Layers that appear in only one LoRA pass through unchanged.
    Layers where ranks differ fall back gracefully to the first LoRA with a warning.
    """
    staging: dict = {}
 
    for entry in lora_stack:
        weights        = entry["weights"]
        strength_model = entry["strength_model"]
        strength_clip  = entry["strength_clip"]
 
        # Collect all base keys regardless of naming convention
        base_keys = set()
        for down_sfx, _ in _LORA_PAIRS:
            for k in weights:
                if k.endswith(down_sfx):
                    base_keys.add(k[: -len(down_sfx)])
 
        for base_key in base_keys:
            down, up = None, None
            for down_sfx, up_sfx in _LORA_PAIRS:
                dk = base_key + down_sfx
                uk = base_key + up_sfx
                if dk in weights and uk in weights:
                    down = weights[dk].float()
                    up   = weights[uk].float()
                    break
            if down is None:
                continue
 
            alpha_key   = base_key + ".alpha"
            rank        = down.shape[0]
            alpha_scale = float(weights[alpha_key]) / rank if alpha_key in weights else 1.0
 
            # Text-encoder keys use strength_clip, all others use strength_model
            is_te    = any(f in base_key for f in ("lora_te", "lora_te1", "lora_te2"))
            strength = strength_clip if is_te else strength_model
 
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
            combined[base_key + ".lora_down.weight"] = downs[0]
            combined[base_key + ".lora_up.weight"]   = ups[0]
        else:
            try:
                combined[base_key + ".lora_down.weight"] = torch.cat(downs, dim=0)
                combined[base_key + ".lora_up.weight"]   = torch.cat(ups,   dim=1)
            except Exception as e:
                logger.warning(
                    f"[FEnodes/merge] Shape mismatch on '{base_key}' "
                    f"-- falling back to first LoRA only. ({e})"
                )
                combined[base_key + ".lora_down.weight"] = downs[0]
                combined[base_key + ".lora_up.weight"]   = ups[0]
 
    return combined
 
 
# ---------------------------------------------------------------------------
# Trigger analysis helpers
# ---------------------------------------------------------------------------
 
_STOP_TOKENS = {
    "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|",
    "}", "~", "a", "an", "the", "of", "in", "is", "it", "to", "and", "or",
    "with", "on", "at", "by", "for", "as", "be", "this", "that", "are", "was",
    "from", "its", "not", "but", "have", "he", "she", "they", "we", "you", "i",
    "</w>", "\u2581", "<pad>", "</s>", "<s>", "<unk>",
    "<|startoftext|>", "<|endoftext|>", "<0x00>",
}
 
_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
 
 
def discover_encoders(clip) -> list:
    """
    Dynamically discover every text encoder in a ComfyUI CLIP object along
    with its token embedding table and vocabulary.
 
    Returns a list of dicts:
        { name, emb_table (Tensor), vocab (dict[int,str]), key_frags (list[str]) }
 
    Architecture-agnostic: covers CLIP-L/G, T5-XXL, LLaMA/Gemma/Qwen.
    """
    cond = getattr(clip, "cond_stage_model", None)
    tokenizer_root = getattr(clip, "tokenizer", None)
    if cond is None:
        return []
 
    encoders = []
 
    def find_embedding_weight(module):
        for _, child in module.named_modules():
            if isinstance(child, nn.Embedding):
                w = child.weight
                if w.dim() == 2 and w.shape[0] > 1000:
                    return w.detach().float().cpu()
            for pname, param in child.named_parameters(recurse=False):
                if param.dim() == 2 and param.shape[0] > 1000 and pname in (
                    "weight", "shared_weight", "embed_weight"
                ):
                    return param.detach().float().cpu()
        return None
 
    def get_vocab_for(tok):
        if tok is None:
            return {}
        if hasattr(tok, "get_vocab"):
            try:
                return {idx: t for t, idx in tok.get_vocab().items()}
            except Exception:
                pass
        inner = getattr(tok, "tokenizer", None)
        if inner is not None and hasattr(inner, "get_vocab"):
            try:
                return {idx: t for t, idx in inner.get_vocab().items()}
            except Exception:
                pass
        for candidate in (tok, inner):
            if candidate is None:
                continue
            sp = getattr(candidate, "sp_model", None)
            if sp is not None:
                try:
                    return {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}
                except Exception:
                    pass
        # tiktoken fallback
        def _find_tiktoken(obj, seen=None, depth=0):
            if depth > 5:
                return None
            if seen is None:
                seen = set()
            if id(obj) in seen:
                return None
            seen.add(id(obj))
            if (hasattr(obj, "_mergeable_ranks") and
                    hasattr(obj, "_special_tokens") and
                    isinstance(getattr(obj, "_mergeable_ranks", None), dict)):
                return obj
            for attr in ("tokenizer", "enc", "encoding", "bpe", "inner", "_tokenizer"):
                child = getattr(obj, attr, None)
                if child is not None:
                    result = _find_tiktoken(child, seen, depth + 1)
                    if result is not None:
                        return result
            return None
 
        tik = _find_tiktoken(tok)
        if tik is not None:
            try:
                vocab = {}
                for tok_bytes, tok_id in tik._mergeable_ranks.items():
                    vocab[tok_id] = tok_bytes.decode("utf-8", errors="replace")
                for tok_str, tok_id in tik._special_tokens.items():
                    vocab[tok_id] = tok_str
                return vocab
            except Exception as e:
                logger.warning(f"[FEnodes/trigger] tiktoken extraction failed: {e}")
 
        logger.warning("[FEnodes/trigger] Could not extract vocab for a discovered encoder.")
        return {}
 
    for enc_name, enc_module in cond.named_children():
        emb = find_embedding_weight(enc_module)
        if emb is None:
            continue
        tok_child = getattr(tokenizer_root, enc_name, None)
        vocab = get_vocab_for(tok_child)
        key_frags = [
            f"lora_te_{enc_name}_",
            f"lora_{enc_name}_",
            "lora_te1_",
            "lora_te2_",
            "lora_te_",
            enc_name,
        ]
        logger.info(
            f"[FEnodes/trigger] Discovered encoder '{enc_name}': "
            f"vocab={emb.shape[0]}, embed_dim={emb.shape[1]}"
        )
        encoders.append({
            "name":      enc_name,
            "emb_table": emb,
            "vocab":     vocab,
            "key_frags": key_frags,
        })
 
    return encoders
 
 
def clean_token(tok: str) -> str:
    """Strip BPE/SentencePiece boundary markers and normalise to lowercase."""
    return (
        tok.replace("</w>", "")
           .replace("\u2581", " ")
           .replace("\u0120", " ")
           .strip()
           .lower()
    )
 
 
def analyse_encoder(enc_info: dict, all_lora_weights: list, top_k: int):
    """
    Score vocabulary tokens against LoRA weight deltas for one encoder.
    Returns ([(token, score), ...], layers_used).
    """
    emb_table  = enc_info["emb_table"]
    vocab      = enc_info["vocab"]
    embed_dim  = emb_table.shape[1]
    vocab_size = emb_table.shape[0]
    key_frags  = enc_info["key_frags"]
 
    scores = torch.zeros(vocab_size, dtype=torch.float32)
    layers_used = 0
 
    for lora_entry in all_lora_weights:
        for key, tensor in lora_entry["weights"].items():
            if not any(key.endswith(s) for s in _DOWN_SUFFIXES):
                continue
            if not any(frag in key for frag in key_frags):
                continue
            t = tensor.detach().float().cpu()
            if t.dim() != 2 or t.shape[1] != embed_dim:
                continue
            scores += (emb_table @ t.T).norm(dim=1)
            layers_used += 1
 
    if layers_used == 0:
        return [], 0
 
    results = []
    for idx in scores.argsort(descending=True).tolist():
        if len(results) >= top_k:
            break
        raw     = vocab.get(idx, f"<token_{idx}>")
        cleaned = clean_token(raw)
        if not cleaned or cleaned in _STOP_TOKENS or len(cleaned) < 2:
            continue
        results.append((cleaned, float(scores[idx])))
 
    return results, layers_used
