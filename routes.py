"""
FEnodes -- API route registrations
Author: FugitiveExpert01

Each route is registered in its own try/except so a failure in one
does not affect the other, and the error message is specific.
"""

import json
import logging
import os

import folder_paths

from .lora_utils import (
    read_safetensors_metadata,
    fetch_civitai_info,
    normalize_lora_name,
)

logger = logging.getLogger("FEnodes")


def register_routes():
    """
    Register all FEnodes API routes with ComfyUI's PromptServer.
    Called once from __init__.py at startup.
    """
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError as e:
        logger.error(
            f"[FEnodes/routes] Could not import server or aiohttp -- "
            f"API routes will be unavailable. ({e})"
        )
        return

    # -- GET /fenodes/loras ------------------------------------------------
    # Returns the flat list of lora filenames for the JS browser widget.
    try:
        @PromptServer.instance.routes.get("/fenodes/loras")
        async def _get_loras(request):
            loras = folder_paths.get_filename_list("loras")
            return web.json_response({"loras": loras})

        logger.info("[FEnodes/routes] Registered GET /fenodes/loras")
    except Exception as e:
        logger.error(f"[FEnodes/routes] Failed to register /fenodes/loras: {e}")

    # -- GET /fenodes/lora_info --------------------------------------------
    # Returns CivitAI info + safetensors metadata for a single LoRA.
    # Query params:
    #   name     (required) -- LoRA filename as returned by /fenodes/loras
    #   refresh  (optional) -- "true" to bypass cache and re-fetch
    try:
        @PromptServer.instance.routes.get("/fenodes/lora_info")
        async def _get_lora_info(request):
            lora_name = request.rel_url.query.get("name", "")
            refresh   = request.rel_url.query.get("refresh", "false").lower() == "true"

            if not lora_name:
                logger.warning("[FEnodes/routes] /fenodes/lora_info called with no name")
                return web.json_response({"error": "missing 'name' query parameter"}, status=400)

            lora_name = normalize_lora_name(lora_name)
            lora_path = folder_paths.get_full_path("loras", lora_name)

            if not lora_path:
                logger.warning(f"[FEnodes/routes] /fenodes/lora_info: not found in loras folder: {lora_name}")
                return web.json_response({"error": f"LoRA not found: {lora_name}"}, status=404)

            if not os.path.isfile(lora_path):
                logger.warning(f"[FEnodes/routes] /fenodes/lora_info: path exists in index but file missing on disk: {lora_path}")
                return web.json_response({"error": f"File missing on disk: {lora_path}"}, status=404)

            cache_path = os.path.splitext(lora_path)[0] + ".fe-info.json"

            # Serve from cache unless refresh requested
            if os.path.isfile(cache_path) and not refresh:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    logger.info(f"[FEnodes/routes] Cache hit for: {lora_name}")
                    return web.json_response(cached)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"[FEnodes/routes] Cache file corrupt or unreadable for {lora_name}, "
                        f"rebuilding. ({e})"
                    )

            logger.info(f"[FEnodes/routes] Building info for: {lora_name}")
            info = {"file": lora_name}

            # Safetensors header metadata
            if lora_path.endswith(".safetensors"):
                meta = read_safetensors_metadata(lora_path)
                if meta:
                    info["metadata"] = meta
                    for field in (
                        "ss_sd_model_name", "ss_base_model_version",
                        "ss_network_module", "ss_num_train_images",
                        "modelspec.title", "modelspec.architecture",
                    ):
                        if field in meta:
                            info[field] = meta[field]

            # CivitAI lookup
            civitai = fetch_civitai_info(lora_path)
            info["civitai"] = civitai
            if "sha256" in civitai:
                info["sha256"] = civitai["sha256"]

            # Write cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2)
            except OSError as e:
                logger.warning(
                    f"[FEnodes/routes] Could not write cache for {lora_name}: {e}"
                )

            return web.json_response(info)

        logger.info("[FEnodes/routes] Registered GET /fenodes/lora_info")
    except Exception as e:
        logger.error(f"[FEnodes/routes] Failed to register /fenodes/lora_info: {e}")
