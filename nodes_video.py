"""
FEnodes — Video utility nodes for VFX production pipelines.
Author: FugitiveExpert01
"""

import gc
import time

import torch
import numpy as np

import comfy.model_management as model_management
import comfy.utils
import folder_paths


# ─────────────────────────────────────────────────────────────────────────────

class VideoUpscaleWithModel:
    """
    Memory-efficient upscaling of video batches using a ComfyUI upscale model.

    Supports three device strategies:
      • auto             — detects available VRAM and chooses the best strategy
      • keep_loaded      — model stays on GPU for the full run (fastest, needs VRAM)
      • load_unload_each_frame — model moves to GPU per batch then back to CPU
      • cpu_only         — runs entirely on CPU (slowest, lowest VRAM usage)

    Tiles large frames automatically to avoid OOM; tile sizes are tuned per
    strategy.  A coloured progress bar is printed to the ComfyUI console.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "images": ("IMAGE",),
                "upscale_method": (cls.upscale_methods,),
                "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "device_strategy": (
                    ["auto", "load_unload_each_frame", "keep_loaded", "cpu_only"],
                    {"default": "auto"},
                ),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "upscale_video"
    CATEGORY = "FEnodes"
    DESCRIPTION = (
        "Upscales every frame of an IMAGE batch with a selected upscale model. "
        "Automatically tiles frames to fit available VRAM."
    )

    def __init__(self):
        self.steps = 0
        self.step = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def upscale_video(self, model_name, images, upscale_method, factor,
                      device_strategy="auto", batch_size=4):
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)
        upscale_model = self._load_upscale_model(upscale_model_path)
        device = model_management.get_torch_device()

        # ── Strategy auto-detection ──────────────────────────────────
        if device_strategy == "auto":
            if torch.cuda.is_available():
                total   = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                device_strategy = (
                    "keep_loaded" if (total - reserved) / total > 0.5
                    else "load_unload_each_frame"
                )
            else:
                device_strategy = "cpu_only"

        num_frames = images.shape[0]
        old_h, old_w = images.shape[1], images.shape[2]
        new_h = int(old_h * factor)
        new_w = int(old_w * factor)

        self.steps = num_frames
        self.step  = 0

        print(
            f"\n[FEnodes/VideoUpscale] {num_frames} frames  "
            f"{old_w}x{old_h} → {new_w}x{new_h}  "
            f"[strategy={device_strategy}, batch_size={batch_size}]"
        )

        # Pre-convert all frames to (N, C, H, W) once
        all_chw = images.movedim(-1, 1)  # (N, H, W, C) → (N, C, H, W)

        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result = self._process_batches(
                upscale_model, all_chw, "cpu",
                upscale_method, new_w, new_h, batch_size,
                tile_x=64, tile_y=64,
            )
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result = self._process_batches(
                upscale_model, all_chw, device,
                upscale_method, new_w, new_h, batch_size,
                tile_x=128, tile_y=128,
            )
        else:  # load_unload_each_frame (actually per-batch)
            result = self._process_batches_load_unload(
                upscale_model, all_chw, device,
                upscale_method, new_w, new_h, batch_size,
                tile_x=96, tile_y=96,
            )

        print()  # newline after progress bar
        return (torch.stack(result),)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_upscale_model(self, model_path):
        from comfy_extras.chainner_models import model_loading

        sd = comfy.utils.load_torch_file(model_path)
        model = model_loading.load_state_dict(sd).eval()
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    # ------------------------------------------------------------------
    # Batch processing — cpu_only / keep_loaded
    # ------------------------------------------------------------------

    def _process_batches(self, upscale_model, all_chw, device,
                         upscale_method, new_w, new_h, batch_size,
                         tile_x, tile_y):
        result = []
        num_frames  = all_chw.shape[0]
        start_time  = time.time()

        with torch.no_grad():
            for start in range(0, num_frames, batch_size):
                end   = min(start + batch_size, num_frames)
                batch = all_chw[start:end].to(device)

                s = comfy.utils.tiled_scale(
                    batch,
                    lambda a: upscale_model(a),
                    tile_x=tile_x, tile_y=tile_y,
                    overlap=8,
                    upscale_amount=upscale_model.scale,
                )
                s = comfy.utils.common_upscale(s, new_w, new_h, upscale_method, crop="disabled")
                s = s.movedim(1, -1).cpu()

                result.extend(s[j] for j in range(s.shape[0]))
                self.step = end
                self._print_progress(start_time)
                del batch, s

        return result

    # ------------------------------------------------------------------
    # Batch processing — load_unload_each_frame (per-batch in practice)
    # ------------------------------------------------------------------

    def _process_batches_load_unload(self, upscale_model, all_chw, device,
                                     upscale_method, new_w, new_h, batch_size,
                                     tile_x, tile_y):
        result = []
        num_frames = all_chw.shape[0]
        start_time = time.time()

        with torch.no_grad():
            for start in range(0, num_frames, batch_size):
                end   = min(start + batch_size, num_frames)
                upscale_model = upscale_model.to(device)
                batch = all_chw[start:end].to(device)

                s = comfy.utils.tiled_scale(
                    batch,
                    lambda a: upscale_model(a),
                    tile_x=tile_x, tile_y=tile_y,
                    overlap=8,
                    upscale_amount=upscale_model.scale,
                )
                s = comfy.utils.common_upscale(s, new_w, new_h, upscale_method, crop="disabled")
                s = s.movedim(1, -1).cpu()

                result.extend(s[j] for j in range(s.shape[0]))
                self.step = end
                self._print_progress(start_time)
                del batch, s

                upscale_model = upscale_model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _print_progress(self, start_time):
        elapsed = time.time() - start_time
        eta     = (elapsed / self.step * (self.steps - self.step)) if self.step > 0 else 0
        percent = (self.step / self.steps) * 100
        bar     = "█" * int(percent / 5)
        space   = " " * (20 - int(percent / 5))
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        eta_str     = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(
            f"\r\033[32m|{bar}{space}| {self.step}/{self.steps} "
            f"[{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m",
            end="", flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────

class FreeVideoMemory:
    """
    Explicitly flushes GPU memory mid-pipeline.

    Pass any IMAGE batch through this node to trigger a garbage-collect and
    CUDA cache clear between heavy nodes.  Optional aggressive mode calls
    additional CUDA allocator APIs when available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "aggressive_cleanup": (["disable", "enable"], {"default": "disable"}),
                "report_memory":      (["disable", "enable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    FUNCTION      = "cleanup_memory"
    CATEGORY      = "FEnodes"
    DESCRIPTION   = (
        "Pass-through node that flushes GPU memory between pipeline stages. "
        "Images are unchanged; only memory is freed."
    )

    def cleanup_memory(self, images, aggressive_cleanup="disable", report_memory="enable"):
        def _gb(bytes_val):
            return bytes_val / (1024 ** 3)

        if report_memory == "enable" and torch.cuda.is_available():
            print(
                f"[FEnodes/FreeVideoMemory] Before: "
                f"{_gb(torch.cuda.memory_allocated()):.2f} GB allocated, "
                f"{_gb(torch.cuda.memory_reserved()):.2f} GB reserved"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive_cleanup == "enable":
                torch.cuda.synchronize()
                if hasattr(torch.cuda, "caching_allocator_delete_caches"):
                    torch.cuda.caching_allocator_delete_caches()

        if report_memory == "enable" and torch.cuda.is_available():
            print(
                f"[FEnodes/FreeVideoMemory] After:  "
                f"{_gb(torch.cuda.memory_allocated()):.2f} GB allocated, "
                f"{_gb(torch.cuda.memory_reserved()):.2f} GB reserved"
            )

        return (images,)


# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VideoUpscaleWithModel": VideoUpscaleWithModel,
    "FreeVideoMemory":       FreeVideoMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUpscaleWithModel": "Video Upscale With Model",
    "FreeVideoMemory":       "Free Video Memory",
}
