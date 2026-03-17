"""
nodes_raft_flow.py  —  FEnodes
RAFT-Large dense optical flow → VACE V2V control signal

Generates a per-clip, globally-normalised Middlebury-encoded RGB flow video
from a ComfyUI IMAGE batch.  Output can be wired directly into a VACE V2V
conditioning node or saved as video.

Ref: VACE §3.1 (V2V control signal format) / §4.1 (data construction)
     WAN 2.1 technical report (optical flow evaluation convention)
"""

__version__ = "0.0.2"

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import folder_paths

log = logging.getLogger("FEnodes")

# Fallback URL matching torchvision's own registry (torchvision >= 0.15)
_RAFT_LARGE_FALLBACK_URL = (
    "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
)

# Module-level model cache  { device_str: model }
_model_cache: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

def _raft_weights_path() -> str:
    raft_dir = os.path.join(folder_paths.models_dir, "raft")
    os.makedirs(raft_dir, exist_ok=True)
    return os.path.join(raft_dir, "raft_large.pth")


def _download_raft_weights(dest: str) -> None:
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights
        url = Raft_Large_Weights.DEFAULT.url
    except AttributeError:
        url = _RAFT_LARGE_FALLBACK_URL

    log.info("[FEnodes/FERaftFlow] Downloading RAFT Large weights -> %s", dest)
    import torch.hub
    torch.hub.download_url_to_file(url, dest, progress=True)
    log.info("[FEnodes/FERaftFlow] Download complete.")


def _load_state_dict(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_or_load_raft(device: torch.device) -> object:
    from torchvision.models.optical_flow import raft_large

    device_key = str(device)

    if device_key in _model_cache:
        return _model_cache[device_key]

    raft_path = _raft_weights_path()

    if not os.path.isfile(raft_path):
        _download_raft_weights(raft_path)

    log.info("[FEnodes/FERaftFlow] Loading RAFT weights from %s", raft_path)
    model = raft_large(weights=None)
    state_dict = _load_state_dict(raft_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    _model_cache.clear()
    _model_cache[device_key] = model
    log.info("[FEnodes/FERaftFlow] RAFT model ready on %s.", device)
    return model


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def _to_raft_input(frames_u8: torch.Tensor) -> torch.Tensor:
    """
    Convert uint8 [N, 3, H, W] -> float32 [N, 3, H, W] in [-1, 1].

    Replicates what Raft_Large_Weights.DEFAULT.transforms() does, as a single
    fused GPU op.  Bypassing the transforms() callable avoids the CPU round-trip
    and removes any version-dependent uncertainty about whether it accepts CUDA
    tensors.
    """
    return frames_u8.float().div_(127.5).sub_(1.0)


def _pad8(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad [B, C, H, W] so H and W are multiples of 8 (RAFT encoder requirement).
    Returns (padded_tensor, (pad_h, pad_w)).
    """
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return t, (0, 0)
    return F.pad(t, (0, pad_w, 0, pad_h), mode="replicate"), (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Flow visualisation with global per-clip normalisation
# ---------------------------------------------------------------------------

def _visualise_flows_global(
    flows: torch.Tensor,   # [N, 2, H, W] float32, pixel units
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Convert [N, 2, H, W] flow tensor to [N, 3, H, W] uint8 Middlebury RGB
    using torchvision's flow_to_image with *global* per-clip normalisation.

    VACE uses a single normalisation constant across the entire clip so that
    temporal motion scale is preserved: a slow frame appears less saturated
    than a fast one rather than being independently stretched to full saturation.

    flow_to_image normalises each frame by its own local max.  To override
    this we inject a reference pixel (global_max, 0) at position (0, 0) across
    all frames simultaneously, forcing flow_to_image to use global_max as its
    rad_max for every frame.  The reference pixel itself receives the colour for
    maximum rightward motion — a single-pixel artefact negligible for a VACE
    control signal.

    Called once on the full [N, 2, H, W] stack rather than per-frame.
    """
    from torchvision.utils import flow_to_image

    eps  = 1e-7
    N    = flows.shape[0]

    magnitudes = (flows[:, 0] ** 2 + flows[:, 1] ** 2).sqrt()
    global_max = float(magnitudes.max().item())

    log.info(
        "[FEnodes/FERaftFlow] Global max magnitude: %.3f px  (%d pairs)",
        global_max, N,
    )

    if global_max > eps:
        flows_adj = flows.clone()
        flows_adj[:, 0, 0, 0] = global_max   # broadcast reference pixel to all frames
        flows_adj[:, 1, 0, 0] = 0.0
        rgb = flow_to_image(flows_adj)        # [N, 3, H, W] uint8 — single call
    else:
        rgb = torch.zeros(N, 3, H, W, dtype=torch.uint8, device=flows.device)

    return rgb


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class FERaftFlow:
    """
    Generate a dense optical flow RGB video using RAFT Large.

    Input  : IMAGE batch [N, H, W, 3] float32 [0, 1]  (N >= 2)
    Output : IMAGE batch [N-1, H, W, 3] float32 [0, 1]

    Encoding uses the Middlebury colour-wheel convention (hue = direction,
    saturation = magnitude, value = 1.0) with per-clip global normalisation,
    matching the VACE training data format (VACE ss4.1).

    Processing is GPU-accelerated throughout:
    - Input conversion and padding performed once on the full clip.
    - Frame pairs batched through RAFT in configurable chunks.
    - flow_to_image called once on the full flow stack.

    Weights are stored in {ComfyUI}/models/raft/raft_large.pth and downloaded
    automatically on first use.
    """

    CATEGORY = "FEnodes"
    FUNCTION = "run"

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("flow_frames",)

    DESCRIPTION = (
        "Computes dense optical flow between consecutive frame pairs using "
        "RAFT Large (torchvision) and returns a Middlebury-coded RGB video "
        "compatible with VACE V2V control signal inputs.  "
        "Per-clip global normalisation matches VACE training conventions.  "
        "Fully GPU-accelerated with configurable batch chunking.  "
        "Weights download automatically to {ComfyUI}/models/raft/raft_large.pth."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Source video frames from an upstream loader "
                        "(e.g. VHS LoadVideo).  Minimum 2 frames.  "
                        "Shape: [N, H, W, 3] float32 [0, 1]."
                    ),
                }),
                "iters": ("INT", {
                    "default": 20,
                    "min":      1,
                    "max":     32,
                    "step":     1,
                    "tooltip": (
                        "RAFT refinement iterations.  Higher = more accurate "
                        "flow at greater compute cost.  "
                        "20 is the standard RAFT paper default."
                    ),
                }),
                "chunk_size": ("INT", {
                    "default":  4,
                    "min":      1,
                    "max":     64,
                    "step":     1,
                    "tooltip": (
                        "Frame pairs per RAFT forward call.  Higher = faster "
                        "but more VRAM.  Reduce if you hit OOM on long clips."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # ------------------------------------------------------------------
    def run(
        self,
        images: torch.Tensor,
        iters: int,
        chunk_size: int,
        unique_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor]:

        N, H, W, C = images.shape

        if N < 2:
            raise ValueError(
                f"[FEnodes/FERaftFlow] At least 2 frames required, got {N}."
            )

        n_pairs = N - 1
        device  = images.device

        log.info(
            "[FEnodes/FERaftFlow] %d pairs | %dx%d | iters=%d | chunk=%d | device=%s",
            n_pairs, H, W, iters, chunk_size, device,
        )

        # ---- Load / retrieve cached model --------------------------------
        model = _get_or_load_raft(device)

        # ---- Build RAFT input tensor — once for the whole clip -----------
        #
        # ComfyUI IMAGE: float32 [N, H, W, 3] [0, 1]
        #   → permute  → [N, 3, H, W]
        #   → uint8    (×255, clamp)
        #   → float32  (÷127.5 − 1.0, i.e. [-1, 1])
        #   → pad H,W to multiples of 8
        #
        # Everything stays on `device`.  The intermediate uint8 is transient;
        # _to_raft_input uses in-place ops so it does not double the allocation.
        frames_chw  = (
            images.permute(0, 3, 1, 2)
            .mul(255.0)
            .clamp(0, 255)
            .to(torch.uint8)
        )
        frames_raft         = _to_raft_input(frames_chw)   # [N, 3, H, W] float32
        del frames_chw                                      # free uint8 copy

        frames_pad, (ph, pw) = _pad8(frames_raft)          # [N, 3, H', W']
        del frames_raft                                     # free unpadded copy

        # ---- Chunked batched RAFT inference ------------------------------
        all_flows: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, n_pairs, chunk_size):
                end = min(start + chunk_size, n_pairs)

                b1 = frames_pad[start : end]       # [B, 3, H', W']
                b2 = frames_pad[start + 1 : end + 1]

                flow_list  = model(b1, b2, num_flow_updates=iters)
                flow_crop  = flow_list[-1][:, :, :H, :W]   # [B, 2, H, W]
                all_flows.append(flow_crop)

        del frames_pad

        # Concatenate chunks → [N-1, 2, H, W]
        flows = torch.cat(all_flows, dim=0)
        del all_flows

        # ---- Global normalisation + Middlebury colour encoding -----------
        # Single call on the full flow stack — no per-frame loop
        rgb = _visualise_flows_global(flows, H, W)   # [N-1, 3, H, W] uint8
        del flows

        # ---- Convert to ComfyUI IMAGE convention -------------------------
        out = rgb.permute(0, 2, 3, 1).float().div(255.0)   # [N-1, H, W, 3]
        del rgb

        log.info(
            "[FEnodes/FERaftFlow] Done. Output: %s dtype=%s.",
            tuple(out.shape), out.dtype,
        )

        self._send_footer(unique_id, n_pairs=n_pairs, h=H, w=W)

        return (out,)

    # ------------------------------------------------------------------
    @staticmethod
    def _send_footer(
        unique_id: Optional[str],
        n_pairs: int,
        h: int,
        w: int,
    ) -> None:
        if unique_id is None:
            return
        try:
            from server import PromptServer
            msg = f"{n_pairs} flow frames  |  {w}x{h}"
            PromptServer.instance.send_sync(
                "fe_progress_text",
                {"node_id": unique_id, "text": msg},
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FERaftFlow": FERaftFlow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FERaftFlow": "RAFT Optical Flow 🌊",
}
