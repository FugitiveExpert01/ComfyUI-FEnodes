"""
nodes_raft_flow.py  —  FEnodes
RAFT-Large dense optical flow → VACE V2V control signal

Generates a per-clip, globally-normalised Middlebury-encoded RGB flow video
from a ComfyUI IMAGE batch.  Output can be wired directly into a VACE V2V
conditioning node or saved as video.

Ref: VACE §3.1 (V2V control signal format) / §4.1 (data construction)
     WAN 2.1 technical report (optical flow evaluation convention)
"""

__version__ = "0.0.1"

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import folder_paths

log = logging.getLogger("FEnodes")

# ---------------------------------------------------------------------------
# Fallback URL in case Raft_Large_Weights.DEFAULT.url is unavailable
# (matches torchvision's own registry as of torchvision 0.15+)
# ---------------------------------------------------------------------------
_RAFT_LARGE_FALLBACK_URL = (
    "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
)

# Module-level model cache  { device_str: model }
_model_cache: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------

def _raft_weights_path() -> str:
    """Return the canonical path for RAFT Large weights inside ComfyUI models/."""
    raft_dir = os.path.join(folder_paths.models_dir, "raft")
    os.makedirs(raft_dir, exist_ok=True)
    return os.path.join(raft_dir, "raft_large.pth")


def _download_raft_weights(dest: str) -> None:
    """Download RAFT Large weights to *dest* using torch.hub."""
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights
        url = Raft_Large_Weights.DEFAULT.url
    except AttributeError:
        url = _RAFT_LARGE_FALLBACK_URL

    log.info("[FEnodes/FERaftFlow] Downloading RAFT Large weights → %s", dest)
    import torch.hub
    torch.hub.download_url_to_file(url, dest, progress=True)
    log.info("[FEnodes/FERaftFlow] Download complete.")


def _load_state_dict(path: str) -> dict:
    """Load a state_dict from *path*, compatible with PyTorch >= 1.13."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only not available in older PyTorch builds
        return torch.load(path, map_location="cpu")


def _get_or_load_raft(device: torch.device) -> object:
    """
    Return a cached, eval-mode RAFT Large model on *device*.

    On first call: checks for weights in {ComfyUI}/models/raft/raft_large.pth,
    downloads if absent, loads, and caches.  Subsequent calls return the cached
    instance; if the requested device differs from the cached model's device the
    cache is invalidated and the model is moved.
    """
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

    _model_cache.clear()          # evict any model on a different device
    _model_cache[device_key] = model
    log.info("[FEnodes/FERaftFlow] RAFT model ready on %s.", device)
    return model


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def _pad8(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad a [B, C, H, W] tensor so H and W are multiples of 8 (RAFT requirement).
    Padding is applied to the bottom and right edges via replication.
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
    raw_flows: List[torch.Tensor],
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Convert a list of [2, H, W] flow tensors to an [N, 3, H, W] uint8 RGB batch
    using torchvision's flow_to_image with *global* per-clip normalisation.

    VACE training data uses a single normalisation constant (the maximum flow
    magnitude across the entire clip) rather than per-frame normalisation.  This
    ensures that temporal motion magnitude is encoded consistently: a slow frame
    appears less saturated than a fast one.

    Implementation note
    -------------------
    torchvision.utils.flow_to_image normalises each frame independently by its
    own maximum magnitude.  To force global normalisation we inject a reference
    pixel (global_max, 0) at position (0, 0) for every frame.  This makes
    flow_to_image use global_max as its rad_max, so every pixel in every frame
    is divided by the same constant.  The reference pixel itself is painted with
    the colour corresponding to maximum rightward motion; this single-pixel
    artefact is inconsequential for a VACE control signal.
    """
    from torchvision.utils import flow_to_image

    eps = 1e-7
    all_flows = torch.stack(raw_flows, dim=0)          # [N, 2, H, W]
    magnitudes = (all_flows[:, 0] ** 2 + all_flows[:, 1] ** 2).sqrt()
    global_max = float(magnitudes.max().item())

    log.info(
        "[FEnodes/FERaftFlow] Global max flow magnitude: %.3f px  (clip: %d frames)",
        global_max, len(raw_flows),
    )

    rgb_frames: List[torch.Tensor] = []

    for flow in raw_flows:  # flow: [2, H, W]
        if global_max > eps:
            # Inject reference pixel so flow_to_image normalises by global_max
            flow_adj = flow.clone()
            flow_adj[0, 0, 0] = global_max
            flow_adj[1, 0, 0] = 0.0
            # flow_to_image accepts [2, H, W] or [N, 2, H, W]; pass with batch dim
            rgb = flow_to_image(flow_adj.unsqueeze(0)).squeeze(0)  # [3, H, W] uint8
        else:
            # Fully static clip — return black frames
            rgb = torch.zeros(3, H, W, dtype=torch.uint8, device=flow.device)

        rgb_frames.append(rgb)

    return torch.stack(rgb_frames, dim=0)   # [N, 3, H, W] uint8


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class FERaftFlow:
    """
    Generate a dense optical flow RGB video using RAFT Large.

    • Input  : IMAGE batch [N, H, W, 3] float32 [0, 1]  (N ≥ 2)
    • Output : IMAGE batch [N-1, H, W, 3] float32 [0, 1]

    The output is encoded using the Middlebury colour-wheel convention
    (hue = direction, saturation = magnitude, value = 1.0) and normalised
    per-clip (single global maximum), matching the format used to construct
    VACE training data (VACE §4.1).  It can be passed directly to a VACE V2V
    conditioning node or saved as a video via VHS Save Video.

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
        "Flow is normalised per-clip (global max) to match VACE training "
        "conventions.  Weights download automatically to "
        "{ComfyUI}/models/raft/raft_large.pth on first use."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Source video or image sequence decoded by an upstream loader "
                        "(e.g. VHS LoadVideo).  Must contain at least 2 frames.  "
                        "Shape: [N, H, W, 3] float32 [0, 1]."
                    ),
                }),
                "iters": ("INT", {
                    "default": 20,
                    "min":      1,
                    "max":     32,
                    "step":     1,
                    "tooltip": (
                        "RAFT refinement iterations.  Higher values produce more "
                        "accurate flow at the cost of compute.  "
                        "20 is the default used in the original RAFT paper."
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
        unique_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor]:

        from torchvision.models.optical_flow import Raft_Large_Weights

        N, H, W, C = images.shape

        if N < 2:
            raise ValueError(
                f"[FEnodes/FERaftFlow] At least 2 frames required, got {N}."
            )

        device = images.device
        log.info(
            "[FEnodes/FERaftFlow] %d frame pairs | %d×%d | iters=%d | device=%s",
            N - 1, H, W, iters, device,
        )

        # ---- Load / retrieve cached model --------------------------------
        model   = _get_or_load_raft(device)
        transforms = Raft_Large_Weights.DEFAULT.transforms()

        # ---- Convert input to uint8 [N, 3, H, W] for torchvision --------
        # RAFT's torchvision transforms expect uint8 [B, 3, H, W].
        # ComfyUI IMAGE convention is float32 [0, 1] → ×255 → uint8.
        frames_chw = (
            images.permute(0, 3, 1, 2)   # [N, H, W, 3] → [N, 3, H, W]
            .mul(255.0)
            .clamp(0, 255)
            .to(torch.uint8)
        )

        # ---- Compute raw flows for every consecutive pair ----------------
        raw_flows: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(N - 1):
                # [1, 3, H, W] uint8
                f1 = frames_chw[i    ].unsqueeze(0)
                f2 = frames_chw[i + 1].unsqueeze(0)

                # Apply RAFT pre-processing transforms (uint8 → float, [-1, 1])
                # Transforms are lightweight tensor ops; keep on CPU, then move.
                f1_t, f2_t = transforms(f1.cpu(), f2.cpu())
                f1_t = f1_t.to(device)
                f2_t = f2_t.to(device)

                # Pad H and W to multiples of 8 (hard requirement of RAFT's
                # feature encoder downsampling path)
                f1_pad, (ph, pw) = _pad8(f1_t)
                f2_pad, _         = _pad8(f2_t)

                # Forward — returns a list of progressively refined [B, 2, H, W] flows
                flow_list = model(f1_pad, f2_pad, num_flow_updates=iters)

                # Take the final (most refined) prediction and crop padding
                flow = flow_list[-1][:, :, :H, :W]   # [1, 2, H, W] px units
                raw_flows.append(flow.squeeze(0))      # → [2, H, W]

        # ---- Global per-clip normalisation + Middlebury colour encoding --
        rgb_batch = _visualise_flows_global(raw_flows, H, W)  # [N-1, 3, H, W] uint8

        # ---- Convert to ComfyUI IMAGE convention  [N-1, H, W, 3] float32 [0,1]
        out = (
            rgb_batch
            .permute(0, 2, 3, 1)   # [N-1, 3, H, W] → [N-1, H, W, 3]
            .to(torch.float32)
            .div(255.0)
        )

        log.info(
            "[FEnodes/FERaftFlow] Done. Output: %s, dtype=%s.",
            tuple(out.shape), out.dtype,
        )

        # ---- Optional node-footer status text ---------------------------
        self._send_footer(
            unique_id,
            n_pairs=N - 1,
            h=H,
            w=W,
        )

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
            msg = f"{n_pairs} flow frames  |  {w}×{h}"
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
