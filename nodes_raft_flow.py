"""
nodes_raft_flow.py  —  FEnodes
RAFT-Large dense optical flow → VACE V2V control signal
 
Generates a per-clip, globally-normalised Middlebury-encoded RGB flow video
from a ComfyUI IMAGE batch.  Output can be wired directly into a VACE V2V
conditioning node or saved as video.
 
Ref: VACE §3.1 (V2V control signal format) / §4.1 (data construction)
     WAN 2.1 technical report (optical flow evaluation convention)
 
Memory note
-----------
RAFT builds an all-pairs correlation volume of shape [B, H/8, W/8, H/8, W/8].
At native 2K (2048×1080) this is ~4.5 GB per batch item — enough to exhaust
system RAM quickly.  The `max_flow_size` parameter caps the longer edge of the
resolution at which RAFT runs; flow vectors are bilinearly upsampled back to the
original resolution afterwards with magnitude rescaled accordingly.  For a VACE
control signal, motion direction and relative magnitude are fully preserved.
 
Correlation volume comparison (chunk_size=4):
  Native 2K  (2048×1080 → features 256×135) : ~18 GB
  max_flow_size=768 (768×408 → features 96×51)  :  ~364 MB
"""
 
__version__ = "0.0.4"
 
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
 
import torch
import torch.nn.functional as F
 
import folder_paths
 
log = logging.getLogger("FEnodes")
 
_RAFT_LARGE_FALLBACK_URL = (
    "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth"
)
 
_model_cache: Dict[str, object] = {}
_BAR_WIDTH = 30
 
 
# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
 
def _bar(done: int, total: int, width: int = _BAR_WIDTH) -> str:
    filled = int(width * done / total) if total > 0 else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"
 
 
def _sec(elapsed: float) -> str:
    if elapsed < 60:
        return f"{elapsed:.2f}s"
    m, s = divmod(int(elapsed), 60)
    return f"{m}m {s:02d}s"
 
 
def _mb(n_elements: int, itemsize: int = 4) -> float:
    return n_elements * itemsize / (1024 ** 2)
 
 
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
 
    log.info("[FEnodes/FERaftFlow] Weights not found — downloading RAFT Large")
    log.info("[FEnodes/FERaftFlow]   URL  : %s", url)
    log.info("[FEnodes/FERaftFlow]   dest : %s", dest)
    t0 = time.time()
    import torch.hub
    torch.hub.download_url_to_file(url, dest, progress=True)
    log.info(
        "[FEnodes/FERaftFlow] Download complete in %s  (%.1f MB)",
        _sec(time.time() - t0),
        os.path.getsize(dest) / (1024 ** 2),
    )
 
 
def _load_state_dict(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
 
 
def _get_or_load_raft(device: torch.device) -> object:
    from torchvision.models.optical_flow import raft_large
 
    device_key = str(device)
 
    if device_key in _model_cache:
        log.info("[FEnodes/FERaftFlow] Using cached RAFT model on %s", device)
        return _model_cache[device_key]
 
    raft_path = _raft_weights_path()
 
    if not os.path.isfile(raft_path):
        _download_raft_weights(raft_path)
    else:
        log.info(
            "[FEnodes/FERaftFlow] Found weights : %s  (%.1f MB)",
            raft_path,
            os.path.getsize(raft_path) / (1024 ** 2),
        )
 
    log.info("[FEnodes/FERaftFlow] Loading RAFT Large onto %s ...", device)
    t0 = time.time()
 
    model = raft_large(weights=None)
    state_dict = _load_state_dict(raft_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
 
    _model_cache.clear()
    _model_cache[device_key] = model
 
    log.info(
        "[FEnodes/FERaftFlow] Model ready on %s  (loaded in %s)",
        device, _sec(time.time() - t0),
    )
    return model
 
 
# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------
 
def _prepare_raft_frames(
    frames_hwc: torch.Tensor,   # [N, H, W, 3] float32 [0,1]  ComfyUI convention
    max_flow_size: int,
) -> Tuple[torch.Tensor, float, int, int]:
    """
    Convert ComfyUI IMAGE batch to RAFT-ready [N, 3, Hr, Wr] float32 [-1, 1].
 
    If max_flow_size > 0 and the longer edge of the input exceeds it, the
    frames are downsampled so the longer edge == max_flow_size and both
    dimensions are snapped to multiples of 8 (RAFT encoder requirement).
    If no downsampling is needed, H/W are still snapped to multiples of 8.
 
    Returns:
        frames_raft  — [N, 3, Hr, Wr] float32 [-1, 1]  on the same device
        scale        — Hr / H  (use to rescale flow magnitudes after upsampling)
        Hr, Wr       — actual RAFT input spatial dimensions
    """
    N, H, W, C = frames_hwc.shape
    device      = frames_hwc.device
 
    # Compute target resolution
    if max_flow_size > 0 and max(H, W) > max_flow_size:
        s = max_flow_size / max(H, W)
    else:
        s = 1.0
 
    # Snap to multiples of 8 — round rather than floor so we don't shrink more
    # than necessary, but ensure a minimum of 8
    Hr = max(8, round(H * s / 8) * 8)
    Wr = max(8, round(W * s / 8) * 8)
 
    # Permute to [N, 3, H, W] for interpolation / RAFT
    frames_chw = frames_hwc.permute(0, 3, 1, 2)   # float32 [0, 1]
 
    if Hr != H or Wr != W:
        frames_chw = F.interpolate(
            frames_chw,
            size=(Hr, Wr),
            mode="bilinear",
            align_corners=False,
        )
 
    # Scale to [-1, 1] — replicates RAFT transforms() without a CPU round-trip
    frames_raft = frames_chw.mul(2.0).sub_(1.0)
 
    actual_scale = Hr / H   # may differ slightly from s due to rounding
    return frames_raft, actual_scale, Hr, Wr
 
 
# ---------------------------------------------------------------------------
# Flow visualisation with global per-clip normalisation
# ---------------------------------------------------------------------------
 
def _visualise_flows_global(
    flows: torch.Tensor,   # [N, 2, H, W] float32, original pixel units
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Convert [N, 2, H, W] flow to [N, 3, H, W] uint8 Middlebury RGB with global
    per-clip normalisation to match VACE training data convention (VACE §4.1).
 
    flow_to_image normalises per-frame by default.  Injecting a reference pixel
    (global_max, 0) at (0, 0) across all frames simultaneously forces it to use
    global_max as rad_max for every frame, preserving temporal magnitude scale.
    """
    from torchvision.utils import flow_to_image
 
    eps = 1e-7
    N   = flows.shape[0]
 
    magnitudes = (flows[:, 0] ** 2 + flows[:, 1] ** 2).sqrt()
    global_max = float(magnitudes.max().item())
    mean_mag   = float(magnitudes.mean().item())
    min_mag    = float(magnitudes.min().item())
 
    log.info("[FEnodes/FERaftFlow] Flow magnitude stats (original px units):")
    log.info("[FEnodes/FERaftFlow]   min  : %.3f px", min_mag)
    log.info("[FEnodes/FERaftFlow]   mean : %.3f px", mean_mag)
    log.info("[FEnodes/FERaftFlow]   max  : %.3f px  (normalisation constant)", global_max)
 
    if global_max > eps:
        flows_adj               = flows.clone()
        flows_adj[:, 0, 0, 0]  = global_max
        flows_adj[:, 1, 0, 0]  = 0.0
        rgb = flow_to_image(flows_adj)          # [N, 3, H, W] uint8 — one call
    else:
        log.info("[FEnodes/FERaftFlow] Static clip — returning black frames")
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
    matching the VACE training data format (VACE §4.1).
 
    RAFT's correlation volume is O(H² × W²), so running at native 2K+ resolution
    will exhaust RAM.  Use max_flow_size to cap the internal RAFT resolution;
    flow vectors are upsampled back to the original dimensions afterwards with
    magnitudes rescaled to preserve original pixel units.
 
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
        "max_flow_size caps internal RAFT resolution to avoid the correlation "
        "volume OOM that occurs at 2K+ input.  Flow is upsampled back to the "
        "original resolution after inference.  "
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
                "max_flow_size": ("INT", {
                    "default":  768,
                    "min":      256,
                    "max":     2048,
                    "step":     8,
                    "tooltip": (
                        "Longer-edge resolution cap for internal RAFT processing. "
                        "RAFT's correlation volume is O(H²×W²) — at native 2K this "
                        "is ~4.5 GB per batch item.  Frames are downsampled to this "
                        "size before RAFT and the resulting flow is bilinearly "
                        "upsampled back to the original resolution afterwards.  "
                        "For a VACE control signal, 512–768 is sufficient.  "
                        "Set to 2048 to disable downsampling."
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
        max_flow_size: int,
        unique_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor]:
 
        t_total = time.time()
 
        N, H, W, C  = images.shape
        n_pairs     = N - 1
 
        if N < 2:
            raise ValueError(
                f"[FEnodes/FERaftFlow] At least 2 frames required, got {N}."
            )
 
        device   = images.device
        n_chunks = (n_pairs + chunk_size - 1) // chunk_size
 
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info("[FEnodes/FERaftFlow] FERaftFlow  v%s", __version__)
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info("[FEnodes/FERaftFlow] Input      : %d frames  %dx%d", N, W, H)
        log.info("[FEnodes/FERaftFlow] Pairs      : %d", n_pairs)
        log.info("[FEnodes/FERaftFlow] Device     : %s", device)
        log.info("[FEnodes/FERaftFlow] Iters      : %d", iters)
        log.info("[FEnodes/FERaftFlow] Chunk      : %d pairs/call  (%d chunks)", chunk_size, n_chunks)
        log.info("[FEnodes/FERaftFlow] Max flow sz: %d px (longer edge)", max_flow_size)
 
        # ---- Load / retrieve cached model --------------------------------
        model = _get_or_load_raft(device)
 
        # ---- Prepare RAFT input —  downsample + normalise ----------------
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info("[FEnodes/FERaftFlow] Preparing input tensors ...")
        t0 = time.time()
 
        frames_raft, scale, Hr, Wr = _prepare_raft_frames(images, max_flow_size)
 
        downsampled = (Hr != H or Wr != W)
        if downsampled:
            # Correlation volume estimate for logging
            fh, fw       = Hr // 8, Wr // 8
            corr_mb_item = fh * fw * fh * fw * 4 / (1024 ** 2)
            log.info(
                "[FEnodes/FERaftFlow] Downsampled : %dx%d → %dx%d  (scale %.4f)",
                W, H, Wr, Hr, scale,
            )
            log.info(
                "[FEnodes/FERaftFlow] Feat maps   : %dx%d  |  corr volume: %.0f MB/item  "
                "%.0f MB/chunk (chunk_size=%d)",
                fw, fh,
                corr_mb_item,
                corr_mb_item * chunk_size,
                chunk_size,
            )
        else:
            fh, fw       = Hr // 8, Wr // 8
            corr_mb_item = fh * fw * fh * fw * 4 / (1024 ** 2)
            log.info(
                "[FEnodes/FERaftFlow] No downsampling  (%dx%d native)",
                Wr, Hr,
            )
            log.info(
                "[FEnodes/FERaftFlow] Feat maps   : %dx%d  |  corr volume: %.0f MB/item  "
                "%.0f MB/chunk (chunk_size=%d)",
                fw, fh,
                corr_mb_item,
                corr_mb_item * chunk_size,
                chunk_size,
            )
 
        log.info(
            "[FEnodes/FERaftFlow] Input tensor: %.0f MB  (%s)",
            _mb(frames_raft.numel()),
            _sec(time.time() - t0),
        )
 
        # ---- Chunked batched RAFT inference ------------------------------
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info("[FEnodes/FERaftFlow] Running RAFT inference ...")
 
        all_flows: List[torch.Tensor] = []
        pairs_done = 0
        t_infer    = time.time()
 
        with torch.no_grad():
            for chunk_idx, start in enumerate(range(0, n_pairs, chunk_size)):
                end     = min(start + chunk_size, n_pairs)
                b_size  = end - start
                t_chunk = time.time()
 
                b1 = frames_raft[start     : end    ]
                b2 = frames_raft[start + 1 : end + 1]
 
                flow_list  = model(b1, b2, num_flow_updates=iters)
                # Crop any padding that was applied to reach multiples of 8
                flow_crop  = flow_list[-1][:, :, :Hr, :Wr]   # [B, 2, Hr, Wr]
                all_flows.append(flow_crop)
 
                pairs_done += b_size
                elapsed     = time.time() - t_infer
                rate        = pairs_done / elapsed if elapsed > 0 else 0.0
                eta         = (n_pairs - pairs_done) / rate if rate > 0 else 0.0
 
                log.info(
                    "[FEnodes/FERaftFlow] %s  chunk %d/%d  "
                    "(%d pairs, %.2fs)  %.1f pairs/s  ETA %s",
                    _bar(pairs_done, n_pairs),
                    chunk_idx + 1, n_chunks,
                    b_size,
                    time.time() - t_chunk,
                    rate,
                    _sec(eta),
                )
 
        del frames_raft
        t_infer_total = time.time() - t_infer
 
        log.info(
            "[FEnodes/FERaftFlow] Inference done : %d pairs in %s  (%.1f pairs/s)",
            n_pairs, _sec(t_infer_total),
            n_pairs / t_infer_total if t_infer_total > 0 else 0,
        )
 
        # ---- Concatenate all chunks → [N-1, 2, Hr, Wr] ------------------
        flows = torch.cat(all_flows, dim=0)
        del all_flows
 
        # ---- Upsample flow back to original resolution and rescale -------
        if downsampled:
            log.info(
                "[FEnodes/FERaftFlow] Upsampling flow %dx%d → %dx%d ...",
                Wr, Hr, W, H,
            )
            flows = F.interpolate(
                flows,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            # Flow vectors are in units of RAFT-resolution pixels.
            # Dividing by scale converts them to original-resolution pixel units.
            flows = flows.div_(scale)
 
        # ---- Global normalisation + Middlebury colour encoding -----------
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info("[FEnodes/FERaftFlow] Encoding flow to RGB ...")
        t0  = time.time()
        rgb = _visualise_flows_global(flows, H, W)   # [N-1, 3, H, W] uint8
        del flows
 
        log.info(
            "[FEnodes/FERaftFlow] Encoding done in %s",
            _sec(time.time() - t0),
        )
 
        # ---- Convert to ComfyUI IMAGE convention -------------------------
        out = rgb.permute(0, 2, 3, 1).float().div_(255.0)   # [N-1, H, W, 3]
        del rgb
 
        t_wall = time.time() - t_total
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
        log.info(
            "[FEnodes/FERaftFlow] Output  : %d frames  %dx%d  float32",
            out.shape[0], W, H,
        )
        log.info(
            "[FEnodes/FERaftFlow] Total   : %s  (%.1f pairs/s wall-clock)",
            _sec(t_wall),
            n_pairs / t_wall if t_wall > 0 else 0,
        )
        log.info("[FEnodes/FERaftFlow] ──────────────────────────────────────────")
 
        self._send_footer(unique_id, n_pairs=n_pairs, h=H, w=W, elapsed=t_wall)
 
        return (out,)
 
    # ------------------------------------------------------------------
    @staticmethod
    def _send_footer(
        unique_id: Optional[str],
        n_pairs: int,
        h: int,
        w: int,
        elapsed: float = 0.0,
    ) -> None:
        if unique_id is None:
            return
        try:
            from server import PromptServer
            msg = f"{n_pairs} flow frames  |  {w}x{h}  |  {_sec(elapsed)}"
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
