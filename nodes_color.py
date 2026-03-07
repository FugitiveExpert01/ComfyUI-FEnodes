"""
FEnodes — Color anchoring nodes for VFX production pipelines.
Author: FugitiveExpert01
Version: v0.0.1

ChromaPin:
    Fixes colour drift in AI-processed video by pinning it to a reference image.

    The core idea:
        1. The user supplies the *original* reference image and tells us which
           frame index in the processed video corresponds to it.
        2. We compute a color-correction transform from
               processed_ref_frame  →  original_reference_image
           This transform captures the model's color drift as a calibration pair.
        3. We apply that *same* transform to every frame in the video.

    Why not just do per-frame color matching?
        Each frame has different content and different color statistics.
        Matching every frame's stats to the reference image would destroy
        the natural color variation between frames.  Instead we use the
        anchor pair to measure the drift *once* and then remove it everywhere.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# sRGB ↔ CIE L*a*b* helpers (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    return np.where(img <= 0.04045,
                    img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, None)
    return np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * (img ** (1.0 / 2.4)) - 0.055).astype(np.float32)


# sRGB D65 matrices
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)

_M_XYZ2RGB = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660,  1.8760108,  0.0415560],
                        [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)

_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)


def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 [0,1] sRGB → CIE L*a*b*."""
    lin = _srgb_to_linear(img)
    xyz = lin.reshape(-1, 3) @ _M_RGB2XYZ.T
    xyz /= _D65
    eps, kappa = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1).reshape(img.shape[0], img.shape[1], 3)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIE L*a*b* → (H, W, 3) float32 [0,1] sRGB."""
    flat = lab.reshape(-1, 3)
    L, a, b = flat[:, 0], flat[:, 1], flat[:, 2]
    eps, kappa = 0.008856, 903.3
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr, yr, zr], axis=-1) * _D65
    rgb_lin = xyz @ _M_XYZ2RGB.T
    return np.clip(_linear_to_srgb(rgb_lin).reshape(lab.shape[0], lab.shape[1], 3), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Color-correction methods
# Each method returns: (params_dict, apply_fn)
# apply_fn(frame_np, params, strength) → corrected_np
# ---------------------------------------------------------------------------

def _fit_reinhard(src: np.ndarray, tgt: np.ndarray):
    """
    Reinhard et al. Lab-space colour transfer.
    Params: per-channel mean & std of source and target in Lab.
    """
    src_lab = _rgb_to_lab(src)
    tgt_lab = _rgb_to_lab(tgt)
    params = {}
    for c in range(3):
        params[f's_mean_{c}'] = float(src_lab[..., c].mean())
        params[f's_std_{c}']  = float(src_lab[..., c].std() + 1e-7)
        params[f't_mean_{c}'] = float(tgt_lab[..., c].mean())
        params[f't_std_{c}']  = float(tgt_lab[..., c].std() + 1e-7)
    return params


def _apply_reinhard(frame: np.ndarray, params: dict, strength: float) -> np.ndarray:
    lab = _rgb_to_lab(frame)
    out = lab.copy()
    for c in range(3):
        sm, ss = params[f's_mean_{c}'], params[f's_std_{c}']
        tm, ts = params[f't_mean_{c}'], params[f't_std_{c}']
        out[..., c] = (lab[..., c] - sm) / ss * ts + tm
    blended = out * strength + lab * (1.0 - strength)
    return _lab_to_rgb(blended)


def _fit_linear_rgb(src: np.ndarray, tgt: np.ndarray):
    """
    Per-channel linear (gain + offset) in sRGB.
    Matches first two moments (mean, std) of each channel.
    """
    params = {}
    for c in range(3):
        sm = float(src[..., c].mean())
        ss = float(src[..., c].std() + 1e-7)
        tm = float(tgt[..., c].mean())
        ts = float(tgt[..., c].std() + 1e-7)
        params[f'gain_{c}']   = ts / ss
        params[f'offset_{c}'] = tm - (ts / ss) * sm
    return params


def _apply_linear_rgb(frame: np.ndarray, params: dict, strength: float) -> np.ndarray:
    out = frame.copy()
    for c in range(3):
        corrected_c = np.clip(frame[..., c] * params[f'gain_{c}'] + params[f'offset_{c}'], 0.0, 1.0)
        out[..., c] = corrected_c * strength + frame[..., c] * (1.0 - strength)
    return out


def _fit_histogram(src: np.ndarray, tgt: np.ndarray, bins: int = 512):
    """
    Per-channel histogram matching.
    Builds a LUT that maps src's CDF onto tgt's CDF.
    This is the most aggressive method and handles non-linear shifts.
    """
    params = {'bins': bins}
    for c in range(3):
        src_hist, _ = np.histogram(src[..., c].ravel(), bins=bins, range=(0.0, 1.0))
        tgt_hist, _ = np.histogram(tgt[..., c].ravel(), bins=bins, range=(0.0, 1.0))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1] + 1e-10
        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
        tgt_cdf /= tgt_cdf[-1] + 1e-10
        # For each src bin-value, find what tgt value has the same CDF
        tgt_vals = np.linspace(0.0, 1.0, bins)
        lut = np.interp(src_cdf, tgt_cdf, tgt_vals).astype(np.float32)
        params[f'lut_{c}'] = lut
    return params


def _apply_histogram(frame: np.ndarray, params: dict, strength: float) -> np.ndarray:
    bins = params['bins']
    out = frame.copy()
    for c in range(3):
        lut = params[f'lut_{c}']
        idx = np.clip((frame[..., c] * (bins - 1)).astype(np.int32), 0, bins - 1)
        corrected_c = np.clip(lut[idx], 0.0, 1.0)
        out[..., c] = corrected_c * strength + frame[..., c] * (1.0 - strength)
    return out


_METHOD_MAP = {
    "reinhard_lab": (_fit_reinhard,   _apply_reinhard),
    "linear_rgb":   (_fit_linear_rgb, _apply_linear_rgb),
    "histogram":    (_fit_histogram,  _apply_histogram),
}

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ChromaPin:
    """
    Pins a processed video's colours to a reference image.

    Feed in any AI-processed video that has drifted from your intended colours,
    point it at the original reference image and the frame index that corresponds
    to it, and ChromaPin will measure the drift once and remove it across the
    whole sequence — without flattening the natural colour variation between frames.

    Workflow
    --------
    1. Identify which frame index in your processed video corresponds to your
       reference image and connect both here.
    2. The node computes the colour-correction transform between the *processed*
       reference frame and your *original* reference image.
    3. That transform is applied to every frame in the video — removing the
       model's colour drift uniformly without disturbing inter-frame variation.

    Propagation modes
    -----------------
    uniform  — Same correction on every frame.  Best when model drift is
               consistent across the whole sequence (the common case).
    falloff  — Correction strength tapers with distance from the anchor frame.
               Useful if you suspect the drift evolves over time, or if you
               only want to lock down a short region around the reference frame.

    Methods
    -------
    reinhard_lab  — Matches per-channel mean & std in CIE L*a*b* space.
                    Best general-purpose method; handles colour casts well.
    linear_rgb    — Matches per-channel mean & std in sRGB.
                    Fastest; good for simple gain/offset shifts.
    histogram     — Full per-channel histogram matching (most aggressive).
                    Best for complex, non-linear colour distributions.
    """

    METHODS     = ["reinhard_lab", "linear_rgb", "histogram"]
    PROPAGATION = ["uniform", "falloff"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {}),
                "reference_image": ("IMAGE", {}),
                "reference_frame_index": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 9999,
                        "tooltip": (
                            "0-based index of the frame in the *video* batch that "
                            "corresponds to the reference_image you are feeding in."
                        ),
                    },
                ),
                "method": (cls.METHODS, {"default": "reinhard_lab"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "Blend between original (0.0) and fully corrected (1.0).",
                    },
                ),
                "propagation": (
                    cls.PROPAGATION,
                    {
                        "default": "uniform",
                        "tooltip": (
                            "uniform  = same correction every frame.\n"
                            "falloff  = correction tapers with distance from the anchor frame."
                        ),
                    },
                ),
            },
            "optional": {
                "falloff_radius": (
                    "INT",
                    {
                        "default": 30, "min": 1, "max": 9999,
                        "tooltip": (
                            "[falloff mode] Number of frames away from the reference "
                            "frame at which the correction reaches zero."
                        ),
                    },
                ),
                "falloff_gamma": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                        "tooltip": (
                            "[falloff mode] Curve shape.  "
                            "1.0 = linear.  >1 = fast initial drop.  <1 = slow initial drop."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("corrected_video", "debug_comparison")
    FUNCTION      = "anchor_color"
    CATEGORY      = "FEnodes"
    DESCRIPTION   = (
        "Pins a processed video's colours to a reference image. "
        "Measures colour drift at a single anchor frame and propagates "
        "the correction across the entire sequence."
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_np(video: torch.Tensor, index: int) -> np.ndarray:
        """Return frame at index as (H, W, 3) float32 numpy [0,1]."""
        index = min(index, video.shape[0] - 1)
        return video[index].cpu().float().numpy()[..., :3]

    @staticmethod
    def _resize_to_match(img_np: np.ndarray, h: int, w: int) -> np.ndarray:
        if img_np.shape[0] == h and img_np.shape[1] == w:
            return img_np
        from PIL import Image as PILImage
        pil = PILImage.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((w, h), PILImage.LANCZOS)
        return np.array(pil).astype(np.float32) / 255.0

    @staticmethod
    def _make_debug(ref_np: np.ndarray,
                    before_np: np.ndarray,
                    after_np: np.ndarray) -> torch.Tensor:
        """
        Three-panel comparison: Reference | Before (processed) | After (corrected).
        Panels are separated by a thin white divider.
        All panels are resized to the same height (ref height).
        """
        H = ref_np.shape[0]
        # Resize panels to match reference height if needed
        def _resize_h(img, target_h):
            if img.shape[0] == target_h:
                return img
            from PIL import Image as PILImage
            pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            ratio = target_h / img.shape[0]
            new_w = max(1, int(img.shape[1] * ratio))
            pil = pil.resize((new_w, target_h), PILImage.LANCZOS)
            return np.array(pil).astype(np.float32) / 255.0

        panels = [_resize_h(p, H) for p in [ref_np[..., :3], before_np[..., :3], after_np[..., :3]]]
        divider = np.ones((H, 4, 3), dtype=np.float32)  # white bar
        row = np.concatenate([panels[0], divider, panels[1], divider, panels[2]], axis=1)
        return torch.from_numpy(row).unsqueeze(0)        # (1, H, 3W+8, 3)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def anchor_color(
        self,
        video: torch.Tensor,
        reference_image: torch.Tensor,
        reference_frame_index: int,
        method: str,
        strength: float,
        propagation: str,
        falloff_radius: int = 30,
        falloff_gamma: float = 1.0,
    ):
        num_frames = video.shape[0]
        ref_idx = min(reference_frame_index, num_frames - 1)

        if ref_idx != reference_frame_index:
            print(
                f"[FEnodes/ChromaPin] Warning: reference_frame_index "
                f"{reference_frame_index} exceeds video length {num_frames}. "
                f"Clamped to {ref_idx}."
            )

        # Processed reference frame (what the model produced at that index)
        src_np = self._frame_np(video, ref_idx)

        # Original reference image (ground-truth colors the user wants)
        tgt_raw = reference_image[0].cpu().float().numpy()[..., :3]
        tgt_np  = self._resize_to_match(tgt_raw, src_np.shape[0], src_np.shape[1])

        print(
            f"[FEnodes/ChromaPin] Fitting '{method}' correction from "
            f"frame {ref_idx}  →  reference_image  "
            f"(src shape {src_np.shape}, tgt shape {tgt_np.shape})"
        )

        # Fit the correction on the anchor pair
        fit_fn, apply_fn = _METHOD_MAP[method]
        params = fit_fn(src_np, tgt_np)

        # Build per-frame strength weights
        if propagation == "uniform":
            weights = [strength] * num_frames
        else:                                         # falloff
            weights = []
            for f in range(num_frames):
                dist = abs(f - ref_idx)
                if dist == 0:
                    w = strength
                elif dist >= falloff_radius:
                    w = 0.0
                else:
                    t = (dist / float(falloff_radius)) ** falloff_gamma
                    w = float(strength * (1.0 - t))
                weights.append(w)

        # Apply frame by frame
        has_alpha     = video.shape[-1] == 4
        out_frames    = []
        for f in range(num_frames):
            frame_np = video[f].cpu().float().numpy()
            rgb_np   = frame_np[..., :3]
            w        = weights[f]

            if abs(w) < 1e-6:
                out_frames.append(video[f].cpu().float())
                continue

            corrected_rgb = apply_fn(rgb_np, params, w)

            if has_alpha:
                result = np.concatenate(
                    [corrected_rgb.astype(np.float32),
                     frame_np[..., 3:4]],
                    axis=-1,
                )
            else:
                result = corrected_rgb.astype(np.float32)

            out_frames.append(torch.from_numpy(result))

        corrected_video = torch.stack(out_frames, dim=0).clamp(0.0, 1.0)

        # Debug panel: Reference | Before | After  (using the anchor frame)
        after_np = corrected_video[ref_idx].numpy()[..., :3]
        debug    = self._make_debug(tgt_np, src_np, after_np)

        print(
            f"[FEnodes/ChromaPin] Done. "
            f"propagation={propagation}, strength={strength:.2f}, "
            f"frames={num_frames}"
        )

        return (corrected_video, debug)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ChromaPin": ChromaPin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChromaPin": "ChromaPin 📌",
}
