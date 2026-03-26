"""
FEnodes — Tiling nodes for VFX production pipelines.
Author: FugitiveExpert01
Version: v0.0.6
"""
 
__version__ = "v0.0.6"
 
import math
import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
 
try:
    from server import PromptServer
except ImportError:
    PromptServer = None
 
 
class TilingNodeBase:
    """Base utility for tile masking and tensor shape normalisation."""
 
    @staticmethod
    def ensure_4d_BCHW(t, device):
        t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            t = t.unsqueeze(0).permute(0, 3, 1, 2)
        elif t.dim() == 4:
            if t.shape[-1] in (1, 3, 4):
                t = t.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"[TileMerge] ensure_4d_BCHW: unexpected shape {t.shape}")
        assert t.dim() == 4, f"[TileMerge] ensure_4d_BCHW failed, result shape: {t.shape}"
        return t.contiguous()
 
    @staticmethod
    def create_feather_mask(h, w, ov_top, ov_bottom, ov_left, ov_right, feather_scale, device):
        """
        Linear feather mask.  feather_scale multiplies each overlap zone so the
        user can widen or tighten the fade without touching TileSplit's overlap_percent.
        Values are clamped so the fade never exceeds half the tile dimension.
        """
        ov_top    = min(int(ov_top    * feather_scale), h // 2)
        ov_bottom = min(int(ov_bottom * feather_scale), h // 2)
        ov_left   = min(int(ov_left   * feather_scale), w // 2)
        ov_right  = min(int(ov_right  * feather_scale), w // 2)
 
        mask = torch.ones((1, 1, h, w), device=device)
        if ov_top > 0:
            mask[:, :, :ov_top, :]     *= torch.linspace(0, 1, ov_top,    device=device).view(1, 1, ov_top,    1)
        if ov_bottom > 0:
            mask[:, :, -ov_bottom:, :] *= torch.linspace(1, 0, ov_bottom, device=device).view(1, 1, ov_bottom, 1)
        if ov_left > 0:
            mask[:, :, :, :ov_left]    *= torch.linspace(0, 1, ov_left,   device=device).view(1, 1, 1, ov_left)
        if ov_right > 0:
            mask[:, :, :, -ov_right:]  *= torch.linspace(1, 0, ov_right,  device=device).view(1, 1, 1, ov_right)
        return mask
 
    @staticmethod
    def _send_text(unique_id, html):
        """Push a progress text update to the node footer in the ComfyUI frontend."""
        if unique_id is not None and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(html, unique_id)
            except Exception:
                pass
 
 
ALIGNMENT_OPTIONS = ["Free", "8 (SD)", "16 (WAN / VACE)"]
 
 
class TileSplit(TilingNodeBase):
    """Splits an image or video batch into a list of tile batches."""
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":           ("IMAGE",),
                "use_pixel_size":  ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Switch between grid mode (tiles_x / tiles_y) and pixel size mode (tile_width / tile_height).",
                }),
                "tiles_x":         ("INT",   {"default": 2,    "min": 1,   "max": 10}),
                "tiles_y":         ("INT",   {"default": 2,    "min": 1,   "max": 10}),
                "tile_width":      ("INT",   {"default": 768,  "min": 64,  "max": 8192, "step": 8,
                    "tooltip": "Target tile width in pixels. Grid is derived via ceil(canvas / tile_px). Only active when use_pixel_size is enabled."}),
                "tile_height":     ("INT",   {"default": 768,  "min": 64,  "max": 8192, "step": 8,
                    "tooltip": "Target tile height in pixels. Grid is derived via ceil(canvas / tile_px). Only active when use_pixel_size is enabled."}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "alignment": (ALIGNMENT_OPTIONS, {
                    "default": "Free",
                    "tooltip": (
                        "Free: no snapping. "
                        "8 (SD): snap tile dimensions to multiples of 8 for SD 1.5 / SDXL. "
                        "16 (WAN / VACE): snap to multiples of 16 — required to avoid token "
                        "count mismatches inside VACE attention blocks."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
 
    RETURN_TYPES   = ("IMAGE", "IMAGE", "TILE_CALC")
    RETURN_NAMES   = ("tiles", "debug_image", "tile_calc")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION       = "split"
    CATEGORY       = "FEnodes"
 
    def split(self, image, use_pixel_size, tiles_x, tiles_y, tile_width, tile_height, overlap_percent, alignment="Free", unique_id=None):
        B, H, W, C = image.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        align_map = {"Free": None, "8 (SD)": 8, "16 (WAN / VACE)": 16}
        align = align_map.get(alignment, None)
 
        # ── Grid resolution ──────────────────────────────────────────────────
        if use_pixel_size:
            tiles_x = max(1, math.ceil(W / tile_width))
            tiles_y = max(1, math.ceil(H / tile_height))
 
        stride_w = W / tiles_x
        stride_h = H / tiles_y
 
        # overlap_percent is a fraction of the full canvas, so the safe upper
        # limit is 1/tiles (beyond that the overlap zone exceeds one stride and
        # pixels get triple-covered).
        max_overlap = 1.0 / max(tiles_x, tiles_y)
        if overlap_percent > max_overlap:
            print(
                f"[FEnodes/TileSplit] overlap_percent {overlap_percent:.2f} exceeds safe "
                f"maximum {max_overlap:.2f} for a {tiles_y}×{tiles_x} grid — clamping."
            )
            overlap_percent = max_overlap
 
        raw_w = stride_w + (W * overlap_percent)
        raw_h = stride_h + (H * overlap_percent)
 
        if align is not None:
            tile_w = int(np.ceil(raw_w / align) * align)
            tile_h = int(np.ceil(raw_h / align) * align)
        else:
            tile_w = int(np.ceil(raw_w))
            tile_h = int(np.ceil(raw_h))
 
        tile_w = min(tile_w, W)
        tile_h = min(tile_h, H)
 
        start_x = np.linspace(0, W - tile_w, tiles_x, dtype=int) if tiles_x > 1 else np.array([0])
        start_y = np.linspace(0, H - tile_h, tiles_y, dtype=int) if tiles_y > 1 else np.array([0])
 
        print(
            f"[FEnodes/TileSplit] canvas={H}x{W}  grid={tiles_y}x{tiles_x}  "
            f"alignment={alignment}  tile={tile_h}x{tile_w}  overlap={overlap_percent:.0%}"
        )
 
        img_tensor     = image.permute(0, 3, 1, 2).to(device)
        num_tiles      = tiles_x * tiles_y
        tile_sequences = [[] for _ in range(num_tiles)]
        tile_layouts   = []
 
        debug_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        overlay   = Image.new("RGBA", debug_pil.size, (0, 0, 0, 120))
        debug_pil = Image.alpha_composite(debug_pil, overlay)
        draw      = ImageDraw.Draw(debug_pil)
        try:
            font = ImageFont.truetype("LiberationSans-Bold.ttf", int(tile_h * 0.15))
        except Exception:
            font = ImageFont.load_default()
 
        for b in range(B):
            tile_count = 0
            for y_idx, y in enumerate(start_y):
                for x_idx, x in enumerate(start_x):
                    tile_frame = img_tensor[b:b+1, :, y:y+tile_h, x:x+tile_w]
                    tile_sequences[tile_count].append(tile_frame.permute(0, 2, 3, 1).cpu())
 
                    if b == 0:
                        ov_l = (start_x[x_idx] - start_x[x_idx-1]) if x_idx > 0 else 0
                        ov_r = (tile_w - (start_x[x_idx+1] - start_x[x_idx])) if x_idx < tiles_x - 1 else 0
                        ov_t = (start_y[y_idx] - start_y[y_idx-1]) if y_idx > 0 else 0
                        ov_b = (tile_h - (start_y[y_idx+1] - start_y[y_idx])) if y_idx < tiles_y - 1 else 0
 
                        tile_layouts.append({
                            "x":    int(x), "y":    int(y),
                            "ov_l": int(tile_w - ov_l) if x_idx > 0 else 0, "ov_r": int(ov_r),
                            "ov_t": int(tile_h - ov_t) if y_idx > 0 else 0, "ov_b": int(ov_b),
                        })
                        color = ((x_idx * 40) % 255, (y_idx * 60) % 255, 255, 100)
                        draw.rectangle([x, y, x + tile_w, y + tile_h], outline="white", fill=color, width=2)
                        draw.text((x + 10, y + 10), f"Tile {tile_count}", fill="white", font=font)
                    tile_count += 1
 
        final_tile_batches = [torch.cat(seq, dim=0) for seq in tile_sequences]
        debug_out = torch.from_numpy(
            np.array(debug_pil.convert("RGB")).astype(np.float32) / 255.0
        ).unsqueeze(0)
 
        tile_calc = {
            "orig_w": W, "orig_h": H,
            "tile_w": tile_w, "tile_h": tile_h,
            "batch_size": B,
            "num_tiles": num_tiles,
            "layouts": tile_layouts,
        }
 
        # ── Footer ────────────────────────────────────────────────────────────
        tile_mb  = (B * tile_h * tile_w * C * 4) / (1024 ** 2)
        total_mb = tile_mb * num_tiles
        self._send_text(unique_id, (
            f"<tr><td>Tiles: </td>"
            f"<td><b>{num_tiles}</b> × <b>{B}</b>×<b>{tile_h}</b>×<b>{tile_w}</b> "
            f"| <b>{total_mb:.0f} MB</b></td></tr>"
        ))
 
        return (final_tile_batches, debug_out, tile_calc)
 
 
class TileMerge(TilingNodeBase):
    """Merges tile batches back into a single image / video sequence."""
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles":     ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                "feather_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": (
                        "Scales the feather (fade) zone relative to the overlap set in TileSplit. "
                        "1.0 = fade across exactly the overlap region. "
                        "0.5 = tighter, harder edge. 2.0 = wider, softer blend."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
 
    INPUT_IS_LIST = True
    RETURN_TYPES  = ("IMAGE",)
    FUNCTION      = "merge"
    CATEGORY      = "FEnodes"
 
    def merge(self, tiles, tile_calc, feather_scale, unique_id=None):
        tc            = tile_calc[0]
        feather_scale = feather_scale[0] if isinstance(feather_scale, list) else feather_scale
        unique_id     = unique_id[0] if isinstance(unique_id, list) else unique_id
        device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        # Unwrap nested list if ComfyUI double-wraps the tile list
        tile_list = tiles
        while isinstance(tile_list, list) and len(tile_list) == 1 and isinstance(tile_list[0], list):
            tile_list = tile_list[0]
 
        print(f"[FEnodes/TileMerge] tiles={len(tile_list)}  canvas={tc['orig_h']}x{tc['orig_w']}  feather_scale={feather_scale}")
        for idx, t in enumerate(tile_list):
            print(f"[FEnodes/TileMerge]   tile[{idx}] shape={t.shape} dtype={t.dtype}")
 
        num_frames    = tile_list[0].shape[0]
        output_frames = []
 
        for b in range(num_frames):
            accum  = torch.zeros((1, 3, tc['orig_h'], tc['orig_w']), device=device, dtype=torch.float32)
            weight = torch.zeros((1, 1, tc['orig_h'], tc['orig_w']), device=device, dtype=torch.float32)
 
            for i, layout in enumerate(tc['layouts']):
                tile_frame = self.ensure_4d_BCHW(tile_list[i][b], device)
 
                # Resize if the model returned a slightly different spatial size
                expected_h, expected_w = tc['tile_h'], tc['tile_w']
                if tile_frame.shape[2] != expected_h or tile_frame.shape[3] != expected_w:
                    print(
                        f"[FEnodes/TileMerge]   resizing tile {i} "
                        f"{tile_frame.shape[2]}x{tile_frame.shape[3]} → {expected_h}x{expected_w}"
                    )
                    tile_frame = F.interpolate(
                        tile_frame, size=(expected_h, expected_w),
                        mode='bilinear', align_corners=False
                    )
 
                curr_h, curr_w = tile_frame.shape[2], tile_frame.shape[3]
                mask = self.create_feather_mask(
                    curr_h, curr_w,
                    layout['ov_t'], layout['ov_b'],
                    layout['ov_l'], layout['ov_r'],
                    feather_scale, device
                )
 
                x, y    = layout['x'], layout['y']
                h_slice = min(curr_h, tc['orig_h'] - y)
                w_slice = min(curr_w, tc['orig_w'] - x)
 
                accum [:, :, y:y+h_slice, x:x+w_slice] += tile_frame[:, :, :h_slice, :w_slice] * mask[:, :, :h_slice, :w_slice]
                weight[:, :, y:y+h_slice, x:x+w_slice] += mask[:, :, :h_slice, :w_slice]
 
            output_frames.append((accum / (weight + 1e-8)).permute(0, 2, 3, 1).cpu())
 
        output = torch.cat(output_frames, dim=0).clamp(0, 1)
 
        # ── Footer ────────────────────────────────────────────────────────────
        B, H, W, C = output.shape
        out_mb = (output.numel() * output.element_size()) / (1024 ** 2)
        self._send_text(unique_id, (
            f"<tr><td>Output: </td>"
            f"<td><b>{B}</b>×<b>{H}</b>×<b>{W}</b>×<b>{C}</b> "
            f"| <b>{out_mb:.2f} MB</b></td></tr>"
        ))
 
        return (output,)
 
 
NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "TileSplit",
    "TileMerge": "TileMerge",
}
