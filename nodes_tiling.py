"""
FEnodes — Tiling nodes for VFX production pipelines.
Author: FugitiveExpert01
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TilingNodeBase:
    """Base utility for Laplacian Frequency Blending and Grid Math."""

    @staticmethod
    def ensure_4d_BCHW(t, device):
        """
        Force any tensor into (1, C, H, W) float32 on the correct device.
        Handles every shape a video model might produce for a single frame.
        """
        t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 2:
            # (H, W) — greyscale, add batch + channel
            t = t.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
        elif t.dim() == 3:
            # Could be (H, W, C) channels-last or (C, H, W) channels-first
            # ComfyUI convention is always channels-last, so treat as (H, W, C)
            t = t.unsqueeze(0)                        # (1, H, W, C)
            t = t.permute(0, 3, 1, 2)                # (1, C, H, W)
        elif t.dim() == 4:
            # Could be (1, H, W, C) or (1, C, H, W)
            # If the last dim looks like a channel count it's channels-last
            if t.shape[-1] in (1, 3, 4):
                t = t.permute(0, 3, 1, 2)            # (1, C, H, W)
            # else already (1, C, H, W)
        else:
            raise ValueError(f"[TileMerge] ensure_4d_BCHW: unexpected shape {t.shape}")

        assert t.dim() == 4, f"[TileMerge] ensure_4d_BCHW failed, result shape: {t.shape}"
        return t.contiguous()

    @staticmethod
    def get_pyramid(img, levels=4):
        """img must be (1, C, H, W)."""
        assert img.dim() == 4, f"get_pyramid expects 4D tensor, got {img.shape}"
        pyramid = []
        current = img
        for i in range(levels - 1):
            orig_h, orig_w = current.shape[2], current.shape[3]
            nxt = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
            up  = F.interpolate(nxt, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pyramid.append(current - up)
            current = nxt
        pyramid.append(current)
        return pyramid

    @staticmethod
    def create_gaussian_mask(h, w, overlap_top, overlap_bottom, overlap_left, overlap_right, device):
        mask = torch.ones((1, 1, h, w), device=device)
        if overlap_top > 0:
            mask[:, :, :overlap_top, :] *= torch.linspace(0, 1, overlap_top, device=device).view(1, 1, overlap_top, 1)
        if overlap_bottom > 0:
            mask[:, :, -overlap_bottom:, :] *= torch.linspace(1, 0, overlap_bottom, device=device).view(1, 1, overlap_bottom, 1)
        if overlap_left > 0:
            mask[:, :, :, :overlap_left] *= torch.linspace(0, 1, overlap_left, device=device).view(1, 1, 1, overlap_left)
        if overlap_right > 0:
            mask[:, :, :, -overlap_right:] *= torch.linspace(1, 0, overlap_right, device=device).view(1, 1, 1, overlap_right)
        return mask


# Alignment options exposed to the user as a dropdown
ALIGNMENT_OPTIONS = ["Free", "8 (SD)", "16 (WAN / VACE)"]


class TileSplit(TilingNodeBase):
    """Splits Video Batches into a List of Video Batches for FEnodes."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 10}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 10}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "alignment": (ALIGNMENT_OPTIONS, {
                    "default": "Free",
                    "tooltip": (
                        "Free: no snapping, tile size is exactly stride × (1 + overlap). "
                        "8 (SD): snap to nearest multiple of 8 for SD 1.5 / SDXL. "
                        "16 (WAN / VACE): snap to nearest multiple of 16 for WAN 2.1 / VACE "
                        "(required to avoid token count mismatches in VACE blocks)."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "TILE_CALC")
    RETURN_NAMES = ("tiles", "debug_image", "tile_calc")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "split"
    CATEGORY = "FEnodes"

    def split(self, image, tiles_x, tiles_y, overlap_percent, alignment="Free"):
        B, H, W, C = image.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve alignment value from dropdown label
        align_map = {"Free": None, "8 (SD)": 8, "16 (WAN / VACE)": 16}
        align = align_map.get(alignment, None)

        # ── Tile sizing ────────────────────────────────────────────────────────
        # Base size = canvas / tiles (stride).
        # Overlap expands each tile beyond the stride so neighbours share pixels.
        # Alignment then snaps UP to the required multiple (or leaves as-is).
        stride_w = W / tiles_x
        stride_h = H / tiles_y
        raw_w = stride_w * (1.0 + overlap_percent)
        raw_h = stride_h * (1.0 + overlap_percent)

        if align is not None:
            tile_w = int(np.ceil(raw_w / align) * align)
            tile_h = int(np.ceil(raw_h / align) * align)
        else:
            tile_w = int(np.ceil(raw_w))
            tile_h = int(np.ceil(raw_h))

        # Clamp: tile must never exceed canvas
        tile_w = min(tile_w, W)
        tile_h = min(tile_h, H)

        # Spread tile start positions evenly across the canvas.
        # linspace(0, W-tile_w) keeps the last tile flush with the right edge.
        start_x = np.linspace(0, W - tile_w, tiles_x, dtype=int) if tiles_x > 1 else np.array([0])
        start_y = np.linspace(0, H - tile_h, tiles_y, dtype=int) if tiles_y > 1 else np.array([0])

        print(
            f"[FEnodes/TileSplit] canvas={H}×{W}  grid={tiles_y}×{tiles_x}  "
            f"alignment={alignment}  tile={tile_h}×{tile_w}  overlap={overlap_percent:.0%}"
        )

        img_tensor = image.permute(0, 3, 1, 2).to(device)
        num_tiles = tiles_x * tiles_y
        tile_sequences = [[] for _ in range(num_tiles)]
        tile_layouts = []

        debug_pil = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", debug_pil.size, (0, 0, 0, 120))
        debug_pil = Image.alpha_composite(debug_pil, overlay)
        draw = ImageDraw.Draw(debug_pil)
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
                            "x": int(x), "y": int(y),
                            "ov_l": int(tile_w - ov_l) if x_idx > 0 else 0, "ov_r": int(ov_r),
                            "ov_t": int(tile_h - ov_t) if y_idx > 0 else 0, "ov_b": int(ov_b)
                        })
                        color = ((x_idx * 40) % 255, (y_idx * 60) % 255, 255, 100)
                        draw.rectangle([x, y, x + tile_w, y + tile_h], outline="white", fill=color, width=2)
                        draw.text((x + 10, y + 10), f"Tile {tile_count}", fill="white", font=font)
                    tile_count += 1

        final_tile_batches = [torch.cat(seq, dim=0) for seq in tile_sequences]
        debug_out = torch.from_numpy(np.array(debug_pil.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
        tile_calc = {
            "orig_w": W, "orig_h": H,
            "tile_w": tile_w, "tile_h": tile_h,
            "batch_size": B,
            "num_tiles": num_tiles,
            "layouts": tile_layouts
        }

        return (final_tile_batches, debug_out, tile_calc)


class TileMerge(TilingNodeBase):
    """Merges Video Batches back into a single image/video sequence."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "FEnodes"

    def merge(self, tiles, tile_calc):
        tc = tile_calc[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        levels = 4

        tile_list = tiles
        while isinstance(tile_list, list) and len(tile_list) == 1 and isinstance(tile_list[0], list):
            tile_list = tile_list[0]

        print(f"[FEnodes/TileMerge] number of tiles received: {len(tile_list)}")
        for idx, t in enumerate(tile_list):
            print(f"[FEnodes/TileMerge]   tile[{idx}] shape={t.shape} dtype={t.dtype}")

        num_frames = tile_list[0].shape[0]
        print(f"[FEnodes/TileMerge] num_frames={num_frames}  canvas={tc['orig_h']}x{tc['orig_w']}")

        output_frames = []

        for b in range(num_frames):
            accum  = torch.zeros((1, 3, tc['orig_h'], tc['orig_w']), device=device, dtype=torch.float32)
            weight = torch.zeros((1, 1, tc['orig_h'], tc['orig_w']), device=device, dtype=torch.float32)

            for i, layout in enumerate(tc['layouts']):
                raw_frame  = tile_list[i][b]
                print(f"[FEnodes/TileMerge]   frame b={b} tile={i} raw shape={raw_frame.shape}")

                tile_frame = self.ensure_4d_BCHW(raw_frame, device)
                print(f"[FEnodes/TileMerge]   frame b={b} tile={i} after ensure_4d shape={tile_frame.shape}")

                # ── Size guard ────────────────────────────────────────────────
                # Video models (e.g. WAN) may return frames at a slightly
                # different spatial resolution due to internal temporal padding.
                # Resize back to expected tile dimensions so the gaussian mask
                # and canvas placement are always correct.
                expected_h, expected_w = tc['tile_h'], tc['tile_w']
                if tile_frame.shape[2] != expected_h or tile_frame.shape[3] != expected_w:
                    print(
                        f"[FEnodes/TileMerge]   resizing tile {i} from "
                        f"{tile_frame.shape[2]}×{tile_frame.shape[3]} → "
                        f"{expected_h}×{expected_w}"
                    )
                    tile_frame = F.interpolate(
                        tile_frame, size=(expected_h, expected_w),
                        mode='bilinear', align_corners=False
                    )

                pyramid = self.get_pyramid(tile_frame, levels)
                current_recon = pyramid[-1]
                for lvl in reversed(range(levels - 1)):
                    target_h, target_w = pyramid[lvl].shape[2], pyramid[lvl].shape[3]
                    current_recon = F.interpolate(current_recon, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    current_recon += pyramid[lvl]

                curr_h, curr_w = current_recon.shape[2], current_recon.shape[3]
                mask = self.create_gaussian_mask(
                    curr_h, curr_w,
                    layout['ov_t'], layout['ov_b'],
                    layout['ov_l'], layout['ov_r'],
                    device
                )

                x, y = layout['x'], layout['y']
                h_slice = min(curr_h, tc['orig_h'] - y)
                w_slice = min(curr_w, tc['orig_w'] - x)

                accum [:, :, y:y+h_slice, x:x+w_slice] += current_recon[:, :, :h_slice, :w_slice] * mask[:, :, :h_slice, :w_slice]
                weight[:, :, y:y+h_slice, x:x+w_slice] += mask[:, :, :h_slice, :w_slice]

            output_frames.append((accum / (weight + 1e-8)).permute(0, 2, 3, 1).cpu())

        return (torch.cat(output_frames, dim=0).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "TileSplit",
    "TileMerge": "TileMerge",
