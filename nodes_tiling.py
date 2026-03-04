"""
FEnodes — Tiling nodes for VFX production pipelines.
Author: FugitiveExpert01
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)


class TilingNodeBase:
    """
    Base class providing Laplacian pyramid blending utilities.

    How multi-band blending works:
      Each tile is decomposed into a Laplacian pyramid (detail at each frequency
      band + a coarse residual). The blending mask is simultaneously downsampled
      into a Gaussian pyramid. Each frequency band is blended independently using
      its corresponding mask level — coarse bands blend over a wide transition
      zone, fine detail bands blend over a narrow one. The blended pyramid is
      then collapsed back into the final image.

      This eliminates the classic trade-off of simple alpha blending: wide feather
      radii smear fine detail, narrow ones leave visible seams. Multi-band blending
      avoids both artefacts simultaneously.

    Video optimisation:
      Mask pyramids depend only on tile layout, not on pixel content. They are
      pre-computed once from the first frame and reused across all subsequent
      frames, avoiding redundant work proportional to frame count.
    """

    BLEND_MODES = ["laplacian", "linear"]

    # ------------------------------------------------------------------
    # Tensor normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_4d_BCHW(t, device):
        """
        Force any tensor into (1, C, H, W) float32 on the correct device.
        Handles every shape a video model might produce for a single frame.
        """
        t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 2:
            log.debug("ensure_4d_BCHW: 2D greyscale %s → (1, 1, H, W)", t.shape)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            log.debug("ensure_4d_BCHW: 3D %s → treating as (H, W, C)", t.shape)
            t = t.unsqueeze(0).permute(0, 3, 1, 2)
        elif t.dim() == 4:
            if t.shape[-1] in (1, 3, 4):
                log.debug("ensure_4d_BCHW: 4D channels-last %s → (B, C, H, W)", t.shape)
                t = t.permute(0, 3, 1, 2)
            else:
                log.debug("ensure_4d_BCHW: 4D %s already (B, C, H, W)", t.shape)
        else:
            raise ValueError(f"[FEnodes] ensure_4d_BCHW: unexpected shape {t.shape}")

        if t.dim() != 4:
            raise RuntimeError(f"[FEnodes] ensure_4d_BCHW failed — result shape: {t.shape}")
        return t.contiguous()

    # ------------------------------------------------------------------
    # Pyramid construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_laplacian_pyramid(img, levels):
        """
        Build a Laplacian pyramid from a (1, C, H, W) tensor.

        Returns a list of `levels` tensors:
          indices 0 … levels-2 : Laplacian detail planes, finest → coarsest
          index   levels-1     : coarsest Gaussian residual

        Pure-PyTorch, GPU-resident operation.
        """
        pyramid = []
        current = img
        for _ in range(levels - 1):
            h, w = current.shape[2], current.shape[3]
            down = F.interpolate(current, scale_factor=0.5,
                                 mode='bilinear', align_corners=False)
            up   = F.interpolate(down, size=(h, w),
                                 mode='bilinear', align_corners=False)
            pyramid.append(current - up)   # detail = original − low-pass
            current = down
        pyramid.append(current)            # coarsest Gaussian residual
        return pyramid

    @staticmethod
    def _build_gaussian_pyramid(mask, levels):
        """
        Build a Gaussian pyramid for a (1, 1, H, W) blending mask.

        Returns `levels` tensors from finest to coarsest. Downsampling the
        mask means low-frequency pyramid bands blend over a proportionally
        wider spatial transition than high-frequency bands — the core benefit
        of multi-band blending.
        """
        pyramid = [mask]
        current = mask
        for _ in range(levels - 1):
            current = F.interpolate(current, scale_factor=0.5,
                                    mode='bilinear', align_corners=False)
            pyramid.append(current)
        return pyramid

    @staticmethod
    def _collapse_pyramid(pyramid):
        """
        Reconstruct a (1, C, H, W) image from a blended Laplacian pyramid.

        pyramid: list of tensors, finest detail first, coarsest Gaussian last.
        """
        result = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            h, w = level.shape[2], level.shape[3]
            result = F.interpolate(result, size=(h, w),
                                   mode='bilinear', align_corners=False)
            result = result + level
        return result

    @staticmethod
    def _canvas_pyramid_sizes(orig_h, orig_w, levels):
        """
        Compute canvas (H, W) at each pyramid level.

        Uses the same floor-division that F.interpolate(scale_factor=0.5)
        applies internally, so canvas and tile pyramids stay aligned.
        """
        sizes = []
        h, w = orig_h, orig_w
        for _ in range(levels):
            sizes.append((h, w))
            h = h // 2
            w = w // 2
        return sizes

    # ------------------------------------------------------------------
    # Blending mask
    # ------------------------------------------------------------------

    @staticmethod
    def create_gaussian_mask(h, w, overlap_top, overlap_bottom,
                             overlap_left, overlap_right, device):
        """
        Linear feathering mask — ramps from 0 → 1 across each overlap edge.
        Used as the level-0 mask; pyramid downsampling widens the ramp at
        coarser levels automatically.
        """
        mask = torch.ones((1, 1, h, w), device=device)
        if overlap_top > 0:
            mask[:, :, :overlap_top, :] *= torch.linspace(
                0, 1, overlap_top, device=device).view(1, 1, overlap_top, 1)
        if overlap_bottom > 0:
            mask[:, :, -overlap_bottom:, :] *= torch.linspace(
                1, 0, overlap_bottom, device=device).view(1, 1, overlap_bottom, 1)
        if overlap_left > 0:
            mask[:, :, :, :overlap_left] *= torch.linspace(
                0, 1, overlap_left, device=device).view(1, 1, 1, overlap_left)
        if overlap_right > 0:
            mask[:, :, :, -overlap_right:] *= torch.linspace(
                1, 0, overlap_right, device=device).view(1, 1, 1, overlap_right)
        return mask


# ======================================================================
# TileSplit
# ======================================================================

class TileSplit(TilingNodeBase):
    """Splits an image or video batch into an overlapping grid of tiles."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image":           ("IMAGE",),
                "tiles_x":         ("INT",   {"default": 2,    "min": 1,   "max": 10}),
                "tiles_y":         ("INT",   {"default": 2,    "min": 1,   "max": 10}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES   = ("IMAGE", "IMAGE", "TILE_CALC")
    RETURN_NAMES   = ("tiles", "debug_image", "tile_calc")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION  = "split"
    CATEGORY  = "FEnodes"

    def split(self, image, tiles_x, tiles_y, overlap_percent):
        B, H, W, C = image.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # overlap_percent means exactly "this fraction of the tile width/height
        # is shared with each neighbour."
        #
        # Derivation:
        #   Each interior tile contributes (tile_w × (1 - overlap)) unique pixels.
        #   The last tile contributes tile_w unique pixels.
        #   Total unique pixels = tile_w × ((1 - overlap) × (n-1) + 1) = W
        #   → tile_w = W / ((1 - overlap) × (n-1) + 1)
        #
        # At overlap=0.50, tiles_x=2: tile_w = W / 1.5  (exactly half overlaps)
        # At overlap=0.00, tiles_x=2: tile_w = W / 2    (no overlap)

        tile_w_f = W / ((1.0 - overlap_percent) * (tiles_x - 1) + 1)
        tile_h_f = H / ((1.0 - overlap_percent) * (tiles_y - 1) + 1)

        # Round up to nearest multiple of 8 (model compatibility), clamp to canvas
        tile_w = min(((int(tile_w_f) + 7) // 8) * 8, W)
        tile_h = min(((int(tile_h_f) + 7) // 8) * 8, H)

        # Space tile origins evenly so they cover the full canvas
        start_x = np.linspace(0, W - tile_w, tiles_x, dtype=int) if tiles_x > 1 else [0]
        start_y = np.linspace(0, H - tile_h, tiles_y, dtype=int) if tiles_y > 1 else [0]

        # Actual overlap in pixels after rounding (for the log)
        actual_ov_x = tile_w - int((W - tile_w) / max(tiles_x - 1, 1)) if tiles_x > 1 else 0
        actual_ov_y = tile_h - int((H - tile_h) / max(tiles_y - 1, 1)) if tiles_y > 1 else 0

        log.info(
            "TileSplit: %dx%d px, %d frames — %dx%d grid, "
            "tile %dx%d px, overlap ~%d×%d px (%.0f%% × %.0f%%)",
            W, H, B, tiles_x, tiles_y, tile_w, tile_h,
            actual_ov_x, actual_ov_y,
            (actual_ov_x / tile_w) * 100, (actual_ov_y / tile_h) * 100,
        )

        img_tensor     = image.permute(0, 3, 1, 2).to(device)
        num_tiles      = tiles_x * tiles_y
        tile_sequences = [[] for _ in range(num_tiles)]
        tile_layouts   = []

        # Debug overlay
        debug_pil = Image.fromarray(
            (image[0].cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        overlay   = Image.new("RGBA", debug_pil.size, (0, 0, 0, 120))
        debug_pil = Image.alpha_composite(debug_pil, overlay)
        draw      = ImageDraw.Draw(debug_pil)
        try:
            font = ImageFont.truetype("LiberationSans-Bold.ttf", int(tile_h * 0.15))
        except Exception:
            log.debug("TileSplit: LiberationSans-Bold.ttf not found, using default font")
            font = ImageFont.load_default()

        for b in range(B):
            tile_count = 0
            for y_idx, y in enumerate(start_y):
                for x_idx, x in enumerate(start_x):
                    tile_frame = img_tensor[b:b+1, :, y:y+tile_h, x:x+tile_w]
                    tile_sequences[tile_count].append(
                        tile_frame.permute(0, 2, 3, 1).cpu())

                    if b == 0:
                        ov_l = int(tile_w - (start_x[x_idx] - start_x[x_idx-1])) if x_idx > 0            else 0
                        ov_r = int(tile_w - (start_x[x_idx+1] - start_x[x_idx])) if x_idx < tiles_x - 1 else 0
                        ov_t = int(tile_h - (start_y[y_idx] - start_y[y_idx-1])) if y_idx > 0            else 0
                        ov_b = int(tile_h - (start_y[y_idx+1] - start_y[y_idx])) if y_idx < tiles_y - 1 else 0

                        tile_layouts.append({
                            "x": int(x), "y": int(y),
                            "ov_l": ov_l, "ov_r": ov_r,
                            "ov_t": ov_t, "ov_b": ov_b,
                        })
                        log.debug("TileSplit: tile %d at (%d,%d) overlaps L=%d R=%d T=%d B=%d",
                                  tile_count, x, y, ov_l, ov_r, ov_t, ov_b)

                        color = ((x_idx * 40) % 255, (y_idx * 60) % 255, 255, 100)
                        draw.rectangle([x, y, x + tile_w, y + tile_h],
                                       outline="white", fill=color, width=2)
                        draw.text((x + 10, y + 10), f"Tile {tile_count}",
                                  fill="white", font=font)
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

        log.info("TileSplit: done — %d tile sequences produced", num_tiles)
        return (final_tile_batches, debug_out, tile_calc)


# ======================================================================
# TileMerge
# ======================================================================

class TileMerge(TilingNodeBase):
    """
    Merges processed tiles back into a full image or video sequence.

    Two blending modes are available via the blend_mode dropdown:

    laplacian — Multi-band blending using a Laplacian pyramid.
      Each frequency band is blended independently so coarse bands
      (colour, tone) transition smoothly over a wide zone while fine
      detail bands transition sharply. Eliminates the classic trade-off
      where wide feathering blurs detail and narrow feathering leaves
      visible seams. Best quality; recommended for final output.

    linear — Simple Gaussian-feathered weighted average.
      Fast and straightforward. Overlap regions are cross-faded with a
      linear ramp. Good for quick previews or when tile content is very
      similar and seams are not an issue.
    """

    PYRAMID_LEVELS = 4   # depth of Laplacian decomposition; 4 is standard

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles":      ("IMAGE",),
                "tile_calc":  ("TILE_CALC",),
                "blend_mode": (TilingNodeBase.BLEND_MODES, {"default": "laplacian"}),
            }
        }

    INPUT_IS_LIST  = True
    RETURN_TYPES   = ("IMAGE",)
    FUNCTION       = "merge"
    CATEGORY       = "FEnodes"

    def merge(self, tiles, tile_calc, blend_mode):
        tc         = tile_calc[0]
        # blend_mode arrives as a list because INPUT_IS_LIST = True
        blend_mode = blend_mode[0] if isinstance(blend_mode, list) else blend_mode
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        orig_h     = tc['orig_h']
        orig_w     = tc['orig_w']

        # Unwrap any extra list nesting ComfyUI may add
        tile_list = tiles
        while (isinstance(tile_list, list)
               and len(tile_list) == 1
               and isinstance(tile_list[0], list)):
            tile_list = tile_list[0]

        num_frames = tile_list[0].shape[0]
        log.info(
            "TileMerge: %d tiles × %d frames — %s blend → %dx%d canvas",
            len(tile_list), num_frames, blend_mode, orig_w, orig_h,
        )
        for idx, t in enumerate(tile_list):
            log.debug("TileMerge: tile[%d] shape=%s dtype=%s", idx, t.shape, t.dtype)

        if blend_mode == "laplacian":
            result = self._merge_laplacian(tile_list, tc, device, orig_h, orig_w)
        else:
            result = self._merge_linear(tile_list, tc, device, orig_h, orig_w)

        log.info("TileMerge: done — output shape=%s", result.shape)
        return (result,)

    # ------------------------------------------------------------------
    # Laplacian pyramid blending
    # ------------------------------------------------------------------

    def _merge_laplacian(self, tile_list, tc, device, orig_h, orig_w):
        """
        Multi-band blend: each Laplacian pyramid level is accumulated
        separately and the final image is collapsed from the blended pyramid.

        Mask pyramids are pre-computed once and reused across all frames
        (video optimisation — masks depend on layout, not pixel content).
        """
        levels       = self.PYRAMID_LEVELS
        canvas_sizes = self._canvas_pyramid_sizes(orig_h, orig_w, levels)
        log.debug("TileMerge [laplacian]: canvas pyramid sizes %s", canvas_sizes)

        # Pre-compute mask pyramids
        log.info("TileMerge [laplacian]: pre-computing mask pyramids for %d tiles",
                 len(tc['layouts']))
        mask_pyramids     = []
        cached_tile_sizes = []

        for i, layout in enumerate(tc['layouts']):
            tf_first = self.ensure_4d_BCHW(tile_list[i][0], device)
            th, tw   = tf_first.shape[2], tf_first.shape[3]
            cached_tile_sizes.append((th, tw))
            mask = self.create_gaussian_mask(
                th, tw,
                layout['ov_t'], layout['ov_b'],
                layout['ov_l'], layout['ov_r'],
                device,
            )
            mask_pyramids.append(self._build_gaussian_pyramid(mask, levels))
            log.debug("TileMerge [laplacian]: mask pyramid %d base %dx%d", i, tw, th)

        output_frames = []

        for b in range(tc['batch_size'] if 'batch_size' in tc else tile_list[0].shape[0]):
            accum  = [torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
                      for h, w in canvas_sizes]
            weight = [torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
                      for h, w in canvas_sizes]

            for i, layout in enumerate(tc['layouts']):
                raw_frame  = tile_list[i][b]
                tile_frame = self.ensure_4d_BCHW(raw_frame, device)
                th, tw     = tile_frame.shape[2], tile_frame.shape[3]

                log.debug("TileMerge [laplacian]: frame %d tile %d shape=%s", b, i, tile_frame.shape)

                # Rebuild mask pyramid if tile size changed (e.g. model resized it)
                if (th, tw) != cached_tile_sizes[i]:
                    log.warning(
                        "TileMerge: tile %d changed size at frame %d "
                        "(expected %s, got %s) — rebuilding mask pyramid",
                        i, b, cached_tile_sizes[i], (th, tw),
                    )
                    mask = self.create_gaussian_mask(
                        th, tw,
                        layout['ov_t'], layout['ov_b'],
                        layout['ov_l'], layout['ov_r'],
                        device,
                    )
                    tile_mask_pyr = self._build_gaussian_pyramid(mask, levels)
                else:
                    tile_mask_pyr = mask_pyramids[i]

                tile_pyr = self._build_laplacian_pyramid(tile_frame, levels)

                for l in range(levels):
                    # Scale tile origin to this pyramid level via right-shift
                    # (floor division by 2^l, matching F.interpolate behaviour)
                    x_l = layout['x'] >> l
                    y_l = layout['y'] >> l
                    canvas_h, canvas_w = canvas_sizes[l]
                    h_sl = min(tile_pyr[l].shape[2], canvas_h - y_l)
                    w_sl = min(tile_pyr[l].shape[3], canvas_w - x_l)

                    if h_sl <= 0 or w_sl <= 0:
                        log.debug("TileMerge: tile %d level %d out of canvas — skipping", i, l)
                        continue

                    m = tile_mask_pyr[l][:, :, :h_sl, :w_sl]
                    accum [l][:, :, y_l:y_l+h_sl, x_l:x_l+w_sl] += tile_pyr[l][:, :, :h_sl, :w_sl] * m
                    weight[l][:, :, y_l:y_l+h_sl, x_l:x_l+w_sl] += m

            blended_pyr = [accum[l] / (weight[l] + 1e-8) for l in range(levels)]
            frame_out   = self._collapse_pyramid(blended_pyr)
            output_frames.append(frame_out.permute(0, 2, 3, 1).cpu())

        return torch.cat(output_frames, dim=0).clamp(0, 1)

    # ------------------------------------------------------------------
    # Linear (Gaussian feather) blending
    # ------------------------------------------------------------------

    def _merge_linear(self, tile_list, tc, device, orig_h, orig_w):
        """
        Simple weighted-average blend using a linear feathering mask.
        Fast; good for previews or low-overlap grids.
        """
        output_frames = []

        for b in range(tile_list[0].shape[0]):
            accum  = torch.zeros((1, 3, orig_h, orig_w), device=device, dtype=torch.float32)
            weight = torch.zeros((1, 1, orig_h, orig_w), device=device, dtype=torch.float32)

            for i, layout in enumerate(tc['layouts']):
                tile_frame = self.ensure_4d_BCHW(tile_list[i][b], device)
                th, tw     = tile_frame.shape[2], tile_frame.shape[3]
                log.debug("TileMerge [linear]: frame %d tile %d shape=%s", b, i, tile_frame.shape)

                mask = self.create_gaussian_mask(
                    th, tw,
                    layout['ov_t'], layout['ov_b'],
                    layout['ov_l'], layout['ov_r'],
                    device,
                )

                x, y = layout['x'], layout['y']
                h_sl = min(th, orig_h - y)
                w_sl = min(tw, orig_w - x)

                accum [:, :, y:y+h_sl, x:x+w_sl] += tile_frame[:, :, :h_sl, :w_sl] * mask[:, :, :h_sl, :w_sl]
                weight[:, :, y:y+h_sl, x:x+w_sl] += mask[:, :, :h_sl, :w_sl]

            output_frames.append((accum / (weight + 1e-8)).permute(0, 2, 3, 1).cpu())

        return torch.cat(output_frames, dim=0).clamp(0, 1)


# ======================================================================
# ComfyUI registration
# ======================================================================

NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "TileSplit",
    "TileMerge": "TileMerge",
}
