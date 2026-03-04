# ComfyUI VFX Nodes
### by [FugitiveExpert01](https://github.com/FugitiveExpert01)

> A growing collection of custom ComfyUI nodes built for VFX production pipelines — designed around the real demands of working with high-resolution imagery, video sequences, and AI-assisted visual effects workflows.

---

## 📌 Overview

This repo fills the gaps between off-the-shelf ComfyUI nodes and the specific needs of VFX work: large format images, temporal consistency across video frames, precise spatial control, and clean integration with diffusion-based upscaling and enhancement models.

Nodes are built with **video batch support as a first-class concern** — not an afterthought.

---

## 🧩 Nodes

### 🔲 TileSplit
> Splits an image or video batch into an overlapping grid of tiles, ready to be passed individually to a model.

| Parameter | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image or video batch |
| `tiles_x` | INT | Number of columns |
| `tiles_y` | INT | Number of rows |
| `overlap_percent` | FLOAT | Overlap between adjacent tiles |

**Outputs:** `tiles (LIST)` · `debug_image` · `tile_calc`

**Features:**
- Overlap calculated automatically and stored for seamless reconstruction
- Outputs tiles as `(F, H, W, C)` tensors — compatible with video model K-samplers
- Debug image showing tile boundaries and overlap regions
- Tile dimensions snapped to multiples of 8 for model compatibility

---

### 🔳 TileMerge
> Reconstructs a full image or video sequence from processed tiles using Laplacian Pyramid blending for seamless, artifact-free joins.

| Parameter | Type | Description |
|---|---|---|
| `tiles` | IMAGE (LIST) | Processed tile sequences |
| `tile_calc` | TILE_CALC | Layout data from TileSplit |

**Outputs:** `image`

**Features:**
- Gaussian feathering masks at overlap regions
- Laplacian frequency blending for natural-looking seams
- Robust handling of tensor shapes returned by video models
- Weighted accumulation handles overlapping regions correctly

---

## ⚙️ Installation

**1. Clone into your ComfyUI custom nodes folder:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FugitiveExpert01/ComfyUI-FEnodes.git
```

**2. Restart ComfyUI.**

Nodes will appear in the node menu. No additional dependencies beyond what ComfyUI already requires (PyTorch, NumPy, Pillow).

---

## 🎬 Typical Workflow

```
Load Image/Video → TileSplit → [Video Model / K-Sampler per tile] → TileMerge → Save Image/Video
```

`TileSplit` outputs a list of tile sequences so each tile can be routed through a video model independently, then `TileMerge` reconstructs the full frame. This allows processing of resolutions that would otherwise exceed VRAM limits.

---

## 🗺️ Roadmap

> This repo is actively developed alongside VFX production work.

| Node | Description | Status |
|---|---|---|
| **TileSplit** | Grid tile splitting for video batches | ✅ Released |
| **TileMerge** | Laplacian pyramid tile reconstruction | ✅ Released |

Have a node idea or a production use case that isn't covered? **[Open an issue](https://github.com/FugitiveExpert01/comfyui-vfx-nodes/issues).**

---

## 🤝 Contributing

Pull requests are welcome. If you're adding a node, please:

- Keep video batch support (`F, H, W, C`) as the primary tensor convention
- Add a docstring describing what the node does and its input/output types
- Test with both single images and multi-frame video batches

---

## 📄 License

Apache License 2.0 — free to use in personal and commercial VFX pipelines.
