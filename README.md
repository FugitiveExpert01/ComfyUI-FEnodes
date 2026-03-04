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

### 🎬 Video Upscale With Model

> Memory-efficient upscaling of every frame in a video batch using a ComfyUI upscale model (ESRGAN, RealESRGAN, etc.).

| Parameter | Type | Description |
|---|---|---|
| `model_name` | upscale_models | Model from your ComfyUI `upscale_models` folder |
| `images` | IMAGE | Input video batch `(F, H, W, C)` |
| `upscale_method` | ENUM | Final resize filter: `nearest-exact`, `bilinear`, `area`, `bicubic` |
| `factor` | FLOAT | Output scale multiplier (e.g. `2.0` = double resolution) |
| `device_strategy` | ENUM | See table below |
| `batch_size` | INT | Frames processed per GPU pass |

**Device strategies:**

| Strategy | Description |
|---|---|
| `auto` | Detects free VRAM and picks the best strategy automatically |
| `keep_loaded` | Model stays on GPU — fastest, requires available VRAM |
| `load_unload_each_frame` | Model moves to GPU per batch then back to CPU — lower peak VRAM |
| `cpu_only` | Runs entirely on CPU — slowest, minimal VRAM usage |

**Outputs:** `upscaled_images (IMAGE)`

**Features:**
- Automatic tiled inference via `comfy.utils.tiled_scale` — no OOM on large frames
- Single `movedim` pre-pass avoids repeated tensor copies per frame
- `torch.no_grad()` throughout for faster inference and lower memory
- Coloured progress bar with elapsed/ETA printed to the ComfyUI console

---

### 🧹 Free Video Memory

> Pass-through node that explicitly flushes GPU memory between heavy pipeline stages.

| Parameter | Type | Description |
|---|---|---|
| `images` | IMAGE | Passed through unchanged |
| `aggressive_cleanup` | ENUM | Also calls `torch.cuda.synchronize()` and allocator cache APIs if available |
| `report_memory` | ENUM | Prints allocated/reserved GB before and after to the console |

**Outputs:** `images (IMAGE)` — identical to input

---

### 🔤 Text List → Batch / Text Batch → List

Bidirectional converters between ComfyUI `LIST` and batched `STRING` types, with optional delimiter joining.

---

## ⚙️ Installation

**1. Clone into your ComfyUI custom nodes folder:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FugitiveExpert01/ComfyUI-FEnodes.git
```

**2. Restart ComfyUI.**

Nodes will appear in the node menu under the **FEnodes** category.  
No additional dependencies beyond what ComfyUI already requires (PyTorch, NumPy, Pillow).

---

## 🎬 Typical Workflows

**Tiled video diffusion:**
```
Load Video → TileSplit → [K-Sampler per tile] → TileMerge → Save Video
```

**Video upscaling:**
```
Load Video → Video Upscale With Model → [Free Video Memory] → Save Video
```

**Combined:**
```
Load Video → TileSplit → [K-Sampler per tile] → TileMerge
          → Video Upscale With Model → Free Video Memory → Save Video
```

---

## 🗺️ Roadmap

| Node | Description | Status |
|---|---|---|
| **TileSplit** | Grid tile splitting for video batches | ✅ Released |
| **TileMerge** | Laplacian pyramid tile reconstruction | ✅ Released |
| **Video Upscale With Model** | Memory-efficient per-frame upscaling | ✅ Released |
| **Free Video Memory** | Explicit VRAM flush between pipeline stages | ✅ Released |
| **Text List → Batch** | LIST to batched STRING conversion | ✅ Released |
| **Text Batch → List** | Batched STRING to LIST conversion | ✅ Released |

Have a node idea or a production use case that isn't covered? **[Open an issue](https://github.com/FugitiveExpert01/ComfyUI-FEnodes/issues).**

---

## 🤝 Contributing

Pull requests are welcome. If you're adding a node, please:

- Keep video batch support (`F, H, W, C`) as the primary tensor convention
- Add a docstring describing what the node does and its input/output types
- Test with both single images and multi-frame video batches
- Set `CATEGORY = "FEnodes"` so nodes appear grouped in the menu

---

## 📄 License

Apache License 2.0 — free to use in personal and commercial VFX pipelines.
