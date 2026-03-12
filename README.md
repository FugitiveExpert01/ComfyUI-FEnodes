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
| `overlap_percent` | FLOAT | Overlap between adjacent tiles as a fraction of tile stride |
| `alignment` | ENUM | `Free` · `8 (SD)` · `16 (WAN / VACE)` — snaps tile dimensions to the selected multiple |
 
**Outputs:** `tiles (LIST)` · `debug_image` · `tile_calc`
 
**Features:**
- Overlap calculated automatically and stored in `tile_calc` for seamless reconstruction
- Outputs tiles as `(F, H, W, C)` tensors — compatible with video model K-samplers
- Debug image showing tile boundaries and overlap regions
- Alignment dropdown snaps tile dimensions to multiples of 8 (SD 1.5 / SDXL) or 16 (WAN 2.1 / VACE), preventing token count mismatches inside attention blocks
- Node footer reports tile count, per-tile tensor dimensions, and total memory footprint
 
---
 
### 🔳 TileMerge
 
> Reconstructs a full image or video sequence from processed tiles using linear weighted blending for seamless, artifact-free joins.
 
| Parameter | Type | Description |
|---|---|---|
| `tiles` | IMAGE (LIST) | Processed tile sequences |
| `tile_calc` | TILE_CALC | Layout data from TileSplit |
| `feather_scale` | FLOAT | Scales the blend zone independently of `overlap_percent`. `1.0` = fade across the full overlap. `0.5` = tighter edge. `2.0` = wider, softer blend |
 
**Outputs:** `image`
 
**Features:**
- Linear feather masks at overlap regions — width controlled independently via `feather_scale`
- Weighted accumulation handles overlapping regions correctly
- Robust handling of tensor shapes returned by video models, including automatic resize if a model returns a slightly different spatial size
- Node footer reports output tensor dimensions and memory size
 
---
 
### 📌 ChromaPin
 
> Pins a processed video's colours to a reference image by measuring colour drift at a single anchor frame and propagating the correction across the entire sequence.
 
The core workflow: supply the *original* reference image and tell ChromaPin which frame in the processed video corresponds to it. ChromaPin fits a colour-correction transform from the processed anchor frame to the reference, then applies that same transform to every frame — removing the model's colour drift uniformly without disturbing the natural colour variation between frames.
 
| Parameter | Type | Description |
|---|---|---|
| `video` | IMAGE | Processed video batch `(F, H, W, C)` |
| `reference_image` | IMAGE | Original reference image to correct towards |
| `reference_frame_index` | INT | 0-based frame index that corresponds to `reference_image` |
| `method` | ENUM | Colour transfer algorithm (see table below) |
| `strength` | FLOAT | Blend between original (`0.0`) and fully corrected (`1.0`) |
| `propagation` | ENUM | `uniform` or `falloff` — how the correction spreads across frames |
| `falloff_radius` | INT | *(falloff only)* Frames from the anchor at which strength reaches zero |
| `falloff_gamma` | FLOAT | *(falloff only)* Curve shape: `1.0` = linear, `>1` = fast drop, `<1` = slow drop |
 
**Outputs:** `corrected_video` · `debug_comparison` (three-panel: Reference / Before / After)
 
**Methods:**
 
| Method | Deps | Description |
|---|---|---|
| `mkl` | — | Monge-Kantorovich Linearization. Full 3×3 cross-channel Lab transform. **Recommended default** |
| `reinhard_lab` | — | Per-channel mean/std in CIE Lab. Good general purpose |
| `linear_rgb` | — | Per-channel gain/offset in sRGB. Fastest |
| `histogram` | — | Per-channel CDF matching. Best for non-linear shifts |
| `reinhard_lab_gpu` | `kornia` | GPU-accelerated Reinhard; falls back to CPU if kornia is absent |
| `hm-mkl-hm` | `color-matcher` | HM → MKL → HM compound. Best overall quality |
| `hm-mvgd-hm` | `color-matcher` | HM → MVGD → HM compound |
| `hm` | `color-matcher` | Histogram matching |
| `mvgd` | `color-matcher` | Multi-Variate Gaussian Distribution transfer |
 
---
 
### ⚡ LoRA Load
 
> Multi-LoRA loader with a custom folder-tree browser UI. Add any number of LoRAs from a searchable tree, toggle each on/off, and set per-LoRA model and CLIP strengths — all from inside the node.
 
| Parameter | Type | Description |
|---|---|---|
| `loras_json` | STRING (hidden) | Serialised LoRA list from the JS widget |
 
**Outputs:** `lora_stack (FE_LORA_STACK)`
 
**Features:**
- Folder-tree browser with search, per-row on/off toggle, and strength sliders
- Optional separate CLIP strength per LoRA (right-click the node)
- Module-level weight cache — identical files shared across multiple nodes or tile streams are read from disk only once
- CivitAI lookup via SHA-256: automatically fetches model name, base model, trained trigger words, and preview images; results are cached to a `.fe-info.json` sidecar file
 
---
 
### ⚡ Apply LoRA
 
> Applies a `FE_LORA_STACK` from LoRA Load to a MODEL (and optionally CLIP). Architecture-agnostic: SD1, SDXL, Flux, WAN 2.1/2.2, HunyuanVideo, and others.
 
| Parameter | Type | Description |
|---|---|---|
| `model` | MODEL | Model to patch |
| `lora_stack` | FE_LORA_STACK | Stack from LoRA Load |
| `application_mode` | ENUM | `Stack` or `Merge` (see below) |
| `clip` | CLIP | *(optional)* CLIP to patch alongside the model |
 
**Outputs:** `model` · `clip`
 
**Application modes:**
 
| Mode | Description |
|---|---|
| `Stack` | Each LoRA applied as a sequential patch via the model patcher. Safe with any combination of LoRAs. Default |
| `Merge` | All LoRA weight deltas are pre-scaled and summed into a single combined dict, then one patch is applied. Best when LoRAs share many of the same target layers |
 
---
 
### 🔍 LoRA Trigger Analysis
 
> Analyses LoRA weight deltas against all text encoders present in the wired CLIP to surface candidate trigger words. Architecture-agnostic — encoders are discovered dynamically, covering CLIP-L/G, T5-XXL, LLaMA/Gemma, and any dual/triple encoder combination.
 
| Parameter | Type | Description |
|---|---|---|
| `lora_stack` | FE_LORA_STACK | Stack from LoRA Load |
| `clip` | CLIP | CLIP object to analyse against |
| `top_k` | INT | Number of candidate tokens to return per encoder |
 
**Outputs:** `candidate_triggers (STRING)`
 
**How it works:** For each text-encoder layer in the LoRA whose `in_features` matches a discovered encoder's embedding dimension, the full token embedding table is projected through the `lora_down` input subspace and L2 activation norms are accumulated. High-scoring tokens are those most aligned with what the LoRA was trained to respond to. Results are labelled per-encoder when multiple are present.
 
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
Core functionality requires no additional dependencies beyond what ComfyUI already provides (PyTorch, NumPy, Pillow).
 
**3. Optional dependencies** (install to unlock additional ChromaPin methods):
 
```bash
pip install kornia          # enables: reinhard_lab_gpu
pip install color-matcher   # enables: hm, mvgd, hm-mkl-hm, hm-mvgd-hm
```
 
---
 
## 🎬 Typical Workflows
 
**Tiled video diffusion:**
```
Load Video → TileSplit → [K-Sampler per tile] → TileMerge → Save Video
```
 
**Tiled diffusion with colour correction:**
```
Load Video → TileSplit → [K-Sampler per tile] → TileMerge
          → ChromaPin (+ reference frame) → Save Video
```
 
**Video upscaling:**
```
Load Video → Video Upscale With Model → [Free Video Memory] → Save Video
```
 
**LoRA workflow:**
```
Load Checkpoint → LoRA Load → Apply LoRA → [K-Sampler] → Save
                              LoRA Trigger Analysis → (use triggers in prompt)
```
 
**Combined:**
```
Load Video → TileSplit → [K-Sampler per tile] → TileMerge
          → ChromaPin → Video Upscale With Model → Free Video Memory → Save Video
```
 
---
 
## 🗺️ Roadmap
 
| Node | Description | Status |
|---|---|---|
| **TileSplit** | Grid tile splitting for video batches with alignment dropdown | ✅ Released (v0.0.4) |
| **TileMerge** | Linear weighted tile reconstruction with independent feather control | ✅ Released (v0.0.4) |
| **Text List → Batch** | LIST to batched STRING conversion | ✅ Released |
| **Text Batch → List** | Batched STRING to LIST conversion | ✅ Released |
| **ChromaPin** | Anchor-based colour correction across video sequences | ✅ Released (v0.0.1) |
| **LoRA Load** | Multi-LoRA browser UI with CivitAI lookup and weight cache | ✅ Released (v0.0.6) |
| **Apply LoRA** | Stack or merge LoRA application, architecture-agnostic | ✅ Released (v0.0.6) |
| **LoRA Trigger Analysis** | Encoder-agnostic trigger word surface analysis | ✅ Released (v0.0.6) |
 
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
