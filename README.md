FugitiveExpert01 — ComfyUI VFX Nodes
A growing collection of custom ComfyUI nodes built for VFX production pipelines — designed around the real demands of working with high-resolution imagery, video sequences, and AI-assisted visual effects workflows.

Overview
This repo aims to fill the gaps between off-the-shelf ComfyUI nodes and the specific needs of VFX work: large format images, temporal consistency across video frames, precise spatial control, and clean integration with diffusion-based upscaling and enhancement models.
Nodes are built with video batch support as a first-class concern — not an afterthought.

Current Nodes
TileSplit
Splits an image or video batch into an overlapping grid of tiles, ready to be passed individually to a model (e.g. a video K-sampler or upscaler).

Configurable grid via tiles_x / tiles_y
Overlap calculated automatically and stored for seamless reconstruction
Outputs tiles as a list of (F, H, W, C) tensors — compatible with video model K-samplers
Debug output image showing tile boundaries and overlap regions
Tile dimensions snapped to multiples of 8 for model compatibility

Outputs: tiles (LIST), debug_image, tile_calc

TileMerge
Reconstructs a full image or video sequence from processed tiles using Laplacian Pyramid blending for seamless, artifact-free joins at tile boundaries.

Gaussian feathering masks at overlap regions
Laplacian frequency blending for natural-looking seams
Robust handling of tensor shapes returned by video models
Weighted accumulation handles overlapping regions correctly

Inputs: tiles (LIST), tile_calc

Installation

Navigate to your ComfyUI custom nodes directory:

ComfyUI/custom_nodes/

Clone this repository:

bashgit clone https://github.com/FugitiveExpert01/comfyui-vfx-nodes.git

Restart ComfyUI. Nodes will appear in the node menu under the VFX category.

No additional dependencies beyond what ComfyUI already requires (PyTorch, NumPy, Pillow).

Typical Workflow
Load Video → TileSplit → [Video Model / K-Sampler per tile] → TileMerge → Save Video
Because TileSplit outputs a list of tile sequences, you can route each tile through a video model independently and then reconstruct the full frame with TileMerge. This allows processing of resolutions that would otherwise exceed VRAM limits.

Roadmap
This repo is actively developed alongside VFX production work. Planned and in-progress nodes include:

Optical Flow Warp — frame-to-frame warping for temporal consistency
Depth-Guided Composite — layer compositing driven by estimated depth maps
Multi-pass Denoise — frequency-separated denoising passes for fine/coarse detail
Temporal Blend — blending across frame windows to reduce flicker
Matte Operations — erode, dilate, blur, and combine alpha mattes
Color Space Utilities — linear/log/sRGB conversions for VFX-accurate colour handling

Have a node idea or a production use case that isn't covered? Open an issue.

Contributing
Pull requests are welcome. If you're adding a node, please:

Keep video batch support (F, H, W, C) as the primary tensor convention
Add a docstring describing what the node does and its input/output types
Test with both single images and multi-frame video batches


License
MIT License — free to use in personal and commercial VFX pipelines.
