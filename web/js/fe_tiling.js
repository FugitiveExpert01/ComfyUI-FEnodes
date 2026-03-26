import { app } from "../../../scripts/app.js";
 
function hideWidget(node, widget) {
    if (widget.__fe_hidden) return;
    widget.__fe_hidden      = true;
    widget.__fe_origType    = widget.type;
    widget.__fe_origCompute = widget.computeSize;
    widget.type             = "tschide";
    widget.computeSize      = () => [0, -4];
    node.setSize(node.computeSize());
}
 
function showWidget(node, widget) {
    if (!widget.__fe_hidden) return;
    widget.__fe_hidden = false;
    widget.type        = widget.__fe_origType;
    widget.computeSize = widget.__fe_origCompute || undefined;
    node.setSize(node.computeSize());
}
 
app.registerExtension({
    name: "FEnodes.TileSplit",
 
    nodeCreated(node) {
        if (node.comfyClass !== "TileSplit") return;
 
        const usePixelWidget = node.widgets?.find(w => w.name === "use_pixel_size");
        const tilesXWidget   = node.widgets?.find(w => w.name === "tiles_x");
        const tilesYWidget   = node.widgets?.find(w => w.name === "tiles_y");
        const tileWWidget    = node.widgets?.find(w => w.name === "tile_width");
        const tileHWidget    = node.widgets?.find(w => w.name === "tile_height");
 
        if (!usePixelWidget || !tilesXWidget || !tilesYWidget || !tileWWidget || !tileHWidget) return;
 
        function updateVisibility() {
            const usePixel = usePixelWidget.value;
            if (usePixel) {
                hideWidget(node, tilesXWidget);
                hideWidget(node, tilesYWidget);
                showWidget(node, tileWWidget);
                showWidget(node, tileHWidget);
            } else {
                showWidget(node, tilesXWidget);
                showWidget(node, tilesYWidget);
                hideWidget(node, tileWWidget);
                hideWidget(node, tileHWidget);
            }
        }
 
        usePixelWidget.callback = updateVisibility;
        updateVisibility();
    },
});
