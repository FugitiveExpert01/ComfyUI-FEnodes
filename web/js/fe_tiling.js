import { app } from "../../../scripts/app.js";

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

        // Store original types so we can restore them
        const origTypeX = tilesXWidget.type;
        const origTypeY = tilesYWidget.type;
        const origTypeW = tileWWidget.type;
        const origTypeH = tileHWidget.type;

        function updateVisibility() {
            const usePixel = usePixelWidget.value;

            tilesXWidget.type = usePixel ? "hidden" : origTypeX;
            tilesYWidget.type = usePixel ? "hidden" : origTypeY;
            tileWWidget.type  = usePixel ? origTypeW : "hidden";
            tileHWidget.type  = usePixel ? origTypeH : "hidden";

            node.setSize(node.computeSize());
        }

        usePixelWidget.callback = updateVisibility;
        updateVisibility();  // apply on load
    },
});
