import { app } from "../../../scripts/app.js";
 
const GRID_DEFAULTS  = { tiles_x: 2,   tiles_y: 2   };
const PIXEL_DEFAULTS = { tiles_x: 768, tiles_y: 768 };
 
const GRID_LABELS  = { tiles_x: "tiles_x",    tiles_y: "tiles_y"     };
const PIXEL_LABELS = { tiles_x: "tile_width",  tiles_y: "tile_height" };
 
app.registerExtension({
    name: "FEnodes.TileSplit",
 
    nodeCreated(node) {
        if (node.comfyClass !== "TileSplit") return;
 
        const usePixelWidget = node.widgets?.find(w => w.name === "use_pixel_size");
        const tilesXWidget   = node.widgets?.find(w => w.name === "tiles_x");
        const tilesYWidget   = node.widgets?.find(w => w.name === "tiles_y");
 
        if (!usePixelWidget || !tilesXWidget || !tilesYWidget) return;
 
        function updateLabels(usePixel, resetValues) {
            const labels   = usePixel ? PIXEL_LABELS   : GRID_LABELS;
            const defaults = usePixel ? PIXEL_DEFAULTS : GRID_DEFAULTS;
 
            tilesXWidget.label = labels.tiles_x;
            tilesYWidget.label = labels.tiles_y;
 
            if (resetValues) {
                tilesXWidget.value = defaults.tiles_x;
                tilesYWidget.value = defaults.tiles_y;
            }
 
            node.setDirtyCanvas(true, true);
        }
 
        // On toggle: rename and reset values
        usePixelWidget.callback = (value) => updateLabels(value, true);
 
        // On load: rename only, preserve saved values
        updateLabels(usePixelWidget.value, false);
    },
});
