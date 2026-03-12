/**
 * FEnodes -- Power LoRA Load custom widget
 * Author: FugitiveExpert01
 *
 * Features:
 *   - Folder-tree browser with search
 *   - Per-row on/off toggle, strength input (+ optional separate CLIP strength)
 *   - Drag-and-drop row reordering
 *   - Right-click row -> LoRA info panel with CivitAI fetch
 *   - "+ Add LoRA" button
 */
 
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
 
console.log("[FEnodes] fe_power_lora.js: module top-level executing");
 
// ---------------------------------------------------------------------------
// Folder tree builder
// ---------------------------------------------------------------------------
function buildTree(loras) {
    const root = { files: [], children: {} };
    for (const p of loras) {
        const parts = p.replace(/\\/g, "/").split("/");
        let node = root;
        for (let i = 0; i < parts.length - 1; i++) {
            const seg = parts[i];
            if (!node.children[seg]) node.children[seg] = { files: [], children: {} };
            node = node.children[seg];
        }
        node.files.push({ label: parts[parts.length - 1], full: p });
    }
    return root;
}
 
// ---------------------------------------------------------------------------
// Shared lora list (fetched once)
// ---------------------------------------------------------------------------
let _loraListPromise = null;
async function getLoraList() {
    if (!_loraListPromise) {
        _loraListPromise = api.fetchApi("/fenodes/loras")
            .then(r => r.json())
            .then(d => d.loras ?? [])
            .catch(e => {
                console.warn("[FEnodes/PowerLoRA] Failed to fetch lora list:", e);
                _loraListPromise = null;
                return [];
            });
    }
    return _loraListPromise;
}
 
// ---------------------------------------------------------------------------
// CSS (injected once)
// ---------------------------------------------------------------------------
const STYLE_ID = "fenodes-power-lora-style";
if (!document.getElementById(STYLE_ID)) {
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
        .fe-plora-row {
            display: flex;
            align-items: center;
            gap: 4px;
            margin-bottom: 3px;
            background: #252525;
            border-radius: 4px;
            padding: 3px 5px;
            min-height: 26px;
            cursor: default;
        }
        .fe-plora-row:hover { background: #2c2c2c; }
        .fe-plora-row.fe-drag-over { outline: 1px dashed #5a9a5a; background: #1e2e1e; }
        .fe-plora-row.fe-dragging  { opacity: 0.4; }
 
        .fe-plora-handle {
            flex-shrink: 0;
            cursor: grab;
            color: #555;
            font-size: 13px;
            padding: 0 2px;
            user-select: none;
        }
        .fe-plora-handle:hover { color: #888; }
 
        .fe-plora-toggle {
            cursor: pointer;
            accent-color: #5a9a5a;
            flex-shrink: 0;
            width: 14px; height: 14px;
        }
        .fe-plora-selector {
            flex: 1;
            background: #2e2e2e;
            border: 1px solid #4a4a4a;
            color: #ccc;
            border-radius: 3px;
            padding: 2px 6px;
            cursor: pointer;
            text-align: left;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 11px;
            min-width: 0;
        }
        .fe-plora-selector:hover { border-color: #6a6a6a; background: #363636; }
        .fe-plora-selector.empty { color: #666; font-style: italic; }
        .fe-plora-strength {
            width: 44px;
            flex-shrink: 0;
            background: #2e2e2e;
            border: 1px solid #4a4a4a;
            color: #ccc;
            border-radius: 3px;
            padding: 2px 3px;
            font-size: 11px;
            text-align: center;
        }
        .fe-plora-strength:focus { border-color: #7ab; outline: none; }
        .fe-plora-del {
            flex-shrink: 0;
            background: transparent;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 13px;
            padding: 0 2px;
            line-height: 1;
        }
        .fe-plora-del:hover { color: #c66; }
        .fe-plora-add {
            width: 100%;
            margin-top: 4px;
            background: #1e2e1e;
            border: 1px dashed #3a6a3a;
            color: #6aaa6a;
            border-radius: 4px;
            padding: 4px;
            cursor: pointer;
            font-size: 11px;
        }
        .fe-plora-add:hover { border-color: #5aaa5a; color: #8aca8a; background: #22341e; }
 
        /* -- Lora browser panel -- */
        .fe-lora-browser {
            position: fixed;
            background: #1a1a1a;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            z-index: 100000;
            width: 300px;
            max-height: 360px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 6px 24px rgba(0,0,0,0.7);
            overflow: hidden;
        }
        .fe-lora-browser-search { padding: 6px; border-bottom: 1px solid #2e2e2e; flex-shrink: 0; }
        .fe-lora-browser-search input {
            width: 100%; box-sizing: border-box;
            background: #252525; border: 1px solid #4a4a4a; color: #ccc;
            border-radius: 4px; padding: 4px 8px; font-size: 12px;
        }
        .fe-lora-browser-search input:focus { border-color: #6ab; outline: none; }
        .fe-lora-browser-list { overflow-y: auto; flex: 1; padding: 4px 0; }
        .fe-lora-folder-header {
            display: flex; align-items: center; gap: 5px;
            padding: 4px 10px; cursor: pointer; color: #999;
            font-size: 11px; font-weight: 600; user-select: none;
        }
        .fe-lora-folder-header:hover { color: #bbb; background: #222; }
        .fe-lora-folder-arrow { font-size: 9px; }
        .fe-lora-folder-children { padding-left: 12px; }
        .fe-lora-file {
            padding: 3px 10px; cursor: pointer; color: #c0c0c0;
            font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .fe-lora-file:hover { background: #2a2a2a; color: #fff; }
        .fe-lora-file.selected { color: #7bf; font-weight: 600; }
 
        /* -- Info panel -- */
        .fe-lora-info-panel {
            position: fixed;
            background: #1a1a1a;
            border: 1px solid #4a4a4a;
            border-radius: 6px;
            z-index: 100001;
            width: 280px;
            max-height: 420px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 6px 24px rgba(0,0,0,0.8);
            overflow: hidden;
            font-size: 11px;
            color: #ccc;
        }
        .fe-lora-info-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 7px 10px; background: #222; border-bottom: 1px solid #333;
            font-weight: 600; font-size: 12px; flex-shrink: 0;
        }
        .fe-lora-info-close {
            background: none; border: none; color: #888;
            cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1;
        }
        .fe-lora-info-close:hover { color: #c66; }
        .fe-lora-info-body { overflow-y: auto; flex: 1; padding: 8px 10px; }
        .fe-lora-info-row { margin-bottom: 6px; }
        .fe-lora-info-label { color: #888; font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }
        .fe-lora-info-value { color: #ddd; word-break: break-word; }
        .fe-lora-info-words {
            display: flex; flex-wrap: wrap; gap: 3px; margin-top: 2px;
        }
        .fe-lora-info-word {
            background: #2a3a2a; border: 1px solid #3a6a3a; color: #8ccc8c;
            border-radius: 3px; padding: 1px 5px; cursor: pointer; font-size: 10px;
        }
        .fe-lora-info-word:hover { background: #334a33; }
        .fe-lora-info-word.copied { background: #3a5a3a; color: #aeeaae; }
        .fe-lora-info-link { color: #7af; text-decoration: none; font-size: 10px; }
        .fe-lora-info-link:hover { text-decoration: underline; }
        .fe-lora-info-footer {
            padding: 6px 10px; border-top: 1px solid #2a2a2a;
            display: flex; gap: 6px; flex-shrink: 0;
        }
        .fe-lora-info-btn {
            flex: 1; background: #1e2e1e; border: 1px solid #3a6a3a;
            color: #6aaa6a; border-radius: 4px; padding: 4px 6px;
            cursor: pointer; font-size: 11px;
        }
        .fe-lora-info-btn:hover { background: #22341e; border-color: #5aaa5a; }
        .fe-lora-info-btn.loading { color: #888; border-color: #444; cursor: wait; }
        .fe-lora-info-spinner { display: inline-block; animation: fe-spin 0.8s linear infinite; }
        @keyframes fe-spin { to { transform: rotate(360deg); } }
        .fe-lora-info-error { color: #c88; font-size: 11px; padding: 4px 0; }
    `;
    document.head.appendChild(s);
}
 
// ---------------------------------------------------------------------------
// Active floating panels (one browser, one info panel at a time)
// ---------------------------------------------------------------------------
let _activeBrowser = null;
let _activeInfoPanel = null;
 
function closeBrowser() { _activeBrowser?.remove(); _activeBrowser = null; }
function closeInfoPanel() { _activeInfoPanel?.remove(); _activeInfoPanel = null; }
 
// ---------------------------------------------------------------------------
// Folder-tree browser
// ---------------------------------------------------------------------------
function openBrowser(anchorEl, currentFull, allLoras, onSelect) {
    closeBrowser();
    const tree = buildTree(allLoras);
    const panel = document.createElement("div");
    panel.className = "fe-lora-browser";
 
    const searchWrap = document.createElement("div");
    searchWrap.className = "fe-lora-browser-search";
    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "search...";
    searchWrap.appendChild(searchInput);
    panel.appendChild(searchWrap);
 
    const listEl = document.createElement("div");
    listEl.className = "fe-lora-browser-list";
    panel.appendChild(listEl);
 
    function renderNode(node, filter, startExpanded) {
        const frag = document.createDocumentFragment();
        for (const f of node.files) {
            if (filter && !f.full.toLowerCase().includes(filter)) continue;
            const item = document.createElement("div");
            item.className = "fe-lora-file" + (f.full === currentFull ? " selected" : "");
            item.textContent = f.label.replace(/\.(safetensors|pt|ckpt)$/i, "");
            item.title = f.full;
            item.addEventListener("click", () => { onSelect(f.full); closeBrowser(); });
            frag.appendChild(item);
        }
        for (const [folderName, child] of Object.entries(node.children)) {
            if (filter) {
                const childFrag = renderNode(child, filter, true);
                if (!childFrag.childNodes.length) continue;
                const header = document.createElement("div");
                header.className = "fe-lora-folder-header";
                header.innerHTML = `<span>[+] ${folderName}</span>`;
                frag.appendChild(header);
                const childDiv = document.createElement("div");
                childDiv.className = "fe-lora-folder-children";
                childDiv.appendChild(childFrag);
                frag.appendChild(childDiv);
            } else {
                let expanded = startExpanded;
                const header = document.createElement("div");
                header.className = "fe-lora-folder-header";
                const arrow = document.createElement("span");
                arrow.className = "fe-lora-folder-arrow";
                arrow.textContent = expanded ? "v" : ">";
                header.appendChild(arrow);
                const label = document.createElement("span");
                label.textContent = " [folder] " + folderName;
                header.appendChild(label);
                const childDiv = document.createElement("div");
                childDiv.className = "fe-lora-folder-children";
                childDiv.style.display = expanded ? "block" : "none";
                childDiv.appendChild(renderNode(child, null, false));
                header.addEventListener("click", () => {
                    expanded = !expanded;
                    arrow.textContent = expanded ? "v" : ">";
                    childDiv.style.display = expanded ? "block" : "none";
                });
                frag.appendChild(header);
                frag.appendChild(childDiv);
            }
        }
        return frag;
    }
 
    function rebuildList(filter) {
        listEl.innerHTML = "";
        const f = filter?.toLowerCase().trim() || null;
        listEl.appendChild(renderNode(tree, f, !!f));
        if (!listEl.firstChild) {
            const empty = document.createElement("div");
            empty.style.cssText = "padding: 8px 12px; color: #666; font-size: 11px;";
            empty.textContent = f ? "No LoRAs match." : "No LoRAs found.";
            listEl.appendChild(empty);
        }
    }
 
    searchInput.addEventListener("input", () => rebuildList(searchInput.value));
    rebuildList("");
    document.body.appendChild(panel);
    _activeBrowser = panel;
 
    const rect = anchorEl.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const top = spaceBelow >= 360 ? rect.bottom + 2 : rect.top - 362;
    panel.style.left = Math.max(0, Math.min(rect.left, window.innerWidth - 304)) + "px";
    panel.style.top  = Math.max(0, top) + "px";
 
    setTimeout(() => {
        document.addEventListener("mousedown", function outsideHandler(ev) {
            if (!panel.contains(ev.target)) {
                closeBrowser();
                document.removeEventListener("mousedown", outsideHandler);
            }
        });
    }, 10);
    searchInput.focus();
}
 
// ---------------------------------------------------------------------------
// LoRA info panel
// ---------------------------------------------------------------------------
async function openInfoPanel(anchorEl, loraName, onRefresh) {
    closeInfoPanel();
    if (!loraName) return;
 
    const panel = document.createElement("div");
    panel.className = "fe-lora-info-panel";
    _activeInfoPanel = panel;
 
    // Header
    const header = document.createElement("div");
    header.className = "fe-lora-info-header";
    const title = document.createElement("span");
    title.textContent = loraName.split("/").pop().replace(/\.(safetensors|pt|ckpt)$/i, "");
    title.style.cssText = "overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:220px;";
    title.title = loraName;
    const closeBtn = document.createElement("button");
    closeBtn.className = "fe-lora-info-close";
    closeBtn.textContent = "x";
    closeBtn.addEventListener("click", closeInfoPanel);
    header.appendChild(title);
    header.appendChild(closeBtn);
    panel.appendChild(header);
 
    // Body
    const body = document.createElement("div");
    body.className = "fe-lora-info-body";
    body.textContent = "Loading...";
    panel.appendChild(body);
 
    // Footer
    const footer = document.createElement("div");
    footer.className = "fe-lora-info-footer";
    const fetchBtn = document.createElement("button");
    fetchBtn.className = "fe-lora-info-btn";
    fetchBtn.textContent = "Fetch from CivitAI";
    footer.appendChild(fetchBtn);
    panel.appendChild(footer);
 
    document.body.appendChild(panel);
 
    // Position next to anchor
    const rect = anchorEl.getBoundingClientRect();
    const spaceRight = window.innerWidth - rect.right;
    const left = spaceRight >= 290 ? rect.right + 4 : rect.left - 284;
    const top  = Math.max(0, Math.min(rect.top, window.innerHeight - 420));
    panel.style.left = Math.max(0, left) + "px";
    panel.style.top  = top + "px";
 
    // Close on outside click
    setTimeout(() => {
        document.addEventListener("mousedown", function outsideHandler(ev) {
            if (!panel.contains(ev.target)) {
                closeInfoPanel();
                document.removeEventListener("mousedown", outsideHandler);
            }
        });
    }, 10);
 
    async function loadInfo(refresh) {
        body.innerHTML = "";
        const spinner = document.createElement("span");
        spinner.className = "fe-lora-info-spinner";
        spinner.textContent = "* ";
        body.appendChild(spinner);
        body.append("Loading...");
        fetchBtn.className = "fe-lora-info-btn loading";
        fetchBtn.textContent = "Fetching...";
 
        let info;
        try {
            const url = `/fenodes/lora_info?name=${encodeURIComponent(loraName)}${refresh ? "&refresh=true" : ""}`;
            const r = await api.fetchApi(url);
            info = await r.json();
        } catch (e) {
            body.innerHTML = `<div class="fe-lora-info-error">Error: ${e.message}</div>`;
            fetchBtn.className = "fe-lora-info-btn";
            fetchBtn.textContent = "Retry";
            return;
        }
 
        body.innerHTML = "";
        fetchBtn.className = "fe-lora-info-btn";
        fetchBtn.textContent = "Refresh from CivitAI";
 
        function row(label, value) {
            if (!value) return;
            const d = document.createElement("div");
            d.className = "fe-lora-info-row";
            d.innerHTML = `<div class="fe-lora-info-label">${label}</div>
                           <div class="fe-lora-info-value">${value}</div>`;
            body.appendChild(d);
        }
 
        const civ = info.civitai;
        if (civ && !civ.error) {
            if (civ.name)        row("Name",       civ.name + (civ.versionName ? ` - ${civ.versionName}` : ""));
            if (civ.baseModel)   row("Base model", civ.baseModel);
            if (civ.url) {
                const linkRow = document.createElement("div");
                linkRow.className = "fe-lora-info-row";
                linkRow.innerHTML = `<div class="fe-lora-info-label">CivitAI</div>`;
                const a = document.createElement("a");
                a.className = "fe-lora-info-link";
                a.href = civ.url;
                a.target = "_blank";
                a.textContent = "Open model page";
                linkRow.appendChild(a);
                body.appendChild(linkRow);
            }
            if (civ.trainedWords?.length) {
                const d = document.createElement("div");
                d.className = "fe-lora-info-row";
                d.innerHTML = `<div class="fe-lora-info-label">Trained words</div>`;
                const words = document.createElement("div");
                words.className = "fe-lora-info-words";
                for (const w of civ.trainedWords) {
                    const chip = document.createElement("span");
                    chip.className = "fe-lora-info-word";
                    chip.textContent = w;
                    chip.title = "Click to copy";
                    chip.addEventListener("click", () => {
                        navigator.clipboard.writeText(w).catch(() => {});
                        chip.classList.add("copied");
                        setTimeout(() => chip.classList.remove("copied"), 1200);
                    });
                    words.appendChild(chip);
                }
                d.appendChild(words);
                body.appendChild(d);
            }
        } else if (civ?.error) {
            const d = document.createElement("div");
            d.className = "fe-lora-info-error";
            d.textContent = "CivitAI: " + civ.error;
            body.appendChild(d);
        }
 
        // Metadata fields from safetensors header
        const meta = info.metadata;
        if (meta) {
            const metaFields = {
                "ss_sd_model_name":        "Base model file",
                "ss_base_model_version":   "Base model version",
                "ss_network_module":       "Network module",
                "ss_num_train_images":     "Training images",
            };
            for (const [key, label] of Object.entries(metaFields)) {
                if (meta[key]) row(label, String(meta[key]));
            }
        }
 
        // SHA256
        if (info.sha256) {
            const d = document.createElement("div");
            d.className = "fe-lora-info-row";
            d.innerHTML = `<div class="fe-lora-info-label">SHA256</div>
                           <div class="fe-lora-info-value" style="font-size:9px; color:#666; word-break:break-all;">${info.sha256}</div>`;
            body.appendChild(d);
        }
 
        if (!body.children.length) {
            body.textContent = "No info available. Try fetching from CivitAI.";
        }
 
        onRefresh?.(info);
    }
 
    fetchBtn.addEventListener("click", () => loadInfo(true));
    loadInfo(false);
}
 
// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
    name: "FEnodes.PowerLoraLoad",
 
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name.startsWith("FE")) {
            console.log("[FEnodes] beforeRegisterNodeDef saw:", nodeData.name);
        }
        if (nodeData.name !== "FELoraLoad") return;
        console.log("[FEnodes] FELoraLoad node def found -- patching onNodeCreated");
 
        const origCreated = nodeType.prototype.onNodeCreated;
 
        nodeType.prototype.onNodeCreated = async function () {
            console.log("[FEnodes] FELoraLoad onNodeCreated fired");
            origCreated?.apply(this, arguments);
 
            const node = this;
 
            // Hide the raw loras_json widget
            const jsonWidget = node.widgets?.find(w => w.name === "loras_json");
            console.log("[FEnodes] loras_json widget found:", !!jsonWidget,
                        "| widgets:", node.widgets?.map(w => w.name));
            if (jsonWidget) {
                jsonWidget.computeSize = () => [0, -4];
                jsonWidget.draw = () => {};
            }
 
            let loraRows = [];
            if (jsonWidget?.value) {
                try { loraRows = JSON.parse(jsonWidget.value); } catch {}
            }
 
            let splitStrength = false;
 
            let allLoras = [];
            getLoraList().then(list => {
                allLoras = list;
                console.log("[FEnodes] lora list fetched, count:", allLoras.length);
            });
 
            function sync() {
                if (jsonWidget) jsonWidget.value = JSON.stringify(loraRows);
            }
 
            // ----------------------------------------------------------------
            // Drag-and-drop state
            // ----------------------------------------------------------------
            let dragSrcIdx = null;
 
            // ----------------------------------------------------------------
            // Build a single row
            // ----------------------------------------------------------------
            function buildRow(entry, idx) {
                const row = document.createElement("div");
                row.className = "fe-plora-row";
                row.draggable = true;
 
                // Drag handle
                const handle = document.createElement("span");
                handle.className = "fe-plora-handle";
                handle.textContent = "::";
                handle.title = "Drag to reorder";
 
                // Drag events
                row.addEventListener("dragstart", (e) => {
                    dragSrcIdx = idx;
                    e.dataTransfer.effectAllowed = "move";
                    // Tiny delay so the drag image renders before we dim it
                    setTimeout(() => row.classList.add("fe-dragging"), 0);
                });
                row.addEventListener("dragend", () => {
                    row.classList.remove("fe-dragging");
                    rowsEl.querySelectorAll(".fe-plora-row").forEach(r => r.classList.remove("fe-drag-over"));
                });
                row.addEventListener("dragover", (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = "move";
                    rowsEl.querySelectorAll(".fe-plora-row").forEach(r => r.classList.remove("fe-drag-over"));
                    row.classList.add("fe-drag-over");
                });
                row.addEventListener("drop", (e) => {
                    e.preventDefault();
                    if (dragSrcIdx === null || dragSrcIdx === idx) return;
                    const moved = loraRows.splice(dragSrcIdx, 1)[0];
                    loraRows.splice(idx, 0, moved);
                    dragSrcIdx = null;
                    sync();
                    renderAll();
                });
 
                // Toggle
                const toggle = document.createElement("input");
                toggle.type = "checkbox";
                toggle.className = "fe-plora-toggle";
                toggle.checked = entry.enabled !== false;
                toggle.title = "Enable / disable";
                toggle.addEventListener("change", () => {
                    loraRows[idx].enabled = toggle.checked;
                    sync();
                    selector.style.opacity = toggle.checked ? "1" : "0.4";
                });
 
                // Selector
                const selector = document.createElement("button");
                selector.className = "fe-plora-selector" + (entry.lora ? "" : " empty");
                selector.style.opacity = entry.enabled !== false ? "1" : "0.4";
                selector.textContent = entry.lora
                    ? entry.lora.split("/").pop().replace(/\.(safetensors|pt|ckpt)$/i, "")
                    : "-- select lora --";
                selector.title = entry.lora || "Click to choose a LoRA";
 
                selector.addEventListener("click", (e) => {
                    e.stopPropagation();
                    openBrowser(selector, loraRows[idx].lora, allLoras, (chosen) => {
                        loraRows[idx].lora = chosen;
                        sync();
                        selector.textContent = chosen.split("/").pop().replace(/\.(safetensors|pt|ckpt)$/i, "");
                        selector.title = chosen;
                        selector.classList.remove("empty");
                    });
                });
 
                // Right-click on row -> info panel
                row.addEventListener("contextmenu", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!loraRows[idx].lora) return;
                    openInfoPanel(row, loraRows[idx].lora);
                });
 
                // Strength inputs
                function makeStrInput(val, title, onChange) {
                    const inp = document.createElement("input");
                    inp.type = "number";
                    inp.className = "fe-plora-strength";
                    inp.value = val.toFixed(2);
                    inp.step = "0.05";
                    inp.min = "-10";
                    inp.max = "10";
                    inp.title = title;
                    inp.addEventListener("change", () => onChange(parseFloat(inp.value) || 0));
                    inp.addEventListener("wheel", (e) => {
                        e.preventDefault();
                        const delta = e.deltaY < 0 ? 0.05 : -0.05;
                        const v = Math.round((parseFloat(inp.value) + delta) * 100) / 100;
                        inp.value = Math.max(-10, Math.min(10, v)).toFixed(2);
                        inp.dispatchEvent(new Event("change"));
                    });
                    return inp;
                }
 
                const strModel = makeStrInput(
                    entry.strength_model ?? 1.0, "Model strength",
                    (v) => {
                        loraRows[idx].strength_model = v;
                        if (!splitStrength) loraRows[idx].strength_clip = v;
                        sync();
                    }
                );
 
                let strClip = null;
                if (splitStrength) {
                    strClip = makeStrInput(
                        entry.strength_clip ?? entry.strength_model ?? 1.0, "CLIP strength",
                        (v) => { loraRows[idx].strength_clip = v; sync(); }
                    );
                }
 
                // Delete
                const del = document.createElement("button");
                del.className = "fe-plora-del";
                del.textContent = "x";
                del.title = "Remove this LoRA";
                del.addEventListener("click", () => {
                    loraRows.splice(idx, 1);
                    sync();
                    renderAll();
                });
 
                row.appendChild(handle);
                row.appendChild(toggle);
                row.appendChild(selector);
                row.appendChild(strModel);
                if (strClip) row.appendChild(strClip);
                row.appendChild(del);
                return row;
            }
 
            // ----------------------------------------------------------------
            // Container
            // ----------------------------------------------------------------
            const container = document.createElement("div");
            container.style.cssText = "width:100%; padding:4px; box-sizing:border-box;";
 
            const rowsEl = document.createElement("div");
 
            const addBtn = document.createElement("button");
            addBtn.className = "fe-plora-add";
            addBtn.textContent = "+ Add LoRA";
            addBtn.addEventListener("click", () => {
                loraRows.push({ enabled: true, lora: "", strength_model: 1.0, strength_clip: 1.0 });
                sync();
                renderAll();
            });
 
            container.appendChild(rowsEl);
            container.appendChild(addBtn);
 
            function renderAll() {
                rowsEl.innerHTML = "";
                loraRows.forEach((entry, idx) => rowsEl.appendChild(buildRow(entry, idx)));
                try {
                    node.setSize([node.size[0], node.computeSize()[1]]);
                    app.graph.setDirtyCanvas(true, false);
                } catch {}
            }
 
            console.log("[FEnodes] About to call addDOMWidget");
            node.addDOMWidget("power_loras_ui", "FEPowerLorasUI", container, {
                getValue: () => JSON.stringify(loraRows),
                setValue: (v) => {
                    try { loraRows = JSON.parse(v); } catch {}
                    renderAll();
                },
                getMinHeight: () => loraRows.length * 30 + 44,
            });
 
            renderAll();
 
            // Node right-click -> separate CLIP strength toggle
            const origGetExtraMenuOptions = node.getExtraMenuOptions?.bind(node);
            node.getExtraMenuOptions = function(_, options) {
                origGetExtraMenuOptions?.(_, options);
                options.push({
                    content: splitStrength
                        ? "[x] Separate CLIP strength (on)"
                        : "[ ] Separate CLIP strength (off)",
                    callback: () => {
                        splitStrength = !splitStrength;
                        renderAll();
                    },
                });
            };
        };
    },
});
