/**
 * FEnodes — Power LoRA Load custom widget
 * Author: FugitiveExpert01
 *
 * Replaces the raw `loras_json` textarea with a proper UI:
 *   • Folder-tree browser dropdown with search
 *   • Per-row: on/off toggle, LoRA selector, strength input, delete
 *   • "+ Add LoRA" button
 *   • Separate model / clip strength toggle (right-click node → Properties)
 *
 * State is serialised as JSON back into the hidden `loras_json` widget so
 * ComfyUI's normal workflow save/load handles everything automatically.
 */
 
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
 
// ─── Folder tree builder ──────────────────────────────────────────────────
 
/**
 * Turn a flat list like ["char/alice.safetensors", "style/oil.safetensors"]
 * into a nested tree: { files: [], children: { char: {...}, style: {...} } }
 */
function buildTree(loras) {
    const root = { files: [], children: {} };
    for (const p of loras) {
        const parts = p.replace(/\\/g, "/").split("/");
        let node = root;
        for (let i = 0; i < parts.length - 1; i++) {
            const seg = parts[i];
            if (!node.children[seg]) {
                node.children[seg] = { files: [], children: {} };
            }
            node = node.children[seg];
        }
        node.files.push({ label: parts[parts.length - 1], full: p });
    }
    return root;
}
 
// ─── Shared lora list (fetched once, shared across all nodes) ─────────────
 
let _loraListPromise = null;
 
async function getLoraList() {
    if (!_loraListPromise) {
        _loraListPromise = api.fetchApi("/fenodes/loras")
            .then(r => r.json())
            .then(d => d.loras ?? [])
            .catch(e => {
                console.warn("[FEnodes/PowerLoRA] Failed to fetch lora list:", e);
                _loraListPromise = null; // allow retry
                return [];
            });
    }
    return _loraListPromise;
}
 
// ─── CSS injected once ────────────────────────────────────────────────────
 
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
        }
        .fe-plora-row:hover { background: #2c2c2c; }
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
            transition: border-color 0.15s;
        }
        .fe-plora-add:hover { border-color: #5aaa5a; color: #8aca8a; background: #22341e; }
 
        /* ── Browser panel ── */
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
        .fe-lora-browser-search {
            padding: 6px;
            border-bottom: 1px solid #2e2e2e;
            flex-shrink: 0;
        }
        .fe-lora-browser-search input {
            width: 100%;
            box-sizing: border-box;
            background: #252525;
            border: 1px solid #4a4a4a;
            color: #ccc;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }
        .fe-lora-browser-search input:focus { border-color: #6ab; outline: none; }
        .fe-lora-browser-list {
            overflow-y: auto;
            flex: 1;
            padding: 4px 0;
        }
        .fe-lora-folder-header {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            cursor: pointer;
            color: #999;
            font-size: 11px;
            font-weight: 600;
            user-select: none;
        }
        .fe-lora-folder-header:hover { color: #bbb; background: #222; }
        .fe-lora-folder-arrow { font-size: 9px; transition: transform 0.1s; }
        .fe-lora-folder-children { padding-left: 12px; }
        .fe-lora-file {
            padding: 3px 10px;
            cursor: pointer;
            color: #c0c0c0;
            font-size: 11px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .fe-lora-file:hover { background: #2a2a2a; color: #fff; }
        .fe-lora-file.selected { color: #7bf; font-weight: 600; }
    `;
    document.head.appendChild(s);
}
 
// ─── Active browser panel (only one at a time) ────────────────────────────
 
let _activeBrowser = null;
 
function closeBrowser() {
    _activeBrowser?.remove();
    _activeBrowser = null;
}
 
/**
 * Open the folder-tree browser near `anchorEl`.
 * @param {HTMLElement} anchorEl  Element to anchor below/above.
 * @param {string}      currentFull  Currently selected lora full path (for highlight).
 * @param {string[]}    allLoras  Flat list of all lora paths.
 * @param {function}    onSelect  Called with chosen full path string.
 */
function openBrowser(anchorEl, currentFull, allLoras, onSelect) {
    closeBrowser();
 
    const tree = buildTree(allLoras);
 
    const panel = document.createElement("div");
    panel.className = "fe-lora-browser";
 
    // Search bar
    const searchWrap = document.createElement("div");
    searchWrap.className = "fe-lora-browser-search";
    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.placeholder = "🔍  search…";
    searchWrap.appendChild(searchInput);
    panel.appendChild(searchWrap);
 
    // Scrollable list
    const listEl = document.createElement("div");
    listEl.className = "fe-lora-browser-list";
    panel.appendChild(listEl);
 
    // ── Tree renderer ───────────────────────────────────────────────────
 
    /**
     * Render a tree node into a DocumentFragment.
     * @param {object} node  Tree node { files, children }.
     * @param {string} filter  Lowercase search string, or null for no filter.
     * @param {boolean} startExpanded  Whether folders start open (true when filtering).
     */
    function renderNode(node, filter, startExpanded) {
        const frag = document.createDocumentFragment();
 
        // Files in this node
        for (const f of node.files) {
            if (filter && !f.full.toLowerCase().includes(filter)) continue;
            const item = document.createElement("div");
            item.className = "fe-lora-file" + (f.full === currentFull ? " selected" : "");
            // Show just the filename without extension
            item.textContent = f.label.replace(/\.(safetensors|pt|ckpt)$/i, "");
            item.title = f.full;
            item.addEventListener("click", () => {
                onSelect(f.full);
                closeBrowser();
            });
            frag.appendChild(item);
        }
 
        // Subfolders
        for (const [folderName, child] of Object.entries(node.children)) {
            // When filtering: only show folder if it has matching descendants
            if (filter) {
                const childFrag = renderNode(child, filter, true);
                if (!childFrag.childNodes.length) continue;
                const header = document.createElement("div");
                header.className = "fe-lora-folder-header";
                header.innerHTML = `<span>📁 ${folderName}</span>`;
                frag.appendChild(header);
                const childDiv = document.createElement("div");
                childDiv.className = "fe-lora-folder-children";
                childDiv.appendChild(childFrag);
                frag.appendChild(childDiv);
            } else {
                // Collapsible folder
                let expanded = startExpanded;
                const header = document.createElement("div");
                header.className = "fe-lora-folder-header";
                const arrow = document.createElement("span");
                arrow.className = "fe-lora-folder-arrow";
                arrow.textContent = expanded ? "▼" : "▶";
                header.appendChild(arrow);
                const label = document.createElement("span");
                label.textContent = "📁 " + folderName;
                header.appendChild(label);
 
                const childDiv = document.createElement("div");
                childDiv.className = "fe-lora-folder-children";
                childDiv.style.display = expanded ? "block" : "none";
                childDiv.appendChild(renderNode(child, null, false));
 
                header.addEventListener("click", () => {
                    expanded = !expanded;
                    arrow.textContent = expanded ? "▼" : "▶";
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
 
    // Position: below anchor, or above if near bottom of viewport
    const rect = anchorEl.getBoundingClientRect();
    const panelH = 360;
    const spaceBelow = window.innerHeight - rect.bottom;
    const top = spaceBelow >= panelH ? rect.bottom + 2 : rect.top - panelH - 2;
    panel.style.left = Math.max(0, Math.min(rect.left, window.innerWidth - 304)) + "px";
    panel.style.top  = Math.max(0, top) + "px";
 
    // Close on outside click (deferred so the triggering click doesn't immediately close)
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
 
// ─── Extension registration ───────────────────────────────────────────────
 
app.registerExtension({
    name: "FEnodes.PowerLoraLoad",
 
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FELoraLoad") return;
 
        const origCreated = nodeType.prototype.onNodeCreated;
 
        nodeType.prototype.onNodeCreated = async function () {
            origCreated?.apply(this, arguments);
 
            const node = this;
 
            // ── Find and hide the raw loras_json widget ─────────────────
            const jsonWidget = node.widgets?.find(w => w.name === "loras_json");
            if (jsonWidget) {
                // Remove it from visual sizing but keep it for serialisation
                jsonWidget.computeSize = () => [0, -4]; // -4 collapses the gap
                const origDraw = jsonWidget.draw;
                jsonWidget.draw = () => {};
            }
 
            // ── Parse existing saved state ───────────────────────────────
            let loraRows = []; // [{ enabled, lora, strength_model, strength_clip }]
            if (jsonWidget?.value) {
                try { loraRows = JSON.parse(jsonWidget.value); } catch {}
            }
 
            // Whether to show separate model / clip strengths
            // (toggleable via node Properties panel)
            let splitStrength = false;
 
            // ── Fetch lora list async ────────────────────────────────────
            let allLoras = [];
            getLoraList().then(list => {
                allLoras = list;
            });
 
            // ── Sync state → hidden widget ───────────────────────────────
            function sync() {
                if (jsonWidget) jsonWidget.value = JSON.stringify(loraRows);
            }
 
            // ── Build a single row element ───────────────────────────────
            function buildRow(entry, idx) {
                const row = document.createElement("div");
                row.className = "fe-plora-row";
 
                // Toggle
                const toggle = document.createElement("input");
                toggle.type = "checkbox";
                toggle.className = "fe-plora-toggle";
                toggle.checked = entry.enabled !== false;
                toggle.title = "Enable / disable";
                toggle.addEventListener("change", () => {
                    loraRows[idx].enabled = toggle.checked;
                    sync();
                    // Dim the selector visually when disabled
                    selector.style.opacity = toggle.checked ? "1" : "0.4";
                });
 
                // Selector button
                const selector = document.createElement("button");
                selector.className = "fe-plora-selector" + (entry.lora ? "" : " empty");
                selector.style.opacity = entry.enabled !== false ? "1" : "0.4";
                selector.textContent = entry.lora
                    ? entry.lora.split("/").pop().replace(/\.(safetensors|pt|ckpt)$/i, "")
                    : "— select lora —";
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
 
                // Strength input(s)
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
                        const newVal = Math.round((parseFloat(inp.value) + delta) * 100) / 100;
                        inp.value = Math.max(-10, Math.min(10, newVal)).toFixed(2);
                        inp.dispatchEvent(new Event("change"));
                    });
                    return inp;
                }
 
                const strModel = makeStrInput(
                    entry.strength_model ?? 1.0,
                    "Model strength",
                    (v) => {
                        loraRows[idx].strength_model = v;
                        if (!splitStrength) loraRows[idx].strength_clip = v;
                        sync();
                    }
                );
 
                let strClip = null;
                if (splitStrength) {
                    strClip = makeStrInput(
                        entry.strength_clip ?? entry.strength_model ?? 1.0,
                        "CLIP strength",
                        (v) => { loraRows[idx].strength_clip = v; sync(); }
                    );
                }
 
                // Delete
                const del = document.createElement("button");
                del.className = "fe-plora-del";
                del.textContent = "✕";
                del.title = "Remove this LoRA";
                del.addEventListener("click", () => {
                    loraRows.splice(idx, 1);
                    sync();
                    renderAll();
                });
 
                row.appendChild(toggle);
                row.appendChild(selector);
                row.appendChild(strModel);
                if (strClip) row.appendChild(strClip);
                row.appendChild(del);
                return row;
            }
 
            // ── Container ────────────────────────────────────────────────
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
                loraRows.forEach((entry, idx) => {
                    rowsEl.appendChild(buildRow(entry, idx));
                });
                // Nudge ComfyUI to reflow node size
                try {
                    node.setSize([node.size[0], node.computeSize()[1]]);
                    app.graph.setDirtyCanvas(true, false);
                } catch {}
            }
 
            // ── Add DOM widget ───────────────────────────────────────────
            node.addDOMWidget("power_loras_ui", "FEPowerLorasUI", container, {
                getValue: () => JSON.stringify(loraRows),
                setValue: (v) => {
                    try { loraRows = JSON.parse(v); } catch {}
                    renderAll();
                },
                getMinHeight: () => {
                    // Approx: 30px per row + 32px for button + 12px padding
                    return loraRows.length * 30 + 44;
                },
            });
 
            renderAll();
 
            // ── Expose splitStrength toggle via context menu ──────────────
            const origGetExtraMenuOptions = node.getExtraMenuOptions?.bind(node);
            node.getExtraMenuOptions = function(_, options) {
                origGetExtraMenuOptions?.(_, options);
                options.push({
                    content: splitStrength
                        ? "✔ Separate CLIP strength (on)"
                        : "  Separate CLIP strength (off)",
                    callback: () => {
                        splitStrength = !splitStrength;
                        renderAll();
                    },
                });
            };
        };
    },
});
