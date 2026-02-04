import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PREVIEW_WIDGET = Symbol("image_loopback_preview_widget");
const LIVE_PREVIEW_DEBOUNCE = Symbol("image_loopback_live_preview_debounce");

function normalizeText(text) {
	if (text === null || text === undefined) {
		return [];
	}
	if (Array.isArray(text)) {
		return text.map((value) => String(value));
	}
	return [String(text)];
}

function coerceTextLines(text) {
	const lines = normalizeText(text);
	if (!lines.length) return [];
	return lines.flatMap((line) => String(line).split("\n"));
}

function buildImageUrl(image) {
	if (!image) return "";
	const params = new URLSearchParams();
	params.set("filename", image.filename);
	if (image.subfolder) params.set("subfolder", image.subfolder);
	if (image.type) params.set("type", image.type);
	const format = app.getPreviewFormatParam?.() ?? "";
	const suffix = format ? `&${format}` : "";
	return `./view?${params.toString()}&t=${Date.now()}${suffix}`;
}

function ensurePreviewWidget(node) {
	if (node[PREVIEW_WIDGET]) return node[PREVIEW_WIDGET];
	if (typeof node.addDOMWidget !== "function") return null;

	const container = document.createElement("div");
	container.className = "image-loopback-preview";
	container.style.display = "flex";
	container.style.flexDirection = "column";
	container.style.gap = "6px";
	container.style.width = "100%";

	const status = document.createElement("div");
	status.className = "image-loopback-status";
	status.style.fontSize = "11px";
	status.style.opacity = "0.7";
	status.style.whiteSpace = "pre-wrap";
	status.textContent = "No preview yet";

	const grid = document.createElement("div");
	grid.className = "image-loopback-grid";
	grid.style.display = "grid";
	grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(90px, 1fr))";
	grid.style.gap = "6px";

	container.appendChild(status);
	container.appendChild(grid);

	const widget = node.addDOMWidget("image_loopback_preview", "div", container, {
		serialize: false,
		hideOnZoom: false,
		getValue() { return container.value; },
		setValue(val) { container.value = val; },
	});

	widget.computeSize = function (width) {
		const count = Math.max(1, widget.imageCount || 0);
		const availWidth = width - 20; // padding
		const minCell = 90;
		const gap = 6;
		// Calculate how many columns fit
		const columns = Math.max(1, Math.floor((availWidth + gap) / (minCell + gap)));
		const rows = Math.max(1, Math.ceil(count / columns));
		// Cell size fills available width
		const cellSize = (availWidth - (columns - 1) * gap) / columns;
		const statusHeight = 36;
		return [width, rows * (cellSize + gap) + statusHeight];
	};

	widget.statusEl = status;
	widget.gridEl = grid;
	widget.imageCount = 0;
	node[PREVIEW_WIDGET] = widget;

	return widget;
}

function populatePreview(node, images, textLines) {
	const widget = ensurePreviewWidget(node);
	if (!widget) return;

	const grid = widget.gridEl;
	grid.innerHTML = "";

	const lines = textLines?.length ? textLines : ["No preview yet"];
	widget.statusEl.textContent = lines.join("\n");

	const list = Array.isArray(images) ? images : (images ? [images] : []);
	widget.imageCount = list.length || 1;

	if (!list.length) {
		const placeholder = document.createElement("div");
		placeholder.style.aspectRatio = "1";
		placeholder.style.border = "1px dashed var(--border-color)";
		placeholder.style.display = "flex";
		placeholder.style.alignItems = "center";
		placeholder.style.justifyContent = "center";
		placeholder.style.fontSize = "11px";
		placeholder.style.opacity = "0.6";
		placeholder.textContent = "empty";
		grid.appendChild(placeholder);
	} else {
		for (const image of list) {
			const container = document.createElement("div");
			container.style.aspectRatio = "1";
			container.style.position = "relative";

			const img = document.createElement("img");
			img.src = buildImageUrl(image);
			img.style.width = "100%";
			img.style.height = "100%";
			img.style.objectFit = "contain";
			img.style.border = "1px solid var(--border-color)";
			container.appendChild(img);
			grid.appendChild(container);
		}
	}

	requestAnimationFrame(() => {
		const size = node.computeSize();
		node.onResize?.(size);
		app.graph.setDirtyCanvas(true, false);
	});
}

/**
 * Populate preview with a mixed list of frames (some may be missing).
 * Each frame is either {filename, subfolder, type, index} or {missing: true, index}.
 */
function populatePreviewWithFrames(node, frameList, textLines) {
	const widget = ensurePreviewWidget(node);
	if (!widget) return;

	const grid = widget.gridEl;
	grid.innerHTML = "";

	const lines = textLines?.length ? textLines : ["No preview yet"];
	widget.statusEl.textContent = lines.join("\n");

	widget.imageCount = frameList.length || 1;

	if (!frameList.length) {
		const placeholder = document.createElement("div");
		placeholder.style.aspectRatio = "1";
		placeholder.style.border = "1px dashed var(--border-color)";
		placeholder.style.display = "flex";
		placeholder.style.alignItems = "center";
		placeholder.style.justifyContent = "center";
		placeholder.style.fontSize = "11px";
		placeholder.style.opacity = "0.6";
		placeholder.textContent = "empty";
		grid.appendChild(placeholder);
	} else {
		for (const frame of frameList) {
			if (frame.startingImage) {
				// Show starting_image preview (faint to indicate it's not yet cached)
				const container = document.createElement("div");
				container.style.aspectRatio = "1";
				container.style.position = "relative";
				container.style.border = "2px dashed var(--input-text)";
				container.style.background = "rgba(100,150,100,0.1)";

				if (frame.startingImageUrl) {
					const img = document.createElement("img");
					img.src = frame.startingImageUrl;
					img.style.width = "100%";
					img.style.height = "100%";
					img.style.objectFit = "contain";
					img.style.opacity = "0.5";
					container.appendChild(img);
				}

				const label = document.createElement("div");
				label.style.position = "absolute";
				label.style.bottom = "4px";
				label.style.left = "4px";
				label.style.fontSize = "9px";
				label.style.background = "rgba(0,0,0,0.7)";
				label.style.padding = "2px 4px";
				label.style.borderRadius = "2px";
				label.textContent = `starting → t-${frame.index}`;
				container.appendChild(label);

				grid.appendChild(container);
			} else if (frame.missing) {
				// Show placeholder for missing frame (dashed border indicates missing)
				const placeholder = document.createElement("div");
				placeholder.style.aspectRatio = "1";
				placeholder.style.border = "1px dashed var(--border-color)";
				placeholder.style.display = "flex";
				placeholder.style.alignItems = "center";
				placeholder.style.justifyContent = "center";
				placeholder.style.fontSize = "11px";
				placeholder.style.opacity = "0.5";
				placeholder.textContent = `t-${frame.index}`;
				grid.appendChild(placeholder);
			} else {
				// Show actual image with index label
				const container = document.createElement("div");
				container.style.position = "relative";
				container.style.aspectRatio = "1";

				const img = document.createElement("img");
				img.src = buildImageUrl(frame);
				img.style.width = "100%";
				img.style.height = "100%";
				img.style.objectFit = "contain";
				img.style.border = "1px solid var(--border-color)";
				container.appendChild(img);

				const label = document.createElement("div");
				label.style.position = "absolute";
				label.style.bottom = "2px";
				label.style.left = "2px";
				label.style.fontSize = "9px";
				label.style.background = "rgba(0,0,0,0.6)";
				label.style.padding = "1px 3px";
				label.style.borderRadius = "2px";
				label.textContent = `t-${frame.index}`;
				container.appendChild(label);

				grid.appendChild(container);
			}
		}
	}

	requestAnimationFrame(() => {
		const size = node.computeSize();
		node.onResize?.(size);
		app.graph.setDirtyCanvas(true, false);
	});
}

/**
 * Fetch live preview from server based on current widget values.
 * Calls the /image_loopback/preview_cache endpoint.
 */
async function fetchLivePreview(node) {
	// Get widget values
	const cachePathWidget = node.widgets?.find(w => w.name === "cache_path");
	const historyIndicesWidget = node.widgets?.find(w => w.name === "history_indices");

	const cachePath = cachePathWidget?.value ?? "loopback_cache";
	const historyIndices = historyIndicesWidget?.value ?? "";

	// Check if starting_image input is connected and try to get its preview
	const startingImageInput = node.inputs?.find(i => i.name === "starting_image");
	const hasStartingImage = startingImageInput?.link != null;
	let startingImageUrl = null;

	if (hasStartingImage && startingImageInput.link != null) {
		// Trace the connection to get the source node's image
		const linkInfo = app.graph.links[startingImageInput.link];
		if (linkInfo) {
			const sourceNode = app.graph.getNodeById(linkInfo.origin_id);
			if (sourceNode) {
				// Try to get image from source node's imgs array (set after execution)
				if (sourceNode.imgs?.length > 0) {
					startingImageUrl = sourceNode.imgs[0].src;
				}
				// Or check for widgets with image info (like Load Image node)
				else if (sourceNode.widgets) {
					const imgWidget = sourceNode.widgets.find(w => w.name === "image");
					if (imgWidget?.value) {
						// Load Image node stores filename in widget
						startingImageUrl = `./view?filename=${encodeURIComponent(imgWidget.value)}&type=input`;
					}
				}
			}
		}
	}

	// Get the current workflow
	const workflow = app.graph.serialize();

	try {
		const response = await api.fetchApi("/image_loopback/preview_cache", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				workflow: workflow,
				cache_path: cachePath,
				history_indices: historyIndices,
			}),
		});

		if (!response.ok) {
			console.warn("[image_loopback] Preview fetch failed:", response.status);
			return;
		}

		const data = await response.json();
		const { frames, total_history, workflow_key } = data;

		// Build status line (compact)
		const existingCount = frames.filter(f => f.exists).length;
		const missingCount = frames.filter(f => !f.exists).length;
		let statusParts = [`${workflow_key} | ${total_history} in cache`];
		if (hasStartingImage) {
			statusParts.push("starting_image connected");
		}
		if (missingCount > 0 && existingCount > 0) {
			statusParts.push(`${missingCount} missing`);
		} else if (missingCount > 0 && !hasStartingImage) {
			statusParts.push("all missing");
		}
		const statusLines = [statusParts.join(" | ")];

		// Build frame list for display
		// If starting_image is connected and would be used (t-1 missing, cache empty), show it
		const frameList = frames.map(f => {
			if (f.exists) {
				return {
					filename: f.filename,
					subfolder: f.subfolder,
					type: f.type,
					index: f.index,
				};
			} else if (hasStartingImage && f.index === 1 && total_history === 0) {
				// starting_image will be injected as t-1 when cache is empty
				return { startingImage: true, startingImageUrl, index: f.index };
			} else {
				return { missing: true, index: f.index };
			}
		});

		populatePreviewWithFrames(node, frameList, statusLines);
	} catch (err) {
		console.warn("[image_loopback] Live preview error:", err);
	}
}

/**
 * Schedule a debounced live preview update.
 */
function scheduleLivePreview(node, delay = 300) {
	if (node[LIVE_PREVIEW_DEBOUNCE]) {
		clearTimeout(node[LIVE_PREVIEW_DEBOUNCE]);
	}
	node[LIVE_PREVIEW_DEBOUNCE] = setTimeout(() => {
		fetchLivePreview(node);
	}, delay);
}

function isLoopbackLoadNode(nodeData) {
	const name = nodeData?.name ?? "";
	const display = nodeData?.display_name ?? nodeData?.displayName ?? "";
	return (
		name === "Image-Loopback-Load" ||
		name === "Load Loopback" ||
		name === "Load Image For Loopback" ||
		display === "Load Loopback" ||
		display === "Load Image For Loopback"
	);
}

app.registerExtension({
	name: "image_loopback.ui",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!isLoopbackLoadNode(nodeData)) {
			return;
		}

		const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			originalOnNodeCreated?.apply(this, arguments);
			populatePreview(this, null, ["Loading preview..."]);

			// Hook widget callbacks to trigger live preview on change
			const node = this;
			const widgetsToWatch = ["cache_path", "history_indices"];

			for (const widget of this.widgets || []) {
				if (widgetsToWatch.includes(widget.name)) {
					// Hook the callback for value changes (fires on blur/enter)
					const originalCallback = widget.callback;
					widget.callback = function (...args) {
						if (originalCallback) {
							originalCallback.apply(this, args);
						}
						scheduleLivePreview(node, 300);
					};

					// Also hook the input element directly for real-time typing feedback
					// ComfyUI string widgets often have an inputEl property
					if (widget.inputEl) {
						widget.inputEl.addEventListener("input", () => {
							scheduleLivePreview(node, 300);
						});
					}
				}
			}

			// Initial live preview fetch
			scheduleLivePreview(this, 100);
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			// After execution, refresh live preview to show updated cache
			scheduleLivePreview(this, 100);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			onConfigure?.apply(this, arguments);
			// Fetch live preview from cache
			scheduleLivePreview(this, 100);
		};
	},
});
