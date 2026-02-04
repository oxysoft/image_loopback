import { app } from "/scripts/app.js";

const STATUS_WIDGETS = Symbol("image_loopback_status_widgets");
const STATUS_VALUES = Symbol("image_loopback_status_values");
const PREVIEW_WIDGET = Symbol("image_loopback_preview_widget");
const PREVIEW_VALUES = Symbol("image_loopback_preview_values");

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
	grid.style.alignItems = "center";

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
		const columns = 3;
		const rows = Math.max(1, Math.ceil(count / columns));
		const cell = 90;
		const gap = 6;
		const statusHeight = 18;
		return [width, rows * (cell + gap) + statusHeight + 12];
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
		placeholder.style.height = "90px";
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
			const img = document.createElement("img");
			img.src = buildImageUrl(image);
			img.style.width = "100%";
			img.style.height = "90px";
			img.style.objectFit = "contain";
			img.style.border = "1px solid var(--border-color)";
			grid.appendChild(img);
		}
	}

	requestAnimationFrame(() => {
		const size = node.computeSize();
		node.onResize?.(size);
		app.graph.setDirtyCanvas(true, false);
	});
}

app.registerExtension({
	name: "image_loopback.ui",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "Image-Loopback-Load") {
			return;
		}

		const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			originalOnNodeCreated?.apply(this, arguments);
			populatePreview(this, null, ["No preview yet"]);
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			if (!message) return;
			this[STATUS_VALUES] = message.text;
			this[PREVIEW_VALUES] = message.images;
			const lines = coerceTextLines(message.text);
			populatePreview(this, message.images, lines);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			onConfigure?.apply(this, arguments);
			if (this[PREVIEW_VALUES] || this[STATUS_VALUES]) {
				populatePreview(
					this,
					this[PREVIEW_VALUES],
					coerceTextLines(this[STATUS_VALUES])
				);
			}
		};
	},
});
