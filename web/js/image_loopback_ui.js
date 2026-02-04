import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const STATUS_WIDGETS = Symbol("image_loopback_status_widgets");
const STATUS_VALUES = Symbol("image_loopback_status_values");

function normalizeText(text) {
	if (text === null || text === undefined) {
		return [];
	}
	if (Array.isArray(text)) {
		return text.map((value) => String(value));
	}
	return [String(text)];
}

function populateStatus(text) {
	const lines = normalizeText(text);
	if (!lines.length) {
		return;
	}

	if (!this[STATUS_WIDGETS]) {
		this[STATUS_WIDGETS] = [];
	}

	for (const widget of this[STATUS_WIDGETS]) {
		widget.onRemove?.();
	}
	this.widgets = (this.widgets || []).filter(
		(widget) => !this[STATUS_WIDGETS].includes(widget)
	);
	this[STATUS_WIDGETS] = [];

	const textValue = lines.join("\n");
	const widget = ComfyWidgets["STRING"](
		this,
		"loopback_status",
		["STRING", { multiline: true }],
		app
	).widget;
	widget.inputEl.readOnly = true;
	widget.inputEl.style.opacity = 0.7;
	widget.value = textValue;
	this[STATUS_WIDGETS].push(widget);

	requestAnimationFrame(() => {
		const size = this.computeSize();
		if (size[0] < this.size[0]) size[0] = this.size[0];
		if (size[1] < this.size[1]) size[1] = this.size[1];
		this.onResize?.(size);
		app.graph.setDirtyCanvas(true, false);
	});
}

app.registerExtension({
	name: "image_loopback.ui",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "Image-Loopback-Load") {
			return;
		}

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			if (message?.text) {
				this[STATUS_VALUES] = message.text;
				populateStatus.call(this, message.text);
			}
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			onConfigure?.apply(this, arguments);
			if (this[STATUS_VALUES]) {
				populateStatus.call(this, this[STATUS_VALUES]);
			}
		};
	},
});
