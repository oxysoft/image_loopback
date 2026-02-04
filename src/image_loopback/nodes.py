from PIL import Image, ImageDraw, ImageFont
import hashlib
import json
import os
import numpy as np
import torch

from comfy_api.latest import ComfyExtension, io, ui
import folder_paths
import comfy.utils

DEFAULT_HISTORY_LIMIT = 64
HISTORY_DIR_NAME = "history"
HISTORY_INDEX_FILE = "history_index.txt"


def _workflow_key_from_hidden(prompt: object | None, extra_pnginfo: object | None) -> str:
    payload = None
    if isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
        payload = extra_pnginfo.get("workflow")
    if payload is None:
        payload = prompt if prompt is not None else {}
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError):
        canonical = json.dumps(str(payload), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"wf_{digest}"


def _cache_dir(cache_path: str, workflow_key: str) -> str:
    workflow_key = workflow_key or "wf_default"
    if os.path.isabs(cache_path):
        base_root = os.path.abspath(os.path.join(cache_path, workflow_key))
        os.makedirs(base_root, exist_ok=True)
        return base_root

    base_root = os.path.abspath(
        os.path.join(folder_paths.get_temp_directory(), "image_loopback", workflow_key)
    )
    os.makedirs(base_root, exist_ok=True)
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_root, cache_path)))
    if not (candidate == base_root or candidate.startswith(base_root + os.sep)):
        return base_root
    return candidate


def _history_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, HISTORY_DIR_NAME)


def _history_path(history_dir: str, index: int) -> str:
    return os.path.join(history_dir, f"history_{index:06d}.png")


def _read_history_index(history_dir: str) -> int:
    index_path = os.path.join(history_dir, HISTORY_INDEX_FILE)
    if not os.path.exists(index_path):
        return 0
    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def _write_history_index(history_dir: str, index: int) -> None:
    os.makedirs(history_dir, exist_ok=True)
    index_path = os.path.join(history_dir, HISTORY_INDEX_FILE)
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(str(index))


def _list_history_entries(history_dir: str) -> list[tuple[int, str]]:
    if not os.path.isdir(history_dir):
        return []
    entries: list[tuple[int, str]] = []
    for name in os.listdir(history_dir):
        if not (name.startswith("history_") and name.endswith(".png")):
            continue
        raw = name[len("history_") : -len(".png")]
        if not raw.isdigit():
            continue
        index = int(raw)
        entries.append((index, os.path.join(history_dir, name)))
    entries.sort(key=lambda item: item[0])
    return entries


def _append_history(image: torch.Tensor, cache_dir: str, history_limit: int) -> None:
    history_dir = _history_dir(cache_dir)
    os.makedirs(history_dir, exist_ok=True)
    next_index = _read_history_index(history_dir) + 1
    _save_tensor_image(image, _history_path(history_dir, next_index))
    _write_history_index(history_dir, next_index)

    if history_limit <= 0:
        return
    entries = _list_history_entries(history_dir)
    if len(entries) <= history_limit:
        return
    excess = entries[:-history_limit]
    for _, path in excess:
        try:
            os.remove(path)
        except OSError:
            pass


def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() == 4:
        return image
    raise ValueError("Expected an IMAGE tensor with 3 or 4 dimensions.")


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = _ensure_batch(image)[0].detach().cpu()
    if image.dtype != torch.float32:
        image = image.float()
    image = (image * 255.0).clamp(0, 255).byte().numpy()
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return Image.fromarray(image, "RGBA")
    return Image.fromarray(image, "RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return torch.from_numpy(arr).unsqueeze(0)


def _save_tensor_image(image: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _tensor_to_pil(image).save(path)


def _load_tensor_image(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        return _pil_to_tensor(img)


def _draw_status_overlay(
    preview: Image.Image,
    status_lines: list[str],
    frame_label: str | None = None,
) -> Image.Image:
    preview = preview.convert("RGB")
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    padding = 6
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1]
    rect_height = (line_height * len(status_lines)) + (padding * 2)
    draw.rectangle([0, 0, preview.width, rect_height], fill=(0, 0, 0))
    y = padding
    for line in status_lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    if frame_label:
        label_text = frame_label
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        x0 = padding
        y0 = preview.height - label_h - (padding * 2)
        draw.rectangle(
            [x0, y0, x0 + label_w + (padding * 2), y0 + label_h + (padding * 2)],
            fill=(0, 0, 0),
        )
        draw.text((x0 + padding, y0 + padding), label_text, fill=(255, 255, 255), font=font)

    return preview


def _make_status_preview(
    image: torch.Tensor,
    status_lines: list[str],
    frame_labels: list[str] | None = None,
) -> torch.Tensor:
    batch = _ensure_batch(image)
    if frame_labels is None:
        frame_labels = [None] * batch.shape[0]
    if len(frame_labels) != batch.shape[0]:
        raise ValueError("Frame labels must match the batch size for previews.")

    previews: list[torch.Tensor] = []
    for idx, frame in enumerate(batch):
        preview = _tensor_to_pil(frame)
        preview = _draw_status_overlay(preview, status_lines, frame_labels[idx])
        previews.append(_pil_to_tensor(preview))
    return torch.cat(previews, dim=0)


def _parse_history_indices(spec: str) -> list[int]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    indices: list[int] = []
    for part in parts:
        if "-" in part:
            bounds = [item.strip() for item in part.split("-", maxsplit=1)]
            if len(bounds) != 2 or not bounds[0] or not bounds[1]:
                raise ValueError(f"Invalid history range: '{part}'.")
            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError as exc:
                raise ValueError(f"Invalid history range: '{part}'.") from exc
            step = 1 if end >= start else -1
            indices.extend(list(range(start, end + step, step)))
        else:
            try:
                indices.append(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid history index: '{part}'.") from exc
    if any(index == 0 for index in indices):
        raise ValueError("History index 0 is invalid (cannot sample the current run).")
    if any(index < 0 for index in indices):
        raise ValueError("History indices must be positive integers.")
    return indices


def _node_workflow_key(node_cls) -> str:
    hidden = getattr(node_cls, "hidden", None)
    prompt = getattr(hidden, "prompt", None) if hidden is not None else None
    extra_pnginfo = getattr(hidden, "extra_pnginfo", None) if hidden is not None else None
    return _workflow_key_from_hidden(prompt, extra_pnginfo)


def _batch_images(images: list[torch.Tensor]) -> torch.Tensor:
    if len(images) == 0:
        raise ValueError("No images provided for batching.")
    max_channels = max(image.shape[-1] for image in images)
    padded_images: list[torch.Tensor] = []
    for image in images:
        if image.shape[-1] < max_channels:
            padded_images.append(torch.nn.functional.pad(image, (0, 1), mode="constant", value=1.0))
        else:
            padded_images.append(image)
    resized_images: list[torch.Tensor] = []
    first_image_shape = padded_images[0].shape
    for image in padded_images:
        if image.shape[1:] != first_image_shape[1:]:
            resized_images.append(
                comfy.utils.common_upscale(
                    image.movedim(-1, 1),
                    first_image_shape[2],
                    first_image_shape[1],
                    "bilinear",
                    "center",
                ).movedim(1, -1)
            )
        else:
            resized_images.append(image)
    return torch.cat(resized_images, dim=0)


def _make_placeholder_tensor(
    width: int,
    height: int,
    mode: str,
) -> torch.Tensor:
    if width <= 0 or height <= 0:
        raise ValueError("Placeholder dimensions must be positive.")
    if mode == "transparent":
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    elif mode == "white":
        image = Image.new("RGB", (width, height), (255, 255, 255))
    elif mode == "gray":
        image = Image.new("RGB", (width, height), (64, 64, 64))
    elif mode == "checkerboard":
        image = Image.new("RGB", (width, height), (48, 48, 48))
        draw = ImageDraw.Draw(image)
        tile = max(16, min(width, height) // 8)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                if (x // tile + y // tile) % 2 == 0:
                    draw.rectangle([x, y, x + tile - 1, y + tile - 1], fill=(96, 96, 96))
    else:
        image = Image.new("RGB", (width, height), (0, 0, 0))
    return _pil_to_tensor(image)


class ImageLoopbackCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image-Loopback-Cache",
            display_name="Cache Image For Loopback",
            category="Utility",
            inputs=[
                io.Image.Input("input_image", tooltip="Image to store as the loopback cache."),
                io.String.Input(
                    "cache_path",
                    default="loopback_cache",
                    multiline=False,
                    tooltip=(
                        "Subfolder under this workflow's temp cache. "
                        "Use different values for independent buffers."
                    ),
                ),
                io.Int.Input(
                    "history_limit",
                    default=DEFAULT_HISTORY_LIMIT,
                    min=0,
                    tooltip="Number of cached frames to keep (0 keeps everything).",
                ),
                io.Boolean.Input(
                    "caching_enabled",
                    default=True,
                    tooltip="Disable to skip writing the cached image.",
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        input_image,
        cache_path: str,
        history_limit: int,
        caching_enabled: bool,
    ) -> io.NodeOutput:
        if not caching_enabled:
            return io.NodeOutput()

        workflow_key = _node_workflow_key(cls)
        cache_dir = _cache_dir(cache_path, workflow_key)
        os.makedirs(cache_dir, exist_ok=True)
        image_path = os.path.join(cache_dir, "cached_img.png")

        image = _tensor_to_pil(input_image)
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as cached_image:
                    if np.array_equal(np.array(cached_image), np.array(image)):
                        return io.NodeOutput()
            except OSError:
                pass

        image.save(image_path)
        _append_history(input_image, cache_dir, history_limit)
        return io.NodeOutput()


class ImageLoopbackLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image-Loopback-Load",
            display_name="Load Image For Loopback",
            category="Utility",
            inputs=[
                io.String.Input(
                    "cache_path",
                    default="loopback_cache",
                    multiline=False,
                    tooltip=(
                        "Subfolder under this workflow's temp cache. "
                        "Use different values for independent buffers."
                    ),
                ),
                io.Boolean.Input(
                    "update_from_cache",
                    default=True,
                    tooltip="If disabled, reuse the last loaded image from disk.",
                ),
                io.String.Input(
                    "history_indices",
                    default="",
                    multiline=False,
                    tooltip="Indices or ranges (e.g. '1,3-4') for past frames; 0 is invalid.",
                ),
                io.Combo.Input(
                    "empty_mode",
                    options=["error", "checkerboard", "black", "gray", "white", "transparent"],
                    default="checkerboard",
                    tooltip="What to output when requested frames are missing.",
                ),
                io.Int.Input(
                    "empty_width",
                    default=512,
                    min=16,
                    tooltip="Width for empty frames when no reference image exists.",
                ),
                io.Int.Input(
                    "empty_height",
                    default=512,
                    min=16,
                    tooltip="Height for empty frames when no reference image exists.",
                ),
                io.Image.Input(
                    "starting_image",
                    optional=True,
                    tooltip="Optional image to seed the cache when none exists.",
                ),
            ],
            outputs=[io.Image.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def validate_inputs(
        cls,
        cache_path: str,
        update_from_cache: bool,
        history_indices: str,
        empty_mode: str,
        empty_width: int,
        empty_height: int,
        starting_image: torch.Tensor | None = None,
    ) -> bool | str:
        try:
            _parse_history_indices(history_indices)
        except ValueError as exc:
            return str(exc)
        if empty_mode not in {"error", "checkerboard", "black", "gray", "white", "transparent"}:
            return f"Invalid empty_mode '{empty_mode}'."
        if empty_width <= 0 or empty_height <= 0:
            return "empty_width and empty_height must be positive."
        return True

    @classmethod
    def execute(
        cls,
        cache_path: str,
        update_from_cache: bool,
        history_indices: str,
        empty_mode: str,
        empty_width: int,
        empty_height: int,
        starting_image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        requested_history = _parse_history_indices(history_indices)
        workflow_key = _node_workflow_key(cls)
        cache_dir = _cache_dir(cache_path, workflow_key)
        cached_image_path = os.path.join(cache_dir, "cached_img.png")
        current_image_path = os.path.join(cache_dir, "current_img.png")

        cache_exists = os.path.exists(cached_image_path)
        current_exists = os.path.exists(current_image_path)
        cache_had_image = cache_exists
        source = "unknown"
        empty_mode = (empty_mode or "error").lower()

        if update_from_cache:
            if cache_exists:
                image = _load_tensor_image(cached_image_path)
                source = "cached"
                _save_tensor_image(image, current_image_path)
                current_exists = True
            elif starting_image is not None:
                image = _ensure_batch(starting_image)
                source = "starting"
                _save_tensor_image(image, current_image_path)
                _save_tensor_image(image, cached_image_path)
                cache_exists = True
                current_exists = True
                _append_history(image, cache_dir, DEFAULT_HISTORY_LIMIT)
            elif current_exists:
                image = _load_tensor_image(current_image_path)
                source = "current"
            else:
                if empty_mode == "error":
                    raise FileNotFoundError(
                        "No cached image found. Provide a starting_image or run the cache node first."
                    )
                image = _make_placeholder_tensor(empty_width, empty_height, empty_mode)
                source = "empty"
        else:
            if current_exists:
                image = _load_tensor_image(current_image_path)
                source = "current"
            elif cache_exists:
                image = _load_tensor_image(cached_image_path)
                source = "cached"
                _save_tensor_image(image, current_image_path)
                current_exists = True
            elif starting_image is not None:
                image = _ensure_batch(starting_image)
                source = "starting"
                _save_tensor_image(image, current_image_path)
                _save_tensor_image(image, cached_image_path)
                cache_exists = True
                current_exists = True
                _append_history(image, cache_dir, DEFAULT_HISTORY_LIMIT)
            else:
                if empty_mode == "error":
                    raise FileNotFoundError(
                        "No cached image found. Provide a starting_image or run the cache node first."
                    )
                image = _make_placeholder_tensor(empty_width, empty_height, empty_mode)
                source = "empty"

        output_image = image
        placeholder_used = source == "empty"
        frame_labels = None
        if requested_history:
            history_dir = _history_dir(cache_dir)
            history_entries = _list_history_entries(history_dir)
            total_available = len(history_entries)
            selected: list[torch.Tensor] = []
            frame_labels = []
            for index in requested_history:
                if total_available and index <= total_available:
                    _, path = history_entries[-index]
                    selected.append(_load_tensor_image(path))
                    frame_labels.append(f"t-{index}")
                elif empty_mode == "error":
                    raise ValueError(
                        f"History only has {total_available} frame(s); requested t-{index}."
                    )
                else:
                    ref = image
                    ref_batch = _ensure_batch(ref)
                    height = ref_batch.shape[1] if ref_batch.dim() == 4 else empty_height
                    width = ref_batch.shape[2] if ref_batch.dim() == 4 else empty_width
                    selected.append(_make_placeholder_tensor(width, height, empty_mode))
                    frame_labels.append(f"t-{index} empty")
                    placeholder_used = True

            output_image = _batch_images(selected)
            source = f"history[{history_indices.strip()}]"

        status_lines = [
            f"wf: {workflow_key}",
            f"cache: {'present' if cache_had_image else 'missing'}",
            f"source: {source}",
            f"update_from_cache: {update_from_cache}",
        ]
        if placeholder_used:
            status_lines.append(f"empty_mode: {empty_mode}")
        if requested_history:
            status_lines.append(f"history: {history_indices.strip()}")
        if output_image.dim() == 4 and output_image.shape[0] > 1:
            status_lines.append(f"batch: {output_image.shape[0]}")
        preview_image = _make_status_preview(output_image, status_lines, frame_labels)
        preview_ui = ui.PreviewImage(preview_image, cls=cls)
        status_text_lines = list(status_lines)
        if frame_labels:
            status_text_lines.append(f"frames: {', '.join(frame_labels)}")
        text_ui = ui.PreviewText("\n".join(status_text_lines))
        ui_payload = preview_ui.as_dict()
        ui_payload.update(text_ui.as_dict())
        return io.NodeOutput(output_image, ui=ui_payload)


class ImageLoopbackExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageLoopbackCache, ImageLoopbackLoad]


async def comfy_entrypoint() -> ComfyExtension:
    return ImageLoopbackExtension()
