from PIL import Image, ImageDraw, ImageFont
import hashlib
import json
import os
import numpy as np
import torch
from aiohttp import web

from comfy_api.latest import ComfyExtension, io, ui
import folder_paths
import comfy.utils
from server import PromptServer

DEFAULT_HISTORY_LIMIT = 64
HISTORY_DIR_NAME = "history"
HISTORY_INDEX_FILE = "history_index.txt"
CONFIG_FILE_NAME = "loopback_config.json"


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


def _config_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, CONFIG_FILE_NAME)


def _read_config(cache_dir: str) -> dict:
    path = _config_path(cache_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_config(cache_dir: str, data: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _config_path(cache_dir)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2)


def _effective_history_limit(cache_dir: str, override: int | None = None) -> int:
    if override is not None and override >= 0:
        return override
    config = _read_config(cache_dir)
    value = config.get("history_limit", DEFAULT_HISTORY_LIMIT)
    if isinstance(value, int) and value >= 0:
        return value
    return DEFAULT_HISTORY_LIMIT

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


# -----------------------------------------------------------------------------
# Live preview API endpoint - allows JS to query cached frames without execution
# -----------------------------------------------------------------------------
@PromptServer.instance.routes.post("/image_loopback/preview_cache")
async def preview_cache_handler(request):
    """
    Returns available history frame URLs for live preview in the UI.

    POST body (JSON):
        workflow: dict - the current workflow JSON (used to compute workflow_key)
        cache_path: str - the cache path input value
        history_indices: str - indices to preview (e.g. "1,2,3" or "1-3")

    Returns:
        JSON with:
            workflow_key: str - computed workflow key
            frames: list[dict] - frame info for each requested index
                Each frame: {index, exists, filename, subfolder, type}
            total_history: int - total frames in history
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    workflow = data.get("workflow", {})
    cache_path = data.get("cache_path", "loopback_cache")
    history_indices_str = data.get("history_indices", "")

    # Compute workflow key from workflow JSON
    workflow_key = _workflow_key_from_hidden(None, {"workflow": workflow})

    # Get cache directory
    cache_directory = _cache_dir(cache_path, workflow_key)
    history_directory = _history_dir(cache_directory)

    # List available history entries
    history_entries = _list_history_entries(history_directory)
    current_index = _read_history_index(history_directory)
    total_history = len(history_entries)

    # Build a map of available frames by their relative index (1 = most recent, etc.)
    # Frames are ordered by index descending (most recent first)
    sorted_entries = sorted(history_entries, key=lambda x: x[0], reverse=True)
    available_frames = {}  # relative_index -> (absolute_index, path)
    for rel_idx, (abs_idx, path) in enumerate(sorted_entries, start=1):
        available_frames[rel_idx] = (abs_idx, path)

    # Parse requested indices
    try:
        requested_indices = _parse_history_indices(history_indices_str)
    except ValueError:
        requested_indices = []

    # If no indices specified, show frame 1 by default (most recent)
    if not requested_indices:
        requested_indices = [1] if total_history > 0 else []

    # Build response frames
    frames = []
    temp_dir = folder_paths.get_temp_directory()

    for rel_idx in requested_indices:
        if rel_idx in available_frames:
            abs_idx, path = available_frames[rel_idx]
            # Make path relative to temp directory for the view endpoint
            rel_path = os.path.relpath(path, temp_dir)
            subfolder = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            frames.append({
                "index": rel_idx,
                "exists": True,
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp",
            })
        else:
            frames.append({
                "index": rel_idx,
                "exists": False,
                "filename": None,
                "subfolder": None,
                "type": None,
            })

    return web.json_response({
        "workflow_key": workflow_key,
        "frames": frames,
        "total_history": total_history,
        "current_index": current_index,
    })


class ImageLoopbackCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image-Loopback-Cache",
            display_name="Store Loopback",
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
        caching_enabled: bool,
    ) -> io.NodeOutput:
        if not caching_enabled:
            return io.NodeOutput()

        workflow_key = _node_workflow_key(cls)
        cache_dir = _cache_dir(cache_path, workflow_key)
        history_limit = _effective_history_limit(cache_dir)
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
            display_name="Load Loopback",
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
                io.Int.Input(
                    "history_limit",
                    default=-1,
                    min=-1,
                    tooltip="Override history limit for this workflow (-1 uses configured/default).",
                ),
                io.Combo.Input(
                    "empty_mode",
                    options=["error", "skip", "checkerboard", "black", "gray", "white", "transparent"],
                    default="checkerboard",
                    tooltip="What to output when requested frames are missing. 'skip' omits missing frames entirely (returns 1x1 void if all missing).",
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
        history_limit: int,
        empty_mode: str,
        empty_width: int,
        empty_height: int,
        starting_image: torch.Tensor | None = None,
    ) -> bool | str:
        try:
            _parse_history_indices(history_indices)
        except ValueError as exc:
            return str(exc)
        if empty_mode not in {"error", "skip", "checkerboard", "black", "gray", "white", "transparent"}:
            return f"Invalid empty_mode '{empty_mode}'."
        if history_limit < -1:
            return "history_limit must be -1 or a non-negative integer."
        if empty_width <= 0 or empty_height <= 0:
            return "empty_width and empty_height must be positive."
        return True

    @classmethod
    def execute(
        cls,
        cache_path: str,
        update_from_cache: bool,
        history_indices: str,
        history_limit: int,
        empty_mode: str,
        empty_width: int,
        empty_height: int,
        starting_image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        requested_history = _parse_history_indices(history_indices)
        workflow_key = _node_workflow_key(cls)
        cache_dir = _cache_dir(cache_path, workflow_key)
        if history_limit is not None and history_limit >= 0:
            _write_config(cache_dir, {"history_limit": history_limit})
        cached_image_path = os.path.join(cache_dir, "cached_img.png")
        current_image_path = os.path.join(cache_dir, "current_img.png")

        cache_exists = os.path.exists(cached_image_path)
        current_exists = os.path.exists(current_image_path)
        cache_had_image = cache_exists
        source = "unknown"
        empty_mode = (empty_mode or "error").lower()
        history_limit_effective = _effective_history_limit(cache_dir, history_limit)

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
                _append_history(image, cache_dir, history_limit_effective)
            elif current_exists:
                image = _load_tensor_image(current_image_path)
                source = "current"
            else:
                if empty_mode == "error":
                    raise FileNotFoundError(
                        "No cached image found. Provide a starting_image or run the cache node first."
                    )
                # "skip" mode in non-history case returns void tensor
                fill_mode = "transparent" if empty_mode == "skip" else empty_mode
                image = _make_placeholder_tensor(empty_width, empty_height, fill_mode)
                source = "void" if empty_mode == "skip" else "empty"
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
                _append_history(image, cache_dir, history_limit_effective)
            else:
                if empty_mode == "error":
                    raise FileNotFoundError(
                        "No cached image found. Provide a starting_image or run the cache node first."
                    )
                # "skip" mode in non-history case returns void tensor
                fill_mode = "transparent" if empty_mode == "skip" else empty_mode
                image = _make_placeholder_tensor(empty_width, empty_height, fill_mode)
                source = "void" if empty_mode == "skip" else "empty"

        output_image = image
        placeholder_used = source in ("empty", "void")
        # Label for non-history source (shown in preview)
        frame_labels = [f"[{source}]"] if source in ("starting", "cached", "current") else None
        skipped_indices = []
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
                elif empty_mode == "skip":
                    # Skip missing frames entirely
                    skipped_indices.append(index)
                else:
                    ref = image
                    ref_batch = _ensure_batch(ref)
                    height = ref_batch.shape[1] if ref_batch.dim() == 4 else empty_height
                    width = ref_batch.shape[2] if ref_batch.dim() == 4 else empty_width
                    selected.append(_make_placeholder_tensor(width, height, empty_mode))
                    frame_labels.append(f"t-{index} empty")
                    placeholder_used = True

            if selected:
                output_image = _batch_images(selected)
            else:
                # All frames were skipped - return a 1x1 void tensor
                output_image = _make_placeholder_tensor(1, 1, "transparent")
                frame_labels = ["void"]
                placeholder_used = True
            source = f"history[{history_indices.strip()}]"

        status_lines = [
            f"wf: {workflow_key}",
            f"cache: {'present' if cache_had_image else 'missing'}",
            f"source: {source}",
        ]
        if placeholder_used:
            status_lines.append(f"empty_mode: {empty_mode}")
        if skipped_indices:
            status_lines.append(f"skipped: {', '.join(f't-{i}' for i in skipped_indices)}")
        if output_image.dim() == 4 and output_image.shape[0] > 1:
            status_lines.append(f"batch: {output_image.shape[0]}")

        # Send status text only - JS widget handles preview display via live API
        status_text_lines = list(status_lines)
        if frame_labels:
            status_text_lines.append(f"frames: {', '.join(frame_labels)}")
        text_ui = ui.PreviewText("\n".join(status_text_lines))
        return io.NodeOutput(output_image, ui=text_ui.as_dict())


class ImageLoopbackConfigure(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image-Loopback-Configure",
            display_name="Configure Loopback",
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
                io.Int.Input(
                    "history_limit",
                    default=DEFAULT_HISTORY_LIMIT,
                    min=0,
                    tooltip="Number of cached frames to keep (0 keeps everything).",
                ),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def validate_inputs(cls, cache_path: str, history_limit: int) -> bool | str:
        if history_limit < 0:
            return "history_limit must be non-negative."
        return True

    @classmethod
    def execute(cls, cache_path: str, history_limit: int) -> io.NodeOutput:
        workflow_key = _node_workflow_key(cls)
        cache_dir = _cache_dir(cache_path, workflow_key)
        _write_config(cache_dir, {"history_limit": history_limit})
        text = "\n".join(
            [
                f"wf: {workflow_key}",
                f"cache_path: {cache_path}",
                f"history_limit: {history_limit}",
            ]
        )
        return io.NodeOutput(ui=ui.PreviewText(text))


class ImageLoopbackExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageLoopbackCache, ImageLoopbackLoad, ImageLoopbackConfigure]


async def comfy_entrypoint() -> ComfyExtension:
    return ImageLoopbackExtension()
