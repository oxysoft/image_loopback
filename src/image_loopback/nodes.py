from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import torch

from comfy_api.latest import ComfyExtension, io, ui
import folder_paths


def _cache_dir(cache_path: str) -> str:
    if os.path.isabs(cache_path):
        return cache_path
    base_dir = os.path.abspath(os.path.join(folder_paths.get_temp_directory(), "image_loopback"))
    os.makedirs(base_dir, exist_ok=True)
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_dir, cache_path)))
    if not (candidate == base_dir or candidate.startswith(base_dir + os.sep)):
        return base_dir
    return candidate


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


def _make_status_preview(image: torch.Tensor, status_lines: list[str]) -> torch.Tensor:
    preview = _tensor_to_pil(image).convert("RGB")
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
    return _pil_to_tensor(preview)


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
                    tooltip="Subfolder under ComfyUI temp/image_loopback (or absolute path).",
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
    def execute(cls, input_image, cache_path: str, caching_enabled: bool) -> io.NodeOutput:
        if not caching_enabled:
            return io.NodeOutput()

        cache_dir = _cache_dir(cache_path)
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
                    tooltip="Subfolder under ComfyUI temp/image_loopback (or absolute path).",
                ),
                io.Boolean.Input(
                    "update_from_cache",
                    default=True,
                    tooltip="If disabled, reuse the last loaded image from disk.",
                ),
                io.Image.Input(
                    "starting_image",
                    optional=True,
                    tooltip="Optional image to seed the cache when none exists.",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(
        cls,
        cache_path: str,
        update_from_cache: bool,
        starting_image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        cache_dir = _cache_dir(cache_path)
        cached_image_path = os.path.join(cache_dir, "cached_img.png")
        current_image_path = os.path.join(cache_dir, "current_img.png")

        cache_exists = os.path.exists(cached_image_path)
        current_exists = os.path.exists(current_image_path)
        cache_had_image = cache_exists
        source = "unknown"

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
            elif current_exists:
                image = _load_tensor_image(current_image_path)
                source = "current"
            else:
                raise FileNotFoundError(
                    "No cached image found. Provide a starting_image or run the cache node first."
                )
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
            else:
                raise FileNotFoundError(
                    "No cached image found. Provide a starting_image or run the cache node first."
                )

        status_lines = [
            f"cache: {'present' if cache_had_image else 'missing'}",
            f"source: {source}",
            f"update_from_cache: {update_from_cache}",
        ]
        preview_image = _make_status_preview(image, status_lines)
        return io.NodeOutput(image, ui=ui.PreviewImage(preview_image, cls=cls))


class ImageLoopbackExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ImageLoopbackCache, ImageLoopbackLoad]


async def comfy_entrypoint() -> ComfyExtension:
    return ImageLoopbackExtension()
