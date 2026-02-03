"""Top-level package for image_loopback."""

__all__ = [
    "comfy_entrypoint",
    "WEB_DIRECTORY",
]

__author__ = """Leo Behe"""
__email__ = "leobehe@gmail.com"
__version__ = "0.0.1"

from .src.image_loopback.nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"
