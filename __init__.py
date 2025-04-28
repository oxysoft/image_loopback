"""Top-level package for image_loopback."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Leo Behe"""
__email__ = "leobehe@gmail.com"
__version__ = "0.0.1"

from .src.image_loopback.nodes import NODE_CLASS_MAPPINGS
from .src.image_loopback.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
