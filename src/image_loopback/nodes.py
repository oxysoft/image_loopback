from inspect import cleandoc
from PIL import Image
import os
import torch
import sys
import numpy as np

class ImageLoopbackCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": (
                    "IMAGE", 
                    {}
                ),
                "cache_path": ( 
                    "STRING", {
                        "multiline": False, 
                        "default": "loopback_cache"
                    },
                ),
                "caching_enabled": (
                    "BOOLEAN", {
                        "default": True
                    }
                )
            },
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    CATEGORY = "Utility"
    FUNCTION = "cache_image"

    def cache_image(self, input_image, cache_path: str, caching_enabled):
        # Check if caching is enabled
        if not caching_enabled:
            return ()

        # Create the cache directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, cache_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Set the cache image path
        image_path = os.path.join(cache_dir, "cached_img.png")

        # Convert the PyTorch tensor to a PIL Image
        img = input_image.squeeze()
        img = (img * 255).clamp(0, 255).byte().cpu().numpy()
        if img.shape[2] == 4:
            image = Image.fromarray(img, 'RGBA')
        else:
            image = Image.fromarray(img, 'RGB')

        #  Check if the previously cached image is the same as the current image
        if os.path.exists(image_path):
            cached_image = Image.open(image_path)
            cached_array = np.array(cached_image)
            current_array = np.array(image)
            if np.array_equal(cached_array, current_array):
                return ()

        # Save the image to the cache directory
        if os.path.exists(image_path):
            os.remove(image_path)
        image.save(image_path)
        
        # Return an empty tuple
        return ()
    
class ImageLoopbackLoad:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_path": (
                    "STRING", {
                        "multiline": False,
                        "default": "loopback_cache"
                    },
                ),
                "update_from_cache": (
                    "BOOLEAN", {
                        "default": True
                    }
                )
            },
        }    
        
    def load_loopback_image(self, cache_path: str, update_from_cache: bool):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, cache_path)
        cached_image_path = os.path.join(cache_dir, "cached_img.png")
        current_image_path = os.path.join(cache_dir, "current_img.png")

        # Check if updating from cache is enabled; if it's not, return the same image used in the previous execution.
        # The idea is to make sure that unnecessary re-executions aren't triggered by the cache updating its return value
        # when it doesn't have to. 
        if not update_from_cache:
            current_img = Image.open(current_image_path)
            current_img = np.array(current_img).astype(np.float32) / 255.0
            current_img = torch.from_numpy(current_img).unsqueeze(0)

            return (current_img,)

        else:
            # Keep a second copy of the image saved so that we can return it for 
            # subsequent executions if update_from_cache is disabled. We can continue
            # to update cached_image to the latest image, but in case the next execution
            # doesn't want to update the cached image, we can still return this current image.
            cached_img = Image.open(cached_image_path)
            current_img = cached_img.copy()
            current_img.save(current_image_path)

            # Convert the PIL Image to a PyTorch tensor, normalize it, and return it
            current_img = Image.open(current_image_path)
            current_img = np.array(current_img).astype(np.float32) / 255.0
            current_img = torch.from_numpy(current_img).unsqueeze(0)

            return (current_img,)

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("loopback_image",)
    CATEGORY = "Utility"
    FUNCTION = "load_loopback_image"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image-Loopback-Cache": ImageLoopbackCache, 
    "Image-Loopback-Load": ImageLoopbackLoad
    }

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image-Loopback-Cache": "Cache Image For Loopback",
    "Image-Loopback-Load": "Load Image For Loopback",
}
