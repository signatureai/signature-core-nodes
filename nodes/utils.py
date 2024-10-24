import gc

import comfy.model_management as mm  # type: ignore
import torch
from signature_core.functional.color import rgb_to_hls, rgb_to_hsv, rgba_to_rgb
from signature_core.img.tensor_image import TensorImage

from .categories import UTILS_CAT
from .shared import any_type


class Any2String:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, value):
        return (str(value),)


class Any2Image:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, value):
        if isinstance(value, torch.Tensor):
            return (value,)
        raise ValueError(f"Unsupported type: {type(value)}")


class Any2Any:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, value):
        return (value,)


class RGB2HSV:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hsv(image_tensor).get_BWHC()
        return (output,)


class RGBHLS:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hls(image_tensor).get_BWHC()
        return (output,)


class RGBA2RGB:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        if image_tensor.shape[1] == 4:
            image_tensor = rgba_to_rgb(image_tensor)
        output = image_tensor.get_BWHC()
        return (output,)


class PurgeVRAM:

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "anything": (any_type, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    OUTPUT_NODE = True

    def execute(self, anything, purge_cache, purge_models):

        if purge_cache:

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if purge_models:
            mm.unload_all_models()
            mm.soft_empty_cache(True)
        return (None,)
