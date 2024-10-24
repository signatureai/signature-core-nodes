import gc
import random

import comfy.model_management as mm  # type: ignore
import folder_paths  # type: ignore
import torch
from signature_core.functional.color import rgb_to_hls, rgb_to_hsv, rgba_to_rgb
from signature_core.img.tensor_image import TensorImage

from nodes import SaveImage  # type: ignore

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


class String2Case:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "case": (["lower", "upper", "capitalize"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, text: str, case: str):
        result = text
        if case == "lower":
            result = text.lower()
        if case == "upper":
            result = text.upper()
        if case == "capitalize":
            result = text.capitalize()

        return (result,)


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


class TextPreview:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "text": (any_type,),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = UTILS_CAT

    def execute(self, text):
        text_string = ""

        for t in text:
            if t is None:
                continue
            if text_string != "":
                text_string += "\n"
            text_string += str(t.shape) if isinstance(t, torch.Tensor) else str(t)
        return {"ui": {"text": [text_string]}, "result": (text_string,)}


class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, mask, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
        preview = TensorImage.from_BWHC(mask).get_rgb_or_rgba().get_BWHC()
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


class ConsoleDebug:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type, {}),
            },
            "optional": {"prefix": ("STRING", {"multiline": False, "default": "Value:"})},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    OUTPUT_NODE = True

    def execute(self, value, prefix):
        print(f"\033[96m{prefix} {value}\033[0m")
        return (None,)


class GetImageShape:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "width", "height", "channels", "debug")
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, image):
        return (image.shape[0], image.shape[2], image.shape[1], image.shape[3], str(image.shape))


class GetMaskShape:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "width", "height", "channels", "debug")
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, mask):
        if len(mask.shape) == 3:
            return (mask.shape[0], mask.shape[2], mask.shape[1], 1, str(mask.shape))
        return (mask.shape[0], mask.shape[2], mask.shape[1], mask.shape[3], str(mask.shape))


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
