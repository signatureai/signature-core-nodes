from .categories import UTILS_CAT
from .shared import any
import random
from nodes import SaveImage # type: ignore
import folder_paths # type: ignore
import torch
from signature_core.img.tensor_image import TensorImage

class Any2String():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, input):
        return (str(input),)

class ImageBatch2List():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "batch": ("IMAGE",),
            }}
    RETURN_TYPES = ('LIST',)
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, batch: torch.Tensor):
        image_list = []
        for image in batch:
            image_list.append(image.unsqueeze(0))
        return (image_list,)

class Any2Any():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = (any,)
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, input):
        return (input,)

class String2Case():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "case": (['lower','upper','capitalize'],),
            },
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = UTILS_CAT


    def process(self, text:str, case:str):
        result = text
        if case == "lower":
            result = text.lower()
        if case == "upper":
            result = text.upper()
        if case == "capitalize":
            result = text.capitalize()
        print(result)
        return (result,)


class TextPreview():
    @classmethod
    def INPUT_TYPES(s):  # type: ignore
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = UTILS_CAT
    def process(self, text):
        return {"ui": {"text": text}, "result": (text,)}

class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {"mask": ("MASK",), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "process"
    CATEGORY = UTILS_CAT

    def process(self, mask, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
        preview = TensorImage.from_BWHC(mask).get_rgb_or_rgba().get_BWHC()
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


class ConsoleDebug:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": (any, {}),
            },
            "optional": {
                "prefix": ("STRING", { "multiline": False, "default": "Value:" })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    OUTPUT_NODE = True

    def process(self, value, prefix):
        print(f"\033[96m{prefix} {value}\033[0m")
        return (None,)

class ImageSize:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, image):
        return (image.shape[2], image.shape[1], )

NODE_CLASS_MAPPINGS = {
    "Signature Any2Any": Any2Any,
    "Signature Any2String": Any2String,
    "Signature String2Case": String2Case,
    "Signature Text Preview": TextPreview,
    "Signature Mask Preview": MaskPreview,
    "Signature Console Debug": ConsoleDebug,
    "Signature Image Batch2List": ImageBatch2List,
    "Signature Get Image Size": ImageSize,
}