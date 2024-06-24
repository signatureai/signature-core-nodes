from .categories import TYPES_CAT
from .shared import any
import random
from nodes import SaveImage # type: ignore
import folder_paths # type: ignore

class Any2String():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = TYPES_CAT
    def process(self, input):
        return (str(input),)

class Any2Any():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = (any,)
    FUNCTION = "process"
    CATEGORY = TYPES_CAT
    def process(self, input):
        return (input,)


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

    CATEGORY = TYPES_CAT
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
    CATEGORY = TYPES_CAT

    def process(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


NODE_CLASS_MAPPINGS = {
    "Signature Any2Any": Any2Any,
    "Signature Any2String": Any2String,
    "Signature Text Preview": TextPreview,
    "Signature Mask Preview": MaskPreview
}