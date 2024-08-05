from .categories import UTILS_CAT
from .shared import any
import random
import gc
from nodes import SaveImage # type: ignore
import comfy # type: ignore
import folder_paths # type: ignore
import model_management # type: ignore
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

        return (result,)


class TextPreview():
    @classmethod
    def INPUT_TYPES(s):  # type: ignore
        return {
            "required": {
                "text": (any,),
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = UTILS_CAT
    def process(self, text):
        print(len(text))
        text_string = []
        for t in text:
            text_string.append(str(t))


        return {"ui": {"text": text_string}, "result": (text_string,)}

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

class ImageShape:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "width", "height", "channels", "debug")
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, image):
        return (image.shape[0], image.shape[2], image.shape[1], image.shape[3], str(image.shape))

class MaskShape:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("batch", "width", "height", "channels", "debug")
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, mask):
        if len(mask.shape) == 3:
            return (mask.shape[0], mask.shape[2], mask.shape[1], 1, str(mask.shape))
        return (mask.shape[0], mask.shape[2], mask.shape[1], mask.shape[3], str(mask.shape))

class UnloadCheckpoint:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": (any, {}),
                "model": ("MODEL",),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = UTILS_CAT
    def process(self, value, model):
        model.model_patches_to("cpu")
        model_management.free_memory(1e300, model_management.get_torch_device())
        _ = model_management.get_free_memory()

        model_management.soft_empty_cache()
        model_management.unload_all_models()
        del model
        return (None,)

class CachedCheckpointLoader:
    def __init__(self):
        self.models = folder_paths.get_filename_list("checkpoints")
        self.cached_models = self.cache_models(self.models)
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "process"
    CATEGORY = UTILS_CAT

    def free_vram(self):
        _ = model_management.get_free_memory()
        model_management.soft_empty_cache()
        model_management.unload_all_models()
        gc.collect()
    def process(self, ckpt_name):
        selected_device = comfy.model_management.get_torch_device()
        ckpt = self.cached_models[ckpt_name]
        ckpt.model_patches_to(selected_device)
        return (ckpt, )

    def cache_models(self, models_list) -> dict:
        cached_models = {}
        for model in models_list:
            ckpt = self.load_checkpoint(model)[0]
            if ckpt is None:
                continue
            ckpt.model_patches_to("cpu")
            cached_models.update({model: ckpt})
            self.free_vram()
        return cached_models

    def load_checkpoint(self, ckpt_name):
        self.free_vram()
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=False, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

NODE_CLASS_MAPPINGS = {
    "signature_any2any": Any2Any,
    "signature_any2string": Any2String,
    "signature_string2case": String2Case,
    "signature_text_preview": TextPreview,
    "signature_mask_preview": MaskPreview,
    "signature_console_debug": ConsoleDebug,
    "signature_get_image_size": ImageShape,
    "signature_get_mask_size": MaskShape,
    "signature_unload_checkpoint": UnloadCheckpoint,
    "signature_cached_checkpoint_loader": CachedCheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_any2any": "SIG Any2Any",
    "signature_any2string": "SIG Any2String",
    "signature_string2case": "SIG String2Case",
    "signature_text_preview": "SIG Text Preview",
    "signature_mask_preview": "SIG Mask Preview",
    "signature_console_debug": "SIG Console Debug",
    "signature_get_image_size": "SIG Get Image Shape",
    "signature_get_mask_size": "SIG Get Mask Shape",
    "signature_unload_checkpoint": "SIG Unload Checkpoint (TEST)",
    "signature_cached_checkpoint_loader": "SIG Cached Checkpoint Loader (TEST)"
}