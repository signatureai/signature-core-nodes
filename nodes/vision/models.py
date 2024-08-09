import random

import folder_paths  # type: ignore
import torch
from signature_core.functional.transform import cutout
from signature_core.img.tensor_image import TensorImage
from signature_core.models.lama import Lama
from signature_core.models.salient_object_detection import SalientObjectDetection
from signature_core.models.seemore import SeeMore

from nodes import SaveImage  # type: ignore

from ..categories import MODELS_CAT


class MagicEraser(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "preview": (["on", "off"],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        preview: str,
        filename_prefix="Signature",
        prompt=None,
        extra_pnginfo=None,
    ):
        model = Lama()
        input_image = TensorImage.from_BWHC(image)
        input_mask = TensorImage.from_BWHC(mask)
        highres = TensorImage(model.forward(input_image, input_mask, "FIXED"))
        output_images = highres.get_BWHC()
        if preview == "off":
            return (output_images,)
        result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
        result.update({"result": (output_images,)})
        del model
        model = None
        return result


class Unblur(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "preview": (["on", "off"],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(
        self, image: torch.Tensor, preview: str, filename_prefix="Signature", prompt=None, extra_pnginfo=None
    ):
        model = SeeMore()
        input_image = TensorImage.from_BWHC(image)
        output_image = model.forward(input_image)
        output_images = TensorImage(output_image).get_BWHC()

        if preview == "off":
            return (output_images,)
        result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
        result.update({"result": (output_images,)})
        del model
        model = None
        return result


class BackgroundRemoval(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "model_name": (["rmbg14", "isnet_general"],),
                "preview": (["mask", "rgba", "none"],),
                "image": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "rgb", "mask")
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(
        self,
        image: torch.Tensor,
        model_name: str,
        preview: str,
        filename_prefix="Signature",
        prompt=None,
        extra_pnginfo=None,
    ):

        model = SalientObjectDetection(model_name=model_name)
        input_image = TensorImage.from_BWHC(image)
        masks = model.forward(input_image)

        output_masks = TensorImage(masks)
        rgb, rgba = cutout(input_image, output_masks)
        rgb_output = TensorImage(rgb).get_BWHC()
        rgba_output = TensorImage(rgba).get_BWHC()
        mask_output = output_masks.get_BWHC()
        if preview == "none":
            return (
                rgba_output,
                rgb_output,
                mask_output,
            )
        preview_images = output_masks.get_rgb_or_rgba().get_BWHC() if preview == "mask" else rgba_output
        result = self.save_images(preview_images, filename_prefix, prompt, extra_pnginfo)
        result.update(
            {
                "result": (
                    rgba_output,
                    rgb_output,
                    mask_output,
                )
            }
        )
        del model
        model = None
        return result


NODE_CLASS_MAPPINGS = {
    "signature_magic_eraser": MagicEraser,
    "signature_background_removal": BackgroundRemoval,
    "signature_unblur": Unblur,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_magic_eraser": "SIG Magic Eraser",
    "signature_background_removal": "SIG Background Removal",
    "signature_unblur": "SIG Unblur",
}
