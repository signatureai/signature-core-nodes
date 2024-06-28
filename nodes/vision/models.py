
import torch
from ..categories import MODELS_CAT
from signature_core.img.tensor_image import TensorImage
from signature_core.models.lama import Lama
from signature_core.models.salient_object_detection import SalientObjectDetection
from signature_core.models.seemore import SeeMore
from nodes import SaveImage # type: ignore
import folder_paths # type: ignore
import random

class MagicEraser(SaveImage):
    def __init__(self):
        self.model = Lama()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "preview": (['on', 'off'],),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor, preview:str, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
        input_image = TensorImage.from_BWHC(image)
        print(mask.shape)
        input_mask = TensorImage.from_BWHC(mask)

        highres = TensorImage(self.model.forward(input_image, input_mask, "FIXED"))
        output_images = highres.get_BWHC()
        if preview == "off":
            return (output_images,)
        result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
        result.update({"result": (output_images,)})
        return result

class Unblur(SaveImage):
    def __init__(self):
        self.model = SeeMore()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "preview": (['on', 'off'],),
            }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT


    def process(self, image: torch.Tensor, preview:str, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
        input_image = TensorImage.from_BWHC(image)
        output_image = self.model.forward(input_image)
        output_images = TensorImage(output_image).get_BWHC()

        if preview == "off":
            return (output_images,)
        result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
        result.update({"result": (output_images,)})
        return result

class BackgroundRemoval(SaveImage):
    def __init__(self):
        self.model_name = "isnet"
        self.model: SalientObjectDetection | None = None
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "model_name": (['rmbg14','isnet_general'],),
            "preview": (['mask','rgba', 'none'],),
            "image": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "mask")
    FUNCTION = "process"
    CATEGORY = MODELS_CAT


    def process(self, image: torch.Tensor, model_name: str, preview:str, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
        if model_name != self.model_name or self.model is None:
            self.model = SalientObjectDetection(model_name=model_name)
            self.model_name = model_name

        input_image = TensorImage.from_BWHC(image)
        masks = self.model.forward(input_image)

        output_cutouts = torch.cat((input_image, masks), dim=1)
        output_masks = TensorImage(masks).get_BWHC()
        output_cutouts = TensorImage(output_cutouts).get_BWHC()
        if preview == "none":
            return (output_cutouts, output_masks,)
        preview_images = TensorImage(masks).get_rgb_or_rgba().get_BWHC() if preview == "mask" else output_cutouts
        result = self.save_images(preview_images, filename_prefix, prompt, extra_pnginfo)
        result.update({"result": (output_cutouts, output_masks,)})
        return result


NODE_CLASS_MAPPINGS = {
    "Signature Magic Eraser": MagicEraser,
    "Signature Background Removal": BackgroundRemoval,
    "Signature Unblur": Unblur,
}