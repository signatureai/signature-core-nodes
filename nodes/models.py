import torch
from .categories import MODELS_CAT
from signature_core.img.tensor_image import TensorImage
from signature_core.models.lama import Lama
from signature_core.models.salient_object_detection import SalientObjectDetection
from signature_core.models.seemore import SeeMore

class MagicEraser:
    def __init__(self):
        self.model = Lama()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor):
        input_image = TensorImage.from_BWHC(image)
        input_mask = TensorImage.from_BWHC(mask)

        highres = self.model.forward(input_image, input_mask, "FIXED")
        highres = TensorImage(highres).get_BWHC()

        return (highres,)

class Unblur:
    def __init__(self):
        self.model = SeeMore()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT


    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_BWHC(image)
        output_image = self.model.forward(input_image)

        output_cutouts = TensorImage(output_image).get_BWHC()
        return (output_cutouts,)

class BackgroundRemoval:
    def __init__(self):
        self.model_name = "isnet"
        self.model: SalientObjectDetection | None = None

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "model_name": (['rmbg14','isnet_general'],),
            "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "mask")
    FUNCTION = "process"
    CATEGORY = MODELS_CAT


    def process(self, image: torch.Tensor, model_name: str):
        if model_name != self.model_name or self.model is None:
            self.model = SalientObjectDetection(model_name=model_name)
            self.model_name = model_name

        input_image = TensorImage.from_BWHC(image)
        output_masks = self.model.forward(input_image)

        output_cutouts = torch.cat((input_image, output_masks), dim=1)
        output_masks = TensorImage(output_masks).get_BWHC()
        output_cutouts = TensorImage(output_cutouts).get_BWHC()
        return (output_cutouts, output_masks,)

NODE_CLASS_MAPPINGS = {
    "Signature Magic Eraser": MagicEraser,
    "Signature Background Removal": BackgroundRemoval,
    "Signature Unblur": Unblur,
}