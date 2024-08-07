import torch
from signature_core.functional.morphology import dilation, erosion
from signature_core.img.tensor_image import TensorImage

from ..categories import MORPHOLOGY_CAT


class MaskErosion:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
                "kernel_size": ("INT", {"default": 3}),
                "iterations": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT

    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_BWHC(mask)
        output = erosion(image=step, kernel_size=kernel_size, iterations=iterations).get_BWHC()
        return (output,)


class MaskDilation:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
                "kernel_size": ("INT", {"default": 3}),
                "iterations": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT

    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_BWHC(mask)
        output = dilation(image=step, kernel_size=kernel_size, iterations=iterations).get_BWHC()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "signature_mask_erosion": MaskErosion,
    "signature_mask_dilation": MaskDilation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_mask_erosion": "SIG Mask Erosion",
    "signature_mask_dilation": "SIG Mask Dilation",
}
