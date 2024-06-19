import torch
from signature_core.nodes.categories import MORPHOLOGY_CAT
from signature_core.img.tensor_image import TensorImage
from signature_core.functional.morphology import erosion, dilation


class MaskErosion:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_BWHC(mask)
        output = erosion(image=step, kernel_size=kernel_size, iterations=iterations).get_BWHC()
        return (output,)


class MaskDilation:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_BWHC(mask)
        output = dilation(image=step, kernel_size=kernel_size, iterations=iterations).get_BWHC()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Mask Erosion": MaskErosion,
    "Mask Dilation": MaskDilation,
}