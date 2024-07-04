import torch
from ..categories import ENHANCE_CAT
from signature_core.img.tensor_image import TensorImage
from signature_core.functional.enhance import adjust_brightness, adjust_saturation, equalize, equalize_clahe

class AdjustBrightness:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor, factor: float):
        tensor_image = TensorImage.from_BWHC(image)
        output = adjust_brightness(tensor_image, factor).get_BWHC()
        return (output,)


class AdjustSaturation:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor, factor: float):
        tensor_image = TensorImage.from_BWHC(image)
        output = adjust_saturation(tensor_image, factor).get_BWHC()
        return (output,)

class Equalize:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor):
        tensor_image = TensorImage.from_BWHC(image)
        output = equalize(tensor_image).get_BWHC()
        return (output,)


class EqualizeClahe:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor):
        tensor_image = TensorImage.from_BWHC(image)
        output = equalize_clahe(tensor_image).get_BWHC()
        return (output,)

NODE_CLASS_MAPPINGS = {
    "signature_adjust_brightness": AdjustBrightness,
    "signature_adjust_saturation": AdjustSaturation,
    "signature_equalize": Equalize,
    "signature_equalize_clahe": EqualizeClahe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_adjust_brightness": "SIG Adjust Brightness",
    "signature_adjust_saturation": "SIG Adjust Saturation",
    "signature_equalize": "SIG Equalize",
    "signature_equalize_clahe": "SIG Equalize Clahe",
}