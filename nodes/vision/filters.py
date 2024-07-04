import torch
from signature_core.img.tensor_image import TensorImage
from signature_core.functional.filters import gaussian_blur2d, unsharp_mask, image_soft_light
from ..categories import FILTER_CAT

class ImageGaussianBlur:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "radius": ("INT", {"default": 13}),
                             "sigma": ("FLOAT", {"default": 10.5}),
                             "interations": ("INT", {"default": 1}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        tensor_image = TensorImage.from_BWHC(image)
        output = gaussian_blur2d(tensor_image, radius, sigma, interations).get_BWHC()
        return (output,)

class ImageUnsharpMask:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "radius": ("INT", {"default": 3}),
                             "sigma": ("FLOAT", {"default": 1.5}),
                             "interations": ("INT", {"default": 1}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        tensor_image = TensorImage.from_BWHC(image)
        output = unsharp_mask(tensor_image, radius, sigma, interations).get_BWHC()
        return (output,)


class MaskGaussianBlur:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("MASK",),
                             "radius": ("INT", {"default": 13}),
                             "sigma": ("FLOAT", {"default": 10.5}),
                             "interations": ("INT", {"default": 1}),
                             "only_outline": ("BOOLEAN", {"default": False}),
                             }
                }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        tensor_image = TensorImage.from_BWHC(image)
        output = gaussian_blur2d(tensor_image, radius, sigma, interations).get_BWHC()
        return (output,)

class ImageSoftLight:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"top": ("IMAGE",),
                             "bottom": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, top: torch.Tensor, bottom:torch.Tensor):
        top_tensor = TensorImage.from_BWHC(top)
        bottom_tensor = TensorImage.from_BWHC(bottom)
        output = image_soft_light(top_tensor, bottom_tensor).get_BWHC()

        return (output,)


NODE_CLASS_MAPPINGS = {
    "signature_image_gaussian_blur": ImageGaussianBlur,
    "signature_image_unsharp_mask" : ImageUnsharpMask,
    "signature_mask_gaussian_blur": MaskGaussianBlur,
    "signature_image_soft_light": ImageSoftLight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_image_gaussian_blur": "SIG Image Gaussian Blur",
    "signature_image_unsharp_mask" : "SIG Image Unsharp Mask",
    "signature_mask_gaussian_blur": "SIG Mask Gaussian Blur",
    "signature_image_soft_light": "SIG Image Soft Light",
}