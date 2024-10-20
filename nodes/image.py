import torch
from signature_core.functional.color import color_average
from signature_core.functional.filters import (
    gaussian_blur2d,
    image_soft_light,
    unsharp_mask,
)
from signature_core.img.tensor_image import TensorImage

from .categories import IMAGE_CAT


class BaseColor:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "hex_color": ("STRING", {"default": "#FFFFFF"}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        hex_color = kwargs.get("hex_color")
        width = kwargs.get("width")
        height = kwargs.get("height")
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        # Create a tensor with the specified color
        color_tensor = torch.tensor([r, g, b], dtype=torch.float32) / 255.0

        # Reshape to (3, 1, 1) and expand to (3, H, W)
        color_tensor = color_tensor.view(3, 1, 1).expand(3, height, width)

        # Repeat for the batch size
        batch_tensor = color_tensor.unsqueeze(0).expand(1, -1, -1, -1)

        output = TensorImage(batch_tensor).get_BWHC()
        return (output,)


class ImageGaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 13}),
                "sigma": ("FLOAT", {"default": 10.5}),
                "interations": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        image = kwargs.get("image")
        radius = kwargs.get("radius")
        sigma = kwargs.get("sigma")
        interations = kwargs.get("interations")
        tensor_image = TensorImage.from_BWHC(image)
        output = gaussian_blur2d(tensor_image, radius, sigma, interations).get_BWHC()
        return (output,)


class ImageUnsharpMask:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 3}),
                "sigma": ("FLOAT", {"default": 1.5}),
                "interations": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        image = kwargs.get("image")
        radius = kwargs.get("radius")
        sigma = kwargs.get("sigma")
        interations = kwargs.get("interations")
        tensor_image = TensorImage.from_BWHC(image)
        output = unsharp_mask(tensor_image, radius, sigma, interations).get_BWHC()
        return (output,)


class ImageSoftLight:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "top": ("IMAGE",),
                "bottom": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        top = kwargs.get("top")
        bottom = kwargs.get("bottom")
        top_tensor = TensorImage.from_BWHC(top)
        bottom_tensor = TensorImage.from_BWHC(bottom)
        output = image_soft_light(top_tensor, bottom_tensor).get_BWHC()

        return (output,)


class ImageAverage:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        image = kwargs.get("image")
        step = TensorImage.from_BWHC(image)
        output = color_average(step).get_BWHC()
        return (output,)


class ImageSubtract:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image_0": ("IMAGE",),
                "image_1": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        image_0 = kwargs.get("image_0")
        image_1 = kwargs.get("image_1")
        image_0_tensor = TensorImage.from_BWHC(image_0)
        image_1_tensor = TensorImage.from_BWHC(image_1)
        image_tensor = torch.abs(image_0_tensor - image_1_tensor)
        output = TensorImage(image_tensor).get_BWHC()
        return (output,)


NODE_CLASS_MAPPINGS = {
    "signature_image_gaussian_blur": ImageGaussianBlur,
    "signature_image_unsharp_mask": ImageUnsharpMask,
    "signature_image_soft_light": ImageSoftLight,
    "signature_image_average": ImageAverage,
    "signature_image_subtract": ImageSubtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_image_gaussian_blur": "SIG ImageGaussianBlur",
    "signature_image_unsharp_mask": "SIG ImageUnsharpMask",
    "signature_image_soft_light": "SIG ImageSoftLight",
    "signature_image_average": "SIG ImageAverage",
    "signature_image_subtract": "SIG ImageSubtract",
}
