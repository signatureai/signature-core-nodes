import torch
from signature_core.functional.color import (
    color_average,
    rgb_to_hls,
    rgb_to_hsv,
    rgba_to_rgb,
)
from signature_core.img.tensor_image import TensorImage

from ..categories import COLOR_CAT


class RGB2HSV:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hsv(image_tensor).get_BWHC()
        return (output,)


class RGBHLS:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hls(image_tensor).get_BWHC()
        return (output,)


class RGBA2RGB:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        if image_tensor.shape[1] == 4:
            image_tensor = rgba_to_rgb(image_tensor)
        output = image_tensor.get_BWHC()
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
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        step = TensorImage.from_BWHC(image)
        output = color_average(step).get_BWHC()
        return (output,)


class Image2Mask:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"image": ("IMAGE",), "channel": (["red", "green", "blue", "alpha"],)}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor, channel: str):
        image_tensor = TensorImage.from_BWHC(image)
        if channel == "red":
            image_tensor = image_tensor[:, 0, :, :].unsqueeze(1)
        if channel == "green":
            image_tensor = image_tensor[:, 1, :, :].unsqueeze(1)
        if channel == "blue":
            image_tensor = image_tensor[:, 2, :, :].unsqueeze(1)
        if channel == "alpha":
            image_tensor = image_tensor[:, 3, :, :].unsqueeze(1)
        output = TensorImage(image_tensor).get_BWHC()
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
    CATEGORY = COLOR_CAT

    def process(self, image_0: torch.Tensor, image_1: torch.Tensor):
        image_0_tensor = TensorImage.from_BWHC(image_0)
        image_1_tensor = TensorImage.from_BWHC(image_1)
        image_tensor = torch.abs(image_0_tensor - image_1_tensor)
        output = TensorImage(image_tensor).get_BWHC()
        return (output,)


class Mask2Image:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, mask: torch.Tensor):
        mask_tensor = TensorImage.from_BWHC(mask)
        output = mask_tensor.repeat(1, 3, 1, 1)
        output = TensorImage(output).get_BWHC()
        return (output,)


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
    CATEGORY = COLOR_CAT

    def process(self, hex_color: str, width: int, height: int):
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


NODE_CLASS_MAPPINGS = {
    "signature_rgb2hsv": RGB2HSV,
    "signature_rgbhls": RGBHLS,
    "signature_rgba2rgb": RGBA2RGB,
    "signature_image_average": ImageAverage,
    "signature_image2mask": Image2Mask,
    "signature_mask2image": Mask2Image,
    "signature_base_color": BaseColor,
    "signature_image_subtract": ImageSubtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_rgb2hsv": "SIG RGB2HSV",
    "signature_rgbhls": "SIG RGBHLS",
    "signature_rgba2rgb": "SIG RGBA2RGB",
    "signature_image_average": "SIG Image Average",
    "signature_image2mask": "SIG Image2Mask",
    "signature_mask2image": "SIG Mask2Image",
    "signature_base_color": "SIG Base Color",
    "signature_image_subtract": "SIG Image Subtract",
}
