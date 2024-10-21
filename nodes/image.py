import torch
from kornia.geometry import transform
from signature_core.functional.color import color_average
from signature_core.functional.filters import (
    gaussian_blur2d,
    image_soft_light,
    unsharp_mask,
)
from signature_core.img.tensor_image import TensorImage

from .categories import IMAGE_CAT


class ImageBaseColor:
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
        if not isinstance(width, int):
            raise ValueError("Width must be an integer")
        if not isinstance(height, int):
            raise ValueError("Height must be an integer")
        if not isinstance(hex_color, str):
            raise ValueError("Hex color must be a string")
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
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        radius = kwargs.get("radius")
        if not isinstance(radius, int):
            raise ValueError("Radius must be an integer")
        sigma = kwargs.get("sigma")
        if not isinstance(sigma, float):
            raise ValueError("Sigma must be a float")
        interations = kwargs.get("interations")
        if not isinstance(interations, int):
            raise ValueError("Interations must be an integer")
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
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        radius = kwargs.get("radius")
        if not isinstance(radius, int):
            raise ValueError("Radius must be an integer")
        sigma = kwargs.get("sigma")
        if not isinstance(sigma, float):
            raise ValueError("Sigma must be a float")
        interations = kwargs.get("interations")
        if not isinstance(interations, int):
            raise ValueError("Interations must be an integer")
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
        if not isinstance(top, torch.Tensor):
            raise ValueError("Top must be a torch.Tensor")
        if not isinstance(bottom, torch.Tensor):
            raise ValueError("Bottom must be a torch.Tensor")
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
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
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
        if not isinstance(image_0, torch.Tensor):
            raise ValueError("Image 0 must be a torch.Tensor")
        if not isinstance(image_1, torch.Tensor):
            raise ValueError("Image 1 must be a torch.Tensor")
        image_0_tensor = TensorImage.from_BWHC(image_0)
        image_1_tensor = TensorImage.from_BWHC(image_1)
        image_tensor = torch.abs(image_0_tensor - image_1_tensor)
        output = TensorImage(image_tensor).get_BWHC()
        return (output,)


class ImageTranspose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_overlay": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 48000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 48000, "step": 1}),
                "X": ("INT", {"default": 0, "min": 0, "max": 48000, "step": 1}),
                "Y": ("INT", {"default": 0, "min": 0, "max": 48000, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"

    CATEGORY = IMAGE_CAT

    def process(self, **kwargs):
        image_element = kwargs.get("image")
        image_bg = kwargs.get("image_overlay")

        if not isinstance(image_element, torch.Tensor):
            raise ValueError("Image element must be a torch.Tensor")
        if not isinstance(image_bg, torch.Tensor):
            raise ValueError("Image background must be a torch.Tensor")

        image_element = TensorImage.from_BWHC(image_element)
        image_bg = TensorImage.from_BWHC(image_bg)

        width = kwargs.get("width")
        if not isinstance(width, int):
            raise ValueError("Width must be an integer")
        height = kwargs.get("height")
        if not isinstance(height, int):
            raise ValueError("Height must be an integer")
        X = kwargs.get("X")
        if not isinstance(X, int):
            raise ValueError("X must be an integer")
        Y = kwargs.get("Y")
        if not isinstance(Y, int):
            raise ValueError("Y must be an integer")
        rotate = kwargs.get("rotation")
        if not isinstance(rotate, int):
            raise ValueError("Rotation must be an integer")
        size = (width, height)
        loc = (X, Y)

        image_element = transform.rotate(image_element, torch.tensor([rotate]).float())
        image_element = transform.resize(image_element, size)

        elem_h, elem_w = image_element.shape[2], image_element.shape[3]

        new_image = torch.zeros_like(image_element)[0]
        print(image_element.shape)
        print(image_bg.shape)
        new_image[:, loc[1] : loc[1] + elem_h, loc[0] : loc[0] + elem_w] = image_element

        alpha_element = image_element[:, 3:4, :, :]
        alpha_bg = 1.0 - alpha_element

        result = alpha_bg * image_bg + alpha_element * new_image
        result = TensorImage(result).get_BWHC()

        return (result,)


NODE_CLASS_MAPPINGS = {
    "signature_image_base_color": ImageBaseColor,
    "signature_image_gaussian_blur": ImageGaussianBlur,
    "signature_image_unsharp_mask": ImageUnsharpMask,
    "signature_image_soft_light": ImageSoftLight,
    "signature_image_average": ImageAverage,
    "signature_image_subtract": ImageSubtract,
    # "signature_image_transpose": ImageTranspose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_image_base_color": "SIG ImageBaseColor",
    "signature_image_gaussian_blur": "SIG ImageGaussianBlur",
    "signature_image_unsharp_mask": "SIG ImageUnsharpMask",
    "signature_image_soft_light": "SIG ImageSoftLight",
    "signature_image_average": "SIG ImageAverage",
    "signature_image_subtract": "SIG ImageSubtract",
    # "signature_image_transpose": "SIG ImageTranspose",
}
