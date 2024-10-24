import torch
from kornia.geometry import transform
from signature_core.functional.color import color_average
from signature_core.functional.filters import (
    gaussian_blur2d,
    image_soft_light,
    unsharp_mask,
)
from signature_core.functional.transform import resize
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
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
                "width": ("INT", {"default": -1, "min": -1, "max": 48000, "step": 1}),
                "height": ("INT", {"default": -1, "min": -1, "max": 48000, "step": 1}),
                "X": ("INT", {"default": 0, "min": 0, "max": 48000, "step": 1}),
                "Y": ("INT", {"default": 0, "min": 0, "max": 48000, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "rgb",
        "rgba",
    )
    FUNCTION = "execute"

    CATEGORY = IMAGE_CAT

    def execute(self, **kwargs):
        image = kwargs.get("image")
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        image_overlay = kwargs.get("image_overlay")
        if not isinstance(image_overlay, torch.Tensor):
            raise ValueError("Image overlay must be a torch.Tensor")
        width = kwargs.get("width")
        if not isinstance(width, int):
            raise ValueError("Width must be an integer")
        height = kwargs.get("height")
        if not isinstance(height, int):
            raise ValueError("Height must be an integer")
        x = kwargs.get("X")
        if not isinstance(x, int):
            raise ValueError("X must be an integer")
        y = kwargs.get("Y")
        if not isinstance(y, int):
            raise ValueError("Y must be an integer")
        rotation = kwargs.get("rotation")
        if not isinstance(rotation, int):
            raise ValueError("Rotation must be an integer")
        feathering = kwargs.get("feathering")
        if not isinstance(feathering, int):
            raise ValueError("Feathering must be an integer")

        base_image = TensorImage.from_BWHC(image)
        overlay_image = TensorImage.from_BWHC(image_overlay)

        if width == -1:
            width = overlay_image.shape[3]
        if height == -1:
            height = overlay_image.shape[2]

        device = base_image.device
        overlay_image = overlay_image.to(device)

        # Resize overlay image
        overlay_image = transform.resize(overlay_image, (height, width))

        if rotation != 0:
            angle = torch.tensor(rotation, dtype=torch.float32, device=device)
            center = torch.tensor([width / 2, height / 2], dtype=torch.float32, device=device)
            overlay_image = transform.rotate(overlay_image, angle, center=center)

        # Create mask (handle both RGB and RGBA cases)
        if overlay_image.shape[1] == 4:
            mask = overlay_image[:, 3:4, :, :]
        else:
            mask = torch.ones((1, 1, height, width), device=device)

        # Pad overlay image and mask
        pad_left = x
        pad_top = y
        pad_right = max(0, base_image.shape[3] - overlay_image.shape[3] - x)
        pad_bottom = max(0, base_image.shape[2] - overlay_image.shape[2] - y)

        overlay_image = torch.nn.functional.pad(overlay_image, (pad_left, pad_right, pad_top, pad_bottom))
        mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom))

        # Resize to match base image
        overlay_image = transform.resize(overlay_image, base_image.shape[2:])
        mask = transform.resize(mask, base_image.shape[2:])

        if feathering > 0:
            kernel_size = 2 * feathering + 1
            feather_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size**2)
            mask = torch.nn.functional.conv2d(mask, feather_kernel, padding=feathering)

        # Blend images
        result = base_image * (1 - mask) + overlay_image[:, :3, :, :] * mask

        result = TensorImage(result).get_BWHC()

        rgb = result
        rgba = torch.cat([rgb, mask.permute(0, 2, 3, 1)], dim=3)

        return (rgb, rgba)


class ImageList2Batch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["STRETCH", "FIT", "FILL", "ASPECT"],),
                "interpolation": (["bilinear", "nearest", "bicubic", "area"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT
    INPUT_IS_LIST = True

    def execute(self, **kwargs):
        images = kwargs.get("images")
        mode = kwargs.get("mode") or "FIT"
        interpolation = kwargs.get("interpolation") or "bilinear"
        if not isinstance(images, list):
            raise ValueError("Images must be a list")
        if isinstance(mode, list) and len(mode) == 1:
            mode = mode[0]
        if isinstance(interpolation, list) and len(interpolation) == 1:
            interpolation = interpolation[0]

        if not isinstance(mode, str):
            raise ValueError("Mode must be a string")
        if not isinstance(interpolation, str):
            raise ValueError("Interpolation must be a string")

        # Check if all images have the same shape
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            # All images have the same shape, no need to resize
            return (torch.stack(images),)

        # Images have different shapes, proceed with resizing
        max_height = max(img.shape[1] for img in images)
        max_width = max(img.shape[2] for img in images)

        resized_images = []
        for img in images:
            tensor_img = TensorImage.from_BWHC(img)
            resized_img = resize(tensor_img, max_width, max_height, mode=mode, interpolation=interpolation)
            resized_images.append(resized_img.get_BWHC().squeeze(0))

        return (torch.stack(resized_images),)


class ImageBatch2List:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = IMAGE_CAT
    OUTPUT_IS_LIST = (True,)

    def execute(self, **kwargs):
        image = kwargs.get("image")
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")

        image_list = [img.unsqueeze(0) for img in image]
        return (image_list,)
