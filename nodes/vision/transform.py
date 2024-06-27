import torch
from ..categories import TRANSFORM_CAT
from signature_core.img.tensor_image import TensorImage
from signature_core.functional.transform import rescale, resize, rotate, auto_crop, cutout


class AutoCrop:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "padding": ("INT", {"default": 0}),
            }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "x", "y", "width", "height")

    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self,
                image: torch.Tensor,
                mask: torch.Tensor,
                padding: int):

        img_tensor = TensorImage.from_BWHC(image)
        mask_tensor = TensorImage.from_BWHC(mask)
        img_result, mask_result, min_x, min_y, width, height = auto_crop(img_tensor, mask_tensor, padding=padding)
        output_img = TensorImage(img_result).get_BWHC()
        output_mask = TensorImage(mask_result).get_BWHC()

        return (output_img, output_mask, min_x, min_y, width, height)


class Rescale:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
                "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
                "antialias": ("BOOLEAN", {"default": True}),
                },
            }
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self,
                image: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                factor: float = 2.0,
                interpolation: str = 'nearest',
                antialias: bool = True):

        input_image = TensorImage.from_BWHC(image) if isinstance(image, torch.Tensor) else TensorImage(torch.zeros((1,3, 1, 1)))
        input_mask = TensorImage.from_BWHC(mask) if isinstance(mask, torch.Tensor) else TensorImage(torch.zeros((1,1, 1, 1)))
        output_image = rescale(input_image, factor, interpolation, antialias).get_BWHC()
        output_mask = rescale(input_mask, factor, interpolation, antialias).get_BWHC()


        return (output_image, output_mask,)


class Resize:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": False}),
                "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
                "antialias": ("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self,
                image: torch.Tensor | None = None,
                mask: torch.Tensor | None = None,
                width:int = 512,
                height:int=512,
                keep_aspect_ratio: bool = False,
                interpolation: str = 'nearest',
                antialias: bool = True):

        input_image = TensorImage.from_BWHC(image) if isinstance(image, torch.Tensor) else TensorImage(torch.zeros((1,3, 1, 1)))
        input_mask = TensorImage.from_BWHC(mask) if isinstance(mask, torch.Tensor) else TensorImage(torch.zeros((1,1, 1, 1)))
        output_image = resize(input_image, width, height, keep_aspect_ratio, interpolation, antialias).get_BWHC()
        output_mask = resize(input_mask, width, height, keep_aspect_ratio, interpolation, antialias).get_BWHC()

        return (output_image, output_mask,)

class Rotate:


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0, "max": 360.0, "step": 1.0}),
                "zoom_to_fit": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, angle: float = 0.0, zoom_to_fit: bool = False):
        input_image = TensorImage.from_BWHC(image) if isinstance(image, torch.Tensor) else TensorImage(torch.zeros((1,3, 1, 1)))
        input_mask = TensorImage.from_BWHC(mask) if isinstance(mask, torch.Tensor) else TensorImage(torch.zeros((1,1, 1, 1)))
        output_image = rotate(input_image, angle, zoom_to_fit).get_BWHC()
        output_mask = rotate(input_mask, angle, zoom_to_fit).get_BWHC()

        return (output_image, output_mask,)


class Cutout:


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("rgb", "rgba")
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor):
        tensor_image = TensorImage.from_BWHC(image)
        tensor_mask = TensorImage.from_BWHC(mask)

        image_rgb, image_rgba = cutout(tensor_image, tensor_mask)

        out_image_rgb = TensorImage(image_rgb).get_BWHC()
        out_image_rgba = TensorImage(image_rgba).get_BWHC()

        return (out_image_rgb, out_image_rgba,)

NODE_CLASS_MAPPINGS = {
    "Signature Cutout": Cutout,
    "Signature Rotate": Rotate,
    "Signature Rescale": Rescale,
    "Signature Resize": Resize,
    "Signature Auto Crop": AutoCrop,
}