import torch
from signature_core.img.tensor_image import TensorImage
from signature_core.functional.color import rgb_to_hsv, rgb_to_hls, color_average, rgba_to_rgb
from .categories import COLOR_CAT

class RGB2HSV:

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
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
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
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
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgba_to_rgb(image_tensor).get_BWHC()
        return (output,)

class ImageAverage:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        step = TensorImage.from_BWHC(image)
        output = color_average(step).get_BWHC()
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Signature RGB2HSV": RGB2HSV,
    "Signature RGBHLS": RGBHLS,
    "Signature RGBA2RGB": RGBA2RGB,
    "Signature Image Average": ImageAverage,
}