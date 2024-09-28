import torch
from signature_core.functional.augmentation import (
    compose_augmentation,
    flip_augmentation,
    random_crop_augmentation,
)
from signature_core.img.tensor_image import TensorImage

from ..categories import AUGMENTATION_CAT


class RandomCropAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("INT", {"default": 1024, "min": 32, "step": 32}),
                "width": ("INT", {"default": 1024, "min": 32, "step": 32}),
                "min_window": ("INT", {"default": 256, "step": 32}),
                "max_window": ("INT", {"default": 1024, "step": 32}),
                "percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "augmentation": ("AUGMENTATION", {"default": None}),
            },
        }

    RETURN_TYPES = ("AUGMENTATION",)
    RETURN_NAMES = ("augmentation",)
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT

    def process(
        self,
        height: int,
        width: int,
        min_window: int,
        max_window: int,
        percent: float,
        augmentation: list | None = None,
    ):
        augmentation = random_crop_augmentation(height, width, min_window, max_window, percent, augmentation)
        return (augmentation,)


class FlipAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flip": (["horizontal", "vertical"], {"default": "horizontal"}),
                "percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "augmentation": ("AUGMENTATION", {"default": None}),
            },
        }

    RETURN_TYPES = ("AUGMENTATION",)
    RETURN_NAMES = ("augmentation",)
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT

    def process(self, flip: str, percent: float, augmentation: list | None = None):
        augmentation = flip_augmentation(flip, percent, augmentation)
        return (augmentation,)


class ComposeAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "augmentation": ("AUGMENTATION",),
                "samples": ("INT", {"default": 1, "min": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 10000000000000000}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT
    OUTPUT_IS_LIST = (True,)

    def process(
        self,
        augmentation,
        samples: int,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        seed: int = -1,
    ):
        image_tensor = TensorImage.from_BWHC(image) if image is not None else None
        mask_tensor = TensorImage.from_BWHC(mask) if mask is not None else None

        total_images, total_masks = compose_augmentation(
            augmentation, samples, image_tensor, mask_tensor, seed
        )
        if total_images is None:
            total_images = []
        if total_masks is None:
            total_masks = []
        node_image = [image.get_BWHC() for image in total_images]
        node_mask = [mask.get_BWHC() for mask in total_masks]
        return (
            node_image,
            node_mask,
        )


NODE_CLASS_MAPPINGS = {
    "signature_compose_augmentation": ComposeAugmentation,
    "signature_random_crop_augmentation": RandomCropAugmentation,
    "signature_flip_augmentation": FlipAugmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_compose_augmentation": "SIG Compose Augmentation",
    "signature_random_crop_augmentation": "SIG Random Crop Augmentation",
    "signature_flip_augmentation": "SIG Flip Augmentation",
}
