import torch
from signature_core.functional.augmentation import (
    compose_augmentation,
    flip_augmentation,
    random_crop_augmentation,
)
from signature_core.img.tensor_image import TensorImage

from .categories import AUGMENTATION_CAT


class RandomCropAugmentation:
    """Applies a random crop augmentation to an image.

    This class performs a random crop on an image based on specified dimensions and percentage.

    Methods:
        execute(**kwargs): Applies the random crop augmentation and returns the augmented image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        min_window (int): The minimum window size for cropping.
        max_window (int): The maximum window size for cropping.
        percent (float): The percentage of the image to crop.
        augmentation: An optional existing augmentation to apply.

    Returns:
        tuple: The augmented image.
    """

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
    FUNCTION = "execute"
    CATEGORY = AUGMENTATION_CAT

    def execute(
        self,
        **kwargs,
    ):
        height = kwargs.get("height") or 1024
        width = kwargs.get("width") or 1024
        min_window = kwargs.get("min_window") or 256
        max_window = kwargs.get("max_window") or 1024
        percent = kwargs.get("percent") or 1.0
        augmentation = kwargs.get("augmentation")
        augmentation = random_crop_augmentation(height, width, min_window, max_window, percent, augmentation)
        return (augmentation,)


class FlipAugmentation:
    """Applies a flip augmentation to an image.

    This class performs a horizontal or vertical flip on an image based on the specified direction and percentage.

    Methods:
        execute(**kwargs): Applies the flip augmentation and returns the augmented image.

    Args:
        flip (str): The direction of the flip ('horizontal' or 'vertical').
        percent (float): The percentage of the image to flip.
        augmentation: An optional existing augmentation to apply.

    Returns:
        tuple: The augmented image.
    """

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
    FUNCTION = "execute"
    CATEGORY = AUGMENTATION_CAT

    def execute(self, **kwargs):
        flip = kwargs.get("flip") or "horizontal"
        percent = kwargs.get("percent") or 0.5
        augmentation = kwargs.get("augmentation")
        augmentation = flip_augmentation(flip, percent, augmentation)
        return (augmentation,)


class ComposeAugmentation:
    """Composes multiple augmentations and applies them to an image and mask.

    This class combines multiple augmentations and applies them to an image and mask,
    supporting multiple samples and random seeds.

    Methods:
        execute(**kwargs): Applies the composed augmentations and returns the augmented images and masks.

    Args:
        augmentation: The augmentation to apply.
        samples (int): The number of samples to generate.
        seed (int): The random seed for augmentation.
        image: The input image to augment.
        mask: The input mask to augment.

    Returns:
        tuple: Lists of augmented images and masks.
    """

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
    FUNCTION = "execute"
    CATEGORY = AUGMENTATION_CAT
    OUTPUT_IS_LIST = (
        True,
        True,
    )

    def execute(
        self,
        **kwargs,
    ):
        augmentation = kwargs.get("augmentation")
        samples = kwargs.get("samples") or 1
        image = kwargs.get("image")
        mask = kwargs.get("mask")
        seed = kwargs.get("seed") or -1

        image_tensor = TensorImage.from_BWHC(image) if isinstance(image, torch.Tensor) else None
        mask_tensor = TensorImage.from_BWHC(mask) if isinstance(mask, torch.Tensor) else None

        total_images, total_masks = compose_augmentation(
            augmentation=augmentation,
            samples=samples,
            image_tensor=image_tensor,
            mask_tensor=mask_tensor,
            seed=seed,
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
