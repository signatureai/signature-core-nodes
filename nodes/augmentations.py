import torch
from signature_core.functional.augmentation import (
    compose_augmentation,
    flip_augmentation,
    random_crop_augmentation,
)
from signature_core.img.tensor_image import TensorImage

from .categories import AUGMENTATION_CAT


class RandomCropAugmentation:
    """Applies random crop augmentation to images with configurable dimensions and frequency.

    This node performs random cropping operations on input images. It allows precise control over the
    crop dimensions through minimum and maximum window sizes, target dimensions, and application
    probability.

    Args:
        height (int): Target height for the crop operation. Must be at least 32 and a multiple of 32.
        width (int): Target width for the crop operation. Must be at least 32 and a multiple of 32.
        min_window (int): Minimum size of the crop window. Must be a multiple of 32.
        max_window (int): Maximum size of the crop window. Must be a multiple of 32.
        percent (float): Probability of applying the crop, from 0.0 to 1.0.
        augmentation (AUGMENTATION, optional): Existing augmentation to chain with. Defaults to None.

    Returns:
        tuple: Contains a single element:
            augmentation (AUGMENTATION): The configured crop augmentation operation.

    Raises:
        ValueError: If any dimension parameters are not multiples of 32.
        ValueError: If min_window is larger than max_window.
        ValueError: If percent is not between 0.0 and 1.0.

    Notes:
        - Window size is randomly selected between min_window and max_window for each operation
        - Can be chained with other augmentations through the augmentation parameter
        - All dimension parameters must be multiples of 32 for proper operation
        - Setting percent to 1.0 ensures the crop is always applied
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
    """Applies horizontal or vertical flip augmentation to images with configurable probability.

    This node performs random flip transformations on input images. It supports both horizontal and
    vertical flip operations with adjustable probability of application.

    Args:
        flip (str): Direction of flip operation, either:
            - "horizontal": Flips image left to right
            - "vertical": Flips image top to bottom
        percent (float): Probability of applying the flip, from 0.0 to 1.0.
        augmentation (AUGMENTATION, optional): Existing augmentation to chain with. Defaults to None.

    Returns:
        tuple: Contains a single element:
            augmentation (AUGMENTATION): The configured flip augmentation operation.

    Raises:
        ValueError: If flip direction is not "horizontal" or "vertical".
        ValueError: If percent is not between 0.0 and 1.0.

    Notes:
        - Flip direction cannot be changed after initialization
        - Setting percent to 0.5 applies the flip to approximately half of all samples
        - Can be chained with other augmentations through the augmentation parameter
        - Transformations are applied consistently when used with ComposeAugmentation
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
    """Combines and applies multiple augmentation operations with consistent random transformations.

    This node orchestrates the application of multiple augmentation operations to images and masks. It
    provides control over sample generation and reproducibility through seed management.

    Args:
        augmentation (AUGMENTATION): The augmentation operation or chain to apply.
        samples (int): Number of augmented versions to generate. Must be >= 1.
        seed (int): Random seed for reproducible results. Use -1 for random seeding.
            Valid range: -1 to 10000000000000000.
        image (IMAGE, optional): Input image to augment. Defaults to None.
        mask (MASK, optional): Input mask to augment. Defaults to None.

    Returns:
        tuple: Contains two elements:
            images (List[IMAGE]): List of augmented versions of the input image.
            masks (List[MASK]): List of augmented versions of the input mask.

    Raises:
        ValueError: If neither image nor mask is provided.
        ValueError: If samples is less than 1.
        ValueError: If seed is outside valid range.

    Notes:
        - At least one of image or mask must be provided
        - All augmentations are applied consistently to both image and mask
        - Output is always returned as lists, even when samples=1
        - Using a fixed seed ensures reproducible augmentations
        - Supports chaining multiple augmentations through the augmentation parameter
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
