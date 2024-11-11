import torch
from signature_core.functional.augmentation import (
    blur_augmentation,
    brightness_contrast_augmentation,
    compose_augmentation,
    cutout_augmentation,
    distortion_augmentation,
    flip_augmentation,
    grid_augmentation,
    perspective_augmentation,
    quality_augmentation,
    random_crop_augmentation,
    rotation_augmentation,
    shift_scale_augmentation,
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
        augmentation = kwargs.get("augmentation") or []
        samples = kwargs.get("samples") or 1
        image = kwargs.get("image")
        mask = kwargs.get("mask")
        seed = kwargs.get("seed") or -1

        # Create a dummy image if only mask is provided
        if image is None and mask is not None:
            image = torch.zeros_like(mask)

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


class BrightnessContrastAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brightness_limit": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "contrast_limit": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
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
        brightness_limit = kwargs.get("brightness_limit") or 0.2
        contrast_limit = kwargs.get("contrast_limit") or 0.2
        percent = kwargs.get("percent") or 0.5
        augmentation = kwargs.get("augmentation")
        augmentation = brightness_contrast_augmentation(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            percent=percent,
            augmentation=augmentation,
        )
        return (augmentation,)


class RotationAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "limit": ("INT", {"default": 45, "min": 0, "max": 180}),
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
        limit = kwargs.get("limit") or 45
        percent = kwargs.get("percent") or 0.5
        augmentation = kwargs.get("augmentation")
        augmentation = rotation_augmentation(limit=limit, percent=percent, augmentation=augmentation)
        return (augmentation,)


class BlurAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blur_type": (["gaussian", "motion", "median"], {"default": "gaussian"}),
                "blur_limit_min": ("INT", {"default": 3, "min": 3, "step": 3}),
                "blur_limit_max": ("INT", {"default": 87, "min": 3, "step": 3}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        blur_type = kwargs.get("blur_type") or "gaussian"
        blur_limit = (kwargs.get("blur_limit_min", 3), kwargs.get("blur_limit_max", 7))
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = blur_augmentation(
            blur_type=blur_type, blur_limit=blur_limit, percent=percent, augmentation=augmentation
        )
        return (augmentation,)


class QualityAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quality_type": (["compression", "downscale"], {"default": "compression"}),
                "quality_limit": ("INT", {"default": 60, "min": 1, "max": 100}),
                "percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
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
        quality_type = kwargs.get("quality_type") or "compression"
        quality_limit = kwargs.get("quality_limit") or 60
        percent = kwargs.get("percent") or 0.2
        augmentation = kwargs.get("augmentation")
        augmentation = quality_augmentation(
            quality_type=quality_type, quality_limit=quality_limit, percent=percent, augmentation=augmentation
        )
        return (augmentation,)


class DistortionAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distortion_type": (["optical", "grid", "elastic"], {"default": "optical"}),
                "severity": ("INT", {"default": 1, "min": 1, "max": 5}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        distortion_type = kwargs.get("distortion_type") or "optical"
        severity = kwargs.get("severity") or 1
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = distortion_augmentation(
            distortion_type=distortion_type, severity=severity, percent=percent, augmentation=augmentation
        )
        return (augmentation,)


class ShiftScaleAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift_limit": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "scale_limit": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "rotate_limit": ("INT", {"default": 45, "min": 0, "max": 180}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        shift_limit = kwargs.get("shift_limit") or 0.1
        scale_limit = kwargs.get("scale_limit") or 0.2
        rotate_limit = kwargs.get("rotate_limit") or 45
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = shift_scale_augmentation(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            percent=percent,
            augmentation=augmentation,
        )
        return (augmentation,)


class CutoutAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_holes": ("INT", {"default": 8, "min": 1, "max": 20}),
                "max_size": ("INT", {"default": 30, "min": 1, "max": 100}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        num_holes = kwargs.get("num_holes") or 8
        max_size = kwargs.get("max_size") or 30
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = cutout_augmentation(
            num_holes=num_holes, max_size=max_size, percent=percent, augmentation=augmentation
        )
        return (augmentation,)


class GridAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grid_type": (["shuffle", "dropout"], {"default": "shuffle"}),
                "grid_size": ("INT", {"default": 3, "min": 2, "max": 10}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        grid_type = kwargs.get("grid_type") or "shuffle"
        grid_size = kwargs.get("grid_size") or 3
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = grid_augmentation(
            grid_type=grid_type, grid_size=grid_size, percent=percent, augmentation=augmentation
        )
        return (augmentation,)


class PerspectiveAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5}),
                "keep_size": ("BOOLEAN", {"default": True}),
                "percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
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
        scale = kwargs.get("scale") or 0.05
        keep_size = kwargs.get("keep_size", True)
        percent = kwargs.get("percent") or 0.3
        augmentation = kwargs.get("augmentation")
        augmentation = perspective_augmentation(
            scale=scale, keep_size=keep_size, percent=percent, augmentation=augmentation
        )
        return (augmentation,)
