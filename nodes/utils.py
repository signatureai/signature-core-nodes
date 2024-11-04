import gc

import comfy.model_management as mm  # type: ignore
import torch
from signature_core.functional.color import rgb_to_hls, rgb_to_hsv, rgba_to_rgb
from signature_core.img.tensor_image import TensorImage

from .categories import UTILS_CAT
from .shared import any_type


class Any2String:
    """Converts any input value to its string representation.

    This utility node provides a simple way to convert any input value into a string format using
    Python's built-in str() function. Useful for debugging, logging, or text-based operations.

    Args:
        value (Any): The input value to be converted to a string.

    Returns:
        tuple[str]: A single-element tuple containing the string representation of the input value.

    Notes:
        - The conversion is done using Python's native str() function
        - All Python types are supported as they all implement __str__
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "any_string"

    def execute(self, value):
        return (str(value),)


class Any2Image:
    """Converts any inputs value to image format.

    A utility node that handles conversion of tensor inputs to a compatible image format for use in
    image processing workflows.

    Args:
        value (Any): The input value to be converted to image format.

    Returns:
        tuple[torch.Tensor]: A single-element tuple containing the image tensor.

    Raises:
        ValueError: If the input value is not a torch.Tensor or cannot be converted to image format.

    Notes:
        - Currently only supports torch.Tensor inputs
        - Input tensors should be in a format compatible with image processing (BWHC format)
        - Future versions may support additional input types and automatic conversion
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "any_image"

    def execute(self, value):
        if isinstance(value, torch.Tensor):
            return (value,)
        raise ValueError(f"Unsupported type: {type(value)}")


class Any2Any:
    """Passes through any input value unchanged.

    A utility node that acts as a pass-through or identity function, returning the input value
    without any modifications. Useful for workflow organization or debugging.

    Args:
        value (Any): Any input value to be passed through.

    Returns:
        tuple[Any]: A single-element tuple containing the unchanged input value.

    Notes:
        - No validation or transformation is performed on the input
        - Useful as a placeholder or for debugging workflow connections
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "any_any"

    def execute(self, value):
        return (value,)


class RGB2HSV:
    """Converts RGB images to HSV color space.

    Transforms images from RGB (Red, Green, Blue) color space to HSV (Hue, Saturation, Value)
    color space while preserving the image structure and dimensions.

    Args:
        image (torch.Tensor): Input RGB image tensor in BWHC format.

    Returns:
        tuple[torch.Tensor]: A single-element tuple containing the HSV image tensor in BWHC format.

    Notes:
        - Input must be in BWHC (Batch, Width, Height, Channels) format
        - RGB values should be normalized to [0, 1] range
        - Output HSV values are in ranges: H[0,360], S[0,1], V[0,1]
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "rgb_hsv"

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hsv(image_tensor).get_BWHC()
        return (output,)


class RGB2HLS:
    """Converts RGB images to HLS color space.

    Transforms images from RGB (Red, Green, Blue) color space to HLS (Hue, Lightness, Saturation)
    color space while preserving the image structure and dimensions.

    Args:
        image (torch.Tensor): Input RGB image tensor in BWHC format.

    Returns:
        tuple[torch.Tensor]: A single-element tuple containing the HLS image tensor in BWHC format.

    Notes:
        - Input must be in BWHC (Batch, Width, Height, Channels) format
        - RGB values should be normalized to [0, 1] range
        - Output HLS values are in ranges: H[0,360], L[0,1], S[0,1]
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "rgb_hls"

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_hls(image_tensor).get_BWHC()
        return (output,)


class RGBA2RGB:
    """Converts RGBA images to RGB format.

    Transforms images from RGBA (Red, Green, Blue, Alpha) format to RGB format by removing the
    alpha channel. Passes through RGB images unchanged.

    Args:
        image (torch.Tensor): Input image tensor in BWHC format (either RGBA or RGB).

    Returns:
        tuple[torch.Tensor]: A single-element tuple containing the RGB image tensor in BWHC format.

    Notes:
        - Input must be in BWHC (Batch, Width, Height, Channels) format
        - Handles both 3-channel (RGB) and 4-channel (RGBA) inputs
        - RGB images are passed through unchanged
        - Alpha channel is removed from RGBA images using rgba_to_rgb conversion
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "rgba_rgb"

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        if image_tensor.shape[1] == 4:
            image_tensor = rgba_to_rgb(image_tensor)
        output = image_tensor.get_BWHC()
        return (output,)


class PurgeVRAM:
    """Cleans up VRAM by purging caches and/or unloading models.

    A comprehensive VRAM management utility that can clear various types of memory caches and
    optionally unload models to free up graphics memory.

    Args:
        anything (Any): Any input value (unused, allows connection in workflow).
        purge_cache (bool): Whether to purge system and CUDA cache. Defaults to True.
        purge_models (bool): Whether to unload all models from memory. Defaults to True.

    Returns:
        tuple[None]: An empty tuple signifying completion.

    Notes:
        - Performs the following cleanup operations when purge_cache is True:
            * Clears Python garbage collector
            * Empties PyTorch CUDA cache
            * Runs CUDA IPC collection
        - When purge_models is True:
            * Unloads all loaded models
            * Performs soft cache emptying
        - Useful for managing memory in complex workflows or between heavy operations
        - Can be used as an OUTPUT_NODE in the workflow
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "anything": (any_type, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    OUTPUT_NODE = True

    def execute(self, anything, purge_cache, purge_models):

        if purge_cache:

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        if purge_models:
            mm.unload_all_models()
            mm.soft_empty_cache(True)
        return (None,)
