import time

import torch
from signature_core.functional.color import (
    grayscale_to_rgb,
    rgb_to_grayscale,
    rgb_to_hls,
    rgb_to_hsv,
    rgba_to_rgb,
)
from signature_core.img.tensor_image import TensorImage

from .categories import UTILS_CAT
from .shared import any_type, clean_memory


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
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "any_string"

    def execute(self, value):
        return (str(value),)


class Any2Int:
    """Converts any input value to its int representation.

    This utility node provides a simple way to convert any input value into a int format using
    Python's built-in int() function. Useful for debugging, logging, or text-based operations.

    Args:
        value (Any): The input value to be converted to a int.

    Returns:
        tuple[int]: A single-element tuple containing the int representation of the input value.

    Notes:
        - The conversion is done using Python's native int() function
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, value):
        return (int(value),)


class Any2Float:
    """Converts any input value to its float representation.

    This utility node provides a simple way to convert any input value into a float format using
    Python's built-in float() function. Useful for debugging, logging, or text-based operations.

    Args:
        value (Any): The input value to be converted to a float.

    Returns:
        tuple[float]: A single-element tuple containing the float representation of the input value.

    Notes:
        - The conversion is done using Python's native float() function
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, value):
        return (float(value),)


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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "any2any"

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
    CLASS_ID = "rgba2rgb"

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        if image_tensor.shape[1] == 4:
            image_tensor = rgba_to_rgb(image_tensor)
        output = image_tensor.get_BWHC()
        return (output,)


class RGB2GRAY:
    """Converts RGB images to grayscale format.

    This node transforms RGB color images to single-channel grayscale images using
    standard luminance conversion factors.

    Args:
        image (torch.Tensor): Input RGB image in BWHC format

    Returns:
        tuple[torch.Tensor]: Single-element tuple containing grayscale image in BWHC format

    Notes:
        - Uses standard RGB to grayscale conversion weights
        - Output is single-channel
        - Preserves image dimensions
        - Values remain in [0,1] range
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

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = rgb_to_grayscale(image_tensor).get_BWHC()
        return (output,)


class GRAY2RGB:
    """Converts grayscale images to RGB format.

    This node transforms single-channel grayscale images to three-channel RGB images
    by replicating the grayscale values across channels.

    Args:
        image (torch.Tensor): Input grayscale image in BWHC format

    Returns:
        tuple[torch.Tensor]: Single-element tuple containing RGB image in BWHC format

    Notes:
        - Replicates grayscale values to all RGB channels
        - Output has three identical channels
        - Preserves image dimensions
        - Values remain in [0,1] range
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

    def execute(self, image: torch.Tensor):
        image_tensor = TensorImage.from_BWHC(image)
        output = grayscale_to_rgb(image_tensor).get_BWHC()
        return (output,)


class PurgeVRAM:
    """Cleans up VRAM by forcing memory deallocation and cache clearing.

    A utility node that performs comprehensive VRAM cleanup by collecting garbage, emptying CUDA cache,
    and unloading models. Useful for managing memory usage in complex workflows.

    Args:
        anything (Any): Any input value that will be passed through unchanged.

    Returns:
        tuple[Any]: A single-element tuple containing the unchanged input value.

    Notes:
        - Calls Python's garbage collector
        - Clears CUDA cache if available
        - Unloads and cleans up ComfyUI models
        - Performs soft cache emptying
        - Input value is passed through unchanged
        - Useful for preventing out-of-memory errors
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "anything": (any_type, {}),
            },
            "optional": {},
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    OUTPUT_NODE = True
    # DEPRECATED = True
    CLASS_ID = "purge_vram"

    def execute(self, anything):
        clean_memory()
        return (anything,)


class WaitSeconds:
    """Pauses execution for a specified number of seconds.

    A utility node that introduces a delay in the workflow by sleeping for a given duration. This can
    be useful for timing control, pacing operations, or waiting for external processes to complete.

    Args:
        value (Any): Any input value to be returned after the wait period.
        seconds (float): The duration to wait in seconds. Defaults to 1.0 seconds.

    Returns:
        tuple[Any]: A single-element tuple containing the unchanged input value after the wait.

    Notes:
        - The wait time can be adjusted by changing the `seconds` argument.
        - The function uses Python's time.sleep() to implement the delay.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (any_type,),
                "seconds": (
                    "FLOAT",
                    {
                        "default": 1.0,
                    },
                ),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT

    def execute(self, **kwargs):
        value = kwargs.get("value")
        seconds = kwargs.get("seconds") or 1.0
        time.sleep(seconds)
        return (value,)


class ListBuilder:

    @classmethod
    def INPUT_TYPES(cls):

        inputs = {
            "required": {
                "num_slots": ([str(i) for i in range(1, 11)], {"default": "1"}),
            },
            "optional": {},
        }

        for i in range(1, 11):
            inputs["optional"].update(
                {
                    f"value_{i}": (any_type, {"forceInput": True}),
                }
            )
        return inputs

    RETURN_TYPES = (
        any_type,
        "LIST",
    )
    RETURN_NAMES = (
        "ANY",
        "LIST",
    )
    FUNCTION = "execute"
    CATEGORY = UTILS_CAT
    CLASS_ID = "list_builder"
    OUTPUT_IS_LIST = (
        True,
        False,
    )

    def execute(self, **kwargs):
        num_slots = int(kwargs.get("num_slots", 1))
        list_stack = []
        for i in range(1, num_slots + 1):
            list_value = kwargs.get(f"value_{i}")
            if list_value is not None:
                list_stack.append(list_value)
        return (
            list_stack,
            list_stack,
        )
