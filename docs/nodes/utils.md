# Utils Nodes

## Any2String

Converts any input value to its string representation.

A utility node that takes any input value and converts it to a string using Python's
str() function.

### Return Types

- `STRING`

::: nodes.utils.Any2String

## Any2Image

Converts any inputs value to image format.

A utility node that converts tensor inputs to image format. Currently only supports
torch.Tensor inputs.

Raises: ValueError: If the input value is not a torch.Tensor.

### Return Types

- `IMAGE`

::: nodes.utils.Any2Image

## Any2Any

Passes through any input value unchanged.

A utility node that acts as a pass-through, returning the input value without
modification.

::: nodes.utils.Any2Any

## RGB2HSV

Converts RGB images to HSV color space.

A utility node that converts RGB format images to HSV (Hue, Saturation, Value) color
space. Expects input images in BWHC format.

### Return Types

- `IMAGE`

::: nodes.utils.RGB2HSV

## RGBHLS

Converts RGB images to HLS color space.

A utility node that converts RGB format images to HLS (Hue, Lightness, Saturation) color
space. Expects input images in BWHC format.

### Return Types

- `IMAGE`

::: nodes.utils.RGBHLS

## RGBA2RGB

Converts RGBA images to RGB format.

A utility node that converts RGBA (Red, Green, Blue, Alpha) images to RGB format. If the
input image is already in RGB format, it will be passed through unchanged. Expects input
images in BWHC format.

### Return Types

- `IMAGE`

::: nodes.utils.RGBA2RGB

## PurgeVRAM

Cleans up VRAM by purging caches and/or unloading models.

A utility node that helps manage VRAM usage by:

- Clearing Python garbage collector
- Emptying PyTorch CUDA cache
- Optionally unloading all models
- Optionally purging system cache

Args: anything: Any input value (unused, allows connection in workflow) purge_cache
(bool): Whether to purge system and CUDA cache purge_models (bool): Whether to unload
all models from memory

::: nodes.utils.PurgeVRAM
