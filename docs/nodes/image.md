# Image Nodes

## ImageBaseColor

Creates a solid color image of specified dimensions.

Parameters: hex_color (str): Hex color code (e.g., "#FFFFFF") width (int): Width of the
output image height (int): Height of the output image

Returns: tuple[torch.Tensor]: Single-element tuple containing a BWHC format tensor

### Return Types

- `IMAGE`

::: nodes.image.ImageBaseColor

## ImageGaussianBlur

Applies Gaussian blur to an input image.

Parameters: image (torch.Tensor): Input image in BWHC format radius (int): Blur radius
(default: 13) sigma (float): Blur sigma/strength (default: 10.5) interations (int):
Number of blur passes (default: 1)

Returns: tuple[torch.Tensor]: Single-element tuple containing the blurred image

### Return Types

- `IMAGE`

::: nodes.image.ImageGaussianBlur

## ImageUnsharpMask

Applies unsharp mask filter to sharpen an image.

Parameters: image (torch.Tensor): Input image in BWHC format radius (int): Sharpening
radius (default: 3) sigma (float): Sharpening strength (default: 1.5) interations (int):
Number of sharpening passes (default: 1)

Returns: tuple[torch.Tensor]: Single-element tuple containing the sharpened image

### Return Types

- `IMAGE`

::: nodes.image.ImageUnsharpMask

## ImageSoftLight

Applies soft light blend mode between two images.

Parameters: top (torch.Tensor): Top layer image in BWHC format bottom (torch.Tensor):
Bottom layer image in BWHC format

Returns: tuple[torch.Tensor]: Single-element tuple containing the blended image

### Return Types

- `IMAGE`

::: nodes.image.ImageSoftLight

## ImageAverage

Calculates the color average of an input image.

Parameters: image (torch.Tensor): Input image in BWHC format

Returns: tuple[torch.Tensor]: Single-element tuple containing the averaged image

### Return Types

- `IMAGE`

::: nodes.image.ImageAverage

## ImageSubtract

Subtracts one image from another (absolute difference).

Parameters: image_0 (torch.Tensor): First image in BWHC format image_1 (torch.Tensor):
Second image in BWHC format

Returns: tuple[torch.Tensor]: Single-element tuple containing the difference image

### Return Types

- `IMAGE`

::: nodes.image.ImageSubtract

## ImageTranspose

Transforms an overlay image onto a base image with various adjustments.

Parameters: image (torch.Tensor): Base image in BWHC format image_overlay
(torch.Tensor): Overlay image in BWHC format width (int): Target width for overlay (-1
for original size) height (int): Target height for overlay (-1 for original size) X
(int): Horizontal offset Y (int): Vertical offset rotation (int): Rotation angle in
degrees feathering (int): Edge feathering amount

Returns: tuple[torch.Tensor, torch.Tensor]: RGB and RGBA versions of the composited image

### Return Types

- `IMAGE`
- `IMAGE`

::: nodes.image.ImageTranspose

## ImageList2Batch

Converts a list of images into a batched tensor, handling different image sizes.

Parameters: images (list[torch.Tensor]): List of input images mode (str): Resize mode
('STRETCH', 'FIT', 'FILL', 'ASPECT') interpolation (str): Interpolation method for
resizing

Returns: tuple[torch.Tensor]: Single-element tuple containing the batched images

### Return Types

- `IMAGE`

::: nodes.image.ImageList2Batch

## ImageBatch2List

Converts a batched tensor of images into a list of individual images.

Parameters: image (torch.Tensor): Batched input images in BWHC format

Returns: tuple[list[torch.Tensor]]: Single-element tuple containing list of images

### Return Types

- `IMAGE`

::: nodes.image.ImageBatch2List

## GetImageShape

Returns the dimensions of an input image.

Parameters: image (torch.Tensor): Input image in BWHC format

Returns: tuple[int, int, int, int, str]: Batch size, width, height, channels, and shape
string

### Return Types

- `INT`
- `INT`
- `INT`
- `INT`
- `STRING`

::: nodes.image.GetImageShape
