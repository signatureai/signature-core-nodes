# Mask Nodes

## BaseMask

Creates a basic binary mask of specified dimensions.

Parameters: color (str): Either "white" or "black" to set mask color width (int): Width
of the output mask (default: 1024) height (int): Height of the output mask
(default: 1024)

Returns: tuple[torch.Tensor]: Single-element tuple containing a binary mask in BWHC
format

### Return Types

- `MASK`

::: nodes.mask.BaseMask

## MaskMorphology

Applies morphological operations to a mask.

Parameters: mask (torch.Tensor): Input mask in BWHC format operation (str): One of:
"dilation", "erosion", "opening", "closing", "gradient", "top_hat", "bottom_hat"
kernel_size (int): Size of the morphological kernel (default: 1) iterations (int):
Number of times to apply the operation (default: 5)

Returns: tuple[torch.Tensor]: Single-element tuple containing the processed mask

### Return Types

- `MASK`

::: nodes.mask.MaskMorphology

## MaskBitwise

Performs bitwise operations between two masks.

Parameters: mask_1 (torch.Tensor): First input mask in BWHC format mask_2
(torch.Tensor): Second input mask in BWHC format mode (str): One of: "and", "or", "xor",
"left_shift", "right_shift"

Returns: tuple[torch.Tensor]: Single-element tuple containing the resulting mask

### Return Types

- `MASK`

::: nodes.mask.MaskBitwise

## MaskDistance

Calculates the Euclidean distance between two masks.

Parameters: mask_0 (torch.Tensor): First input mask in BWHC format mask_1
(torch.Tensor): Second input mask in BWHC format

Returns: tuple[float]: Single-element tuple containing the distance value

### Return Types

- `FLOAT`

::: nodes.mask.MaskDistance

## Mask2Trimap

Converts a mask to a trimap representation (foreground, background, unknown regions).

Parameters: mask (torch.Tensor): Input mask in BWHC format inner_min_threshold (int):
Minimum threshold for inner region (default: 200) inner_max_threshold (int): Maximum
threshold for inner region (default: 255) outer_min_threshold (int): Minimum threshold
for outer region (default: 15) outer_max_threshold (int): Maximum threshold for outer
region (default: 240) kernel_size (int): Size of morphological kernel (default: 10)

Returns: tuple[torch.Tensor, torch.Tensor]: Mask and trimap tensors

### Return Types

- `MASK`
- `TRIMAP`

::: nodes.mask.Mask2Trimap

## MaskBinaryFilter

Applies binary thresholding to a mask.

Parameters: mask (torch.Tensor): Input mask in BWHC format threshold (float): Threshold
value between 0 and 1 (default: 0.01)

Returns: tuple[torch.Tensor]: Single-element tuple containing the binary mask

### Return Types

- `MASK`

::: nodes.mask.MaskBinaryFilter

## MaskInvert

Inverts a mask (1 becomes 0 and vice versa).

Parameters: mask (torch.Tensor): Input mask in BWHC format

Returns: tuple[torch.Tensor]: Single-element tuple containing the inverted mask

### Return Types

- `MASK`

::: nodes.mask.MaskInvert

## MaskGaussianBlur

Applies Gaussian blur to a mask.

Parameters: image (torch.Tensor): Input mask in BWHC format radius (int): Blur radius
(default: 13) sigma (float): Blur sigma/strength (default: 10.5) iterations (int):
Number of blur passes (default: 1) only_outline (bool): Whether to blur only the outline
(default: False)

Returns: tuple[torch.Tensor]: Single-element tuple containing the blurred mask

### Return Types

- `MASK`

::: nodes.mask.MaskGaussianBlur

## Mask2Image

Converts a single-channel mask to a 3-channel image.

Parameters: mask (torch.Tensor): Input mask in BWHC format

Returns: tuple[torch.Tensor]: Single-element tuple containing the converted image

### Return Types

- `IMAGE`

::: nodes.mask.Mask2Image

## MaskGrowWithBlur

Expands or contracts a mask with optional blur and tapering effects.

Parameters: mask (torch.Tensor): Input mask in BWHC format expand (int): Pixels to
expand (positive) or contract (negative) incremental_expandrate (float): Rate of
expansion per iteration tapered_corners (bool): Whether to taper corners (default: True)
flip_input (bool): Whether to invert input before processing (default: False)
blur_radius (float): Radius for final blur (default: 0.0) lerp_alpha (float): Linear
interpolation factor (default: 1.0) decay_factor (float): Decay factor for expansion
(default: 1.0)

Returns: tuple[torch.Tensor]: Single-element tuple containing the processed mask

### Return Types

- `MASK`

::: nodes.mask.MaskGrowWithBlur

## GetMaskShape

Returns the dimensions of an input mask.

Parameters: mask (torch.Tensor): Input mask in BWHC format

Returns: tuple[int, int, int, int, str]: Batch size, width, height, channels, and shape
string

### Return Types

- `INT`
- `INT`
- `INT`
- `INT`
- `STRING`

::: nodes.mask.GetMaskShape

## MaskPreview

Saves a preview of a mask as an image file.

Parameters: mask (torch.Tensor): Input mask in BWHC format filename_prefix (str): Prefix
for the output filename (default: "Signature") prompt (Optional[str]): Optional prompt
to include in metadata extra_pnginfo (Optional[dict]): Additional PNG metadata

Returns: tuple[str, str]: Paths to the saved preview images

::: nodes.mask.MaskPreview
