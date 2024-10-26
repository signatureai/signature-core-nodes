# Transform Nodes

## AutoCrop

Automatically crops an image based on a mask.

Parameters: image (torch.Tensor): Input image in BWHC format mask (torch.Tensor): Input
mask in BWHC format mask_threshold (float): Threshold for mask detection (0.0-1.0)
left_padding (int): Additional padding on left side right_padding (int): Additional
padding on right side top_padding (int): Additional padding on top bottom_padding (int):
Additional padding on bottom

Returns: tuple[torch.Tensor, torch.Tensor, int, int, int, int]: - Cropped image - Cropped
mask - X coordinate of crop - Y coordinate of crop - Width of crop - Height of crop

### Return Types

- `IMAGE`
- `MASK`
- `INT`
- `INT`
- `INT`
- `INT`

::: nodes.transform.AutoCrop

## Rescale

Rescales an image and mask by a given factor.

Parameters: image (torch.Tensor, optional): Input image in BWHC format mask
(torch.Tensor, optional): Input mask in BWHC format factor (float): Scale factor
(default: 2.0) interpolation (str): Interpolation method (default: "nearest") antialias
(bool): Whether to use antialiasing (default: True)

Returns: tuple[torch.Tensor, torch.Tensor]: Rescaled image and mask

### Return Types

- `IMAGE`
- `MASK`

::: nodes.transform.Rescale

## Resize

Resizes an image and mask to specified dimensions.

Parameters: image (torch.Tensor, optional): Input image in BWHC format mask
(torch.Tensor, optional): Input mask in BWHC format width (int): Target width
(default: 1024) height (int): Target height (default: 1024) mode (str): Resize mode
("STRETCH", "FIT", "FILL", "ASPECT") interpolation (str): Interpolation method (default:
"nearest") antialias (bool): Whether to use antialiasing (default: True)

Returns: tuple[torch.Tensor, torch.Tensor]: Resized image and mask

### Return Types

- `IMAGE`
- `MASK`

::: nodes.transform.Resize

## Cutout

Cuts out a portion of an image based on a mask.

Parameters: image (torch.Tensor): Input image in BWHC format mask (torch.Tensor): Mask
defining the cutout area

Returns: tuple[torch.Tensor, torch.Tensor]: RGB and RGBA versions of the cutout image

### Return Types

- `IMAGE`
- `IMAGE`

::: nodes.transform.Cutout
