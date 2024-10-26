# File Nodes

## ImageFromWeb

Loads an image from a URL.

Parameters: url (str): URL of the image to load

Returns: tuple[torch.Tensor, torch.Tensor]: Image tensor in BWHC format and mask

### Return Types

- `IMAGE`
- `MASK`

::: nodes.file.ImageFromWeb

## ImageFromBase64

Converts a base64 string to an image.

Parameters: base64 (str): Base64-encoded image string

Returns: tuple[torch.Tensor, torch.Tensor]: Image tensor in BWHC format and mask

### Return Types

- `IMAGE`
- `MASK`

::: nodes.file.ImageFromBase64

## Base64FromImage

Converts an image to a base64 string.

Parameters: image (torch.Tensor): Input image in BWHC format

Returns: tuple[str]: Base64-encoded string representation of the image

### Return Types

- `STRING`

::: nodes.file.Base64FromImage

## FileLoader

Loads file data from a string value.

Parameters: value (str): Input string containing file data (JSON format)

Returns: tuple[list]: List of file data with updated paths

### Return Types

- `FILE`

::: nodes.file.FileLoader

## FolderLoader

Loads file data from a folder path.

Parameters: value (str): Input string containing folder path data

Returns: tuple[list]: List of file data with updated paths from the folder

### Return Types

- `FILE`

::: nodes.file.FolderLoader

## Rotate

Rotates an image and mask by a specified angle.

Parameters: image (torch.Tensor, optional): Input image in BWHC format mask
(torch.Tensor, optional): Input mask in BWHC format angle (float): Rotation angle in
degrees (0-360) zoom_to_fit (bool): Whether to zoom to fit rotated content (default:
False)

Returns: tuple[torch.Tensor, torch.Tensor]: Rotated image and mask

::: nodes.file.Rotate

## MaskGaussianBlur

Applies Gaussian blur to a mask.

Parameters: image (torch.Tensor): Input mask in BWHC format radius (int): Blur radius
(default: 13) sigma (float): Blur sigma/strength (default: 10.5) iterations (int):
Number of blur passes (default: 1) only_outline (bool): Whether to blur only the outline
(default: False)

Returns: tuple[torch.Tensor]: Single-element tuple containing the blurred mask

::: nodes.file.MaskGaussianBlur
