# Models Nodes

## MagicEraser

Removes content from an image based on a mask using the Lama inpainting model.

Parameters: image (torch.Tensor): Input image in BWHC format mask (torch.Tensor): Mask
indicating areas to erase preview (str): Whether to save preview images ("on" or "off")
filename_prefix (str, optional): Prefix for saved files prompt (str, optional): Optional
prompt for metadata extra_pnginfo (dict, optional): Additional PNG metadata

Returns: tuple[torch.Tensor]: Single-element tuple containing the processed image

### Return Types

- `IMAGE`

::: nodes.models.MagicEraser

## Unblur

Reduces blur in an image using the SeeMore model.

Parameters: image (torch.Tensor): Input image in BWHC format preview (str): Whether to
save preview images ("on" or "off") filename_prefix (str, optional): Prefix for saved
files prompt (str, optional): Optional prompt for metadata extra_pnginfo (dict,
optional): Additional PNG metadata

Returns: tuple[torch.Tensor]: Single-element tuple containing the unblurred image

### Return Types

- `IMAGE`

::: nodes.models.Unblur

## BackgroundRemoval

Removes the background from an image using various AI models.

Parameters: image (torch.Tensor): Input image in BWHC format model_name (str): Model to
use ("inspyrenet", "rmbg14", "isnet_general", "fakepng") preview (str): Preview mode
("mask", "rgba", "none") filename_prefix (str, optional): Prefix for saved files prompt
(str, optional): Optional prompt for metadata extra_pnginfo (dict, optional): Additional
PNG metadata

Returns: tuple[torch.Tensor, torch.Tensor, torch.Tensor]: RGBA, RGB, and mask versions
of the processed image

### Return Types

- `IMAGE`
- `IMAGE`
- `MASK`

::: nodes.models.BackgroundRemoval
