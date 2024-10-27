# Models Nodes

## MagicEraser

Removes content from an image based on a mask using the Lama inpainting model

### Inputs

| Group    | Name    | Type    | Default | Extras |
| -------- | ------- | ------- | ------- | ------ |
| required | image   | `IMAGE` |         |        |
| required | mask    | `MASK`  |         |        |
| required | preview | `LIST`  |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in models.py"

    ```python
    class MagicEraser(SaveImage):
        """Removes content from an image based on a mask using the Lama inpainting model.

        Parameters:
            image (torch.Tensor): Input image in BWHC format
            mask (torch.Tensor): Mask indicating areas to erase
            preview (str): Whether to save preview images ("on" or "off")
            filename_prefix (str, optional): Prefix for saved files
            prompt (str, optional): Optional prompt for metadata
            extra_pnginfo (dict, optional): Additional PNG metadata

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the processed image
        """

        def __init__(self):
            self.output_dir = folder_paths.get_temp_directory()
            self.type = "temp"
            self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
            self.compress_level = 4

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "image": ("IMAGE",),
                    "mask": ("MASK",),
                    "preview": (["on", "off"],),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"
        CATEGORY = MODELS_CAT

        def execute(
            self,
            image: torch.Tensor,
            mask: torch.Tensor,
            preview: str,
            filename_prefix="Signature",
            prompt=None,
            extra_pnginfo=None,
        ):
            model = Lama()
            input_image = TensorImage.from_BWHC(image)
            input_mask = TensorImage.from_BWHC(mask)
            highres = TensorImage(model.forward(input_image, input_mask, "FIXED"))
            output_images = highres.get_BWHC()
            if preview == "off":
                return (output_images,)
            result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
            result.update({"result": (output_images,)})
            del model
            model = None
            return result
    ```

## Unblur

Reduces blur in an image using the SeeMore model

### Inputs

| Group    | Name    | Type    | Default | Extras |
| -------- | ------- | ------- | ------- | ------ |
| required | image   | `IMAGE` |         |        |
| required | preview | `LIST`  |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in models.py"

    ```python
    class Unblur(SaveImage):
        """Reduces blur in an image using the SeeMore model.

        Parameters:
            image (torch.Tensor): Input image in BWHC format
            preview (str): Whether to save preview images ("on" or "off")
            filename_prefix (str, optional): Prefix for saved files
            prompt (str, optional): Optional prompt for metadata
            extra_pnginfo (dict, optional): Additional PNG metadata

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the unblurred image
        """

        def __init__(self):
            self.output_dir = folder_paths.get_temp_directory()
            self.type = "temp"
            self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
            self.compress_level = 4

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "image": ("IMAGE",),
                    "preview": (["on", "off"],),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"
        CATEGORY = MODELS_CAT

        def execute(
            self, image: torch.Tensor, preview: str, filename_prefix="Signature", prompt=None, extra_pnginfo=None
        ):
            model = SeeMore()
            input_image = TensorImage.from_BWHC(image)
            output_image = model.forward(input_image)
            output_images = TensorImage(output_image).get_BWHC()

            if preview == "off":
                return (output_images,)
            result = self.save_images(output_images, filename_prefix, prompt, extra_pnginfo)
            result.update({"result": (output_images,)})
            del model
            model = None
            return result
    ```

## BackgroundRemoval

Removes the background from an image using various AI models

### Inputs

| Group    | Name       | Type    | Default | Extras |
| -------- | ---------- | ------- | ------- | ------ |
| required | model_name | `LIST`  |         |        |
| required | preview    | `LIST`  |         |        |
| required | image      | `IMAGE` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |
| image | `IMAGE` |
| mask  | `MASK`  |

??? note "Pick the code in models.py"

    ```python
    class BackgroundRemoval(SaveImage):
        """Removes the background from an image using various AI models.

        Parameters:
            image (torch.Tensor): Input image in BWHC format
            model_name (str): Model to use ("inspyrenet", "rmbg14", "isnet_general", "fakepng")
            preview (str): Preview mode ("mask", "rgba", "none")
            filename_prefix (str, optional): Prefix for saved files
            prompt (str, optional): Optional prompt for metadata
            extra_pnginfo (dict, optional): Additional PNG metadata

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: RGBA, RGB, and mask versions of the processed image
        """

        def __init__(self):
            self.output_dir = folder_paths.get_temp_directory()
            self.type = "temp"
            self.prefix_append = "_temp_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
            self.compress_level = 4

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "model_name": (["inspyrenet", "rmbg14", "isnet_general", "fakepng"],),
                    "preview": (["mask", "rgba", "none"],),
                    "image": ("IMAGE",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

        RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
        RETURN_NAMES = ("rgba", "rgb", "mask")
        FUNCTION = "execute"
        CATEGORY = MODELS_CAT

        def execute(
            self,
            image: torch.Tensor,
            model_name: str,
            preview: str,
            filename_prefix="Signature",
            prompt=None,
            extra_pnginfo=None,
        ):

            model = SalientObjectDetection(model_name=model_name)
            input_image = TensorImage.from_BWHC(image)
            masks = model.forward(input_image)

            output_masks = TensorImage(masks)
            rgb, rgba = cutout(input_image, output_masks)
            rgb_output = TensorImage(rgb).get_BWHC()
            rgba_output = TensorImage(rgba).get_BWHC()
            mask_output = output_masks.get_BWHC()
            if preview == "none":
                return (
                    rgba_output,
                    rgb_output,
                    mask_output,
                )
            preview_images = output_masks.get_rgb_or_rgba().get_BWHC() if preview == "mask" else rgba_output
            result = self.save_images(preview_images, filename_prefix, prompt, extra_pnginfo)
            result.update(
                {
                    "result": (
                        rgba_output,
                        rgb_output,
                        mask_output,
                    )
                }
            )
            del model
            model = None
            return result
    ```
