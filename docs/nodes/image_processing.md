# Image Processing Nodes

## AutoCrop

Automatically crops an image based on a mask

### Inputs

| Group    | Name           | Type    | Default | Extras                      |
| -------- | -------------- | ------- | ------- | --------------------------- |
| required | image          | `IMAGE` |         |                             |
| required | mask           | `MASK`  |         |                             |
| required | mask_threshold | `FLOAT` | 0.1     | min=0.0, max=1.0, step=0.01 |
| required | left_padding   | `INT`   | 0       |                             |
| required | right_padding  | `INT`   | 0       |                             |
| required | top_padding    | `INT`   | 0       |                             |
| required | bottom_padding | `INT`   | 0       |                             |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |
| mask  | `MASK`  |
| int   | `INT`   |
| int   | `INT`   |
| int   | `INT`   |
| int   | `INT`   |

??? note "Pick the code in image_processing.py"

    ```python
    class AutoCrop:
        """Automatically crops an image based on a mask.

        Parameters:
            image (torch.Tensor): Input image in BWHC format
            mask (torch.Tensor): Input mask in BWHC format
            mask_threshold (float): Threshold for mask detection (0.0-1.0)
            left_padding (int): Additional padding on left side
            right_padding (int): Additional padding on right side
            top_padding (int): Additional padding on top
            bottom_padding (int): Additional padding on bottom

        Returns:
            tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
                - Cropped image
                - Cropped mask
                - X coordinate of crop
                - Y coordinate of crop
                - Width of crop
                - Height of crop
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "image": ("IMAGE",),
                    "mask": ("MASK",),
                    "mask_threshold": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 1.00, "step": 0.01}),
                    "left_padding": ("INT", {"default": 0}),
                    "right_padding": ("INT", {"default": 0}),
                    "top_padding": ("INT", {"default": 0}),
                    "bottom_padding": ("INT", {"default": 0}),
                }
            }

        RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
        RETURN_NAMES = ("cropped_image", "cropped_mask", "x", "y", "width", "height")

        FUNCTION = "execute"
        CATEGORY = IMAGE_PROCESSING_CAT

        def execute(self, **kwargs):
            img_tensor = TensorImage.from_BWHC(kwargs["image"])
            mask_tensor = TensorImage.from_BWHC(kwargs["mask"])
            if img_tensor.shape[1] != 3:
                img_tensor = rgba_to_rgb(img_tensor)

            padding = (
                kwargs["left_padding"],
                kwargs["right_padding"],
                kwargs["top_padding"],
                kwargs["bottom_padding"],
            )
            img_result, mask_result, min_x, min_y, width, height = auto_crop(
                img_tensor, mask_tensor, mask_threshold=kwargs["mask_threshold"], padding=padding
            )
            output_img = TensorImage(img_result).get_BWHC()
            output_mask = TensorImage(mask_result).get_BWHC()

            return (output_img, output_mask, min_x, min_y, width, height)
    ```

## Rescale

Rescales an image and mask by a given factor

### Inputs

| Group    | Name          | Type      | Default | Extras                         |
| -------- | ------------- | --------- | ------- | ------------------------------ |
| optional | image         | `IMAGE`   | None    |                                |
| optional | mask          | `MASK`    | None    |                                |
| optional | factor        | `FLOAT`   | 2.0     | min=0.01, max=100.0, step=0.01 |
| optional | interpolation | `LIST`    |         |                                |
| optional | antialias     | `BOOLEAN` | True    |                                |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |
| mask  | `MASK`  |

??? note "Pick the code in image_processing.py"

    ```python
    class Rescale:
        """Rescales an image and mask by a given factor.

        Parameters:
            image (torch.Tensor, optional): Input image in BWHC format
            mask (torch.Tensor, optional): Input mask in BWHC format
            factor (float): Scale factor (default: 2.0)
            interpolation (str): Interpolation method (default: "nearest")
            antialias (bool): Whether to use antialiasing (default: True)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rescaled image and mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {},
                "optional": {
                    "image": ("IMAGE", {"default": None}),
                    "mask": ("MASK", {"default": None}),
                    "factor": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                    "interpolation": (
                        ["nearest", "nearest-exact", "bilinear", "bicubic", "box", "hamming", "lanczos"],
                    ),
                    "antialias": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = (
            "IMAGE",
            "MASK",
        )
        FUNCTION = "execute"
        CATEGORY = IMAGE_PROCESSING_CAT

        def execute(self, **kwargs):
            image = kwargs.get("image")
            mask = kwargs.get("mask")
            if not isinstance(image, torch.Tensor) and not isinstance(mask, torch.Tensor):
                raise ValueError("Either image or mask must be provided")

            input_image = (
                TensorImage.from_BWHC(image)
                if isinstance(image, torch.Tensor)
                else TensorImage(torch.zeros((1, 3, 1, 1)))
            )
            input_mask = (
                TensorImage.from_BWHC(mask)
                if isinstance(mask, torch.Tensor)
                else TensorImage(torch.zeros((1, 1, 1, 1)))
            )
            output_image = rescale(
                input_image,
                kwargs.get("factor", 2.0),
                kwargs.get("interpolation", "nearest"),
                kwargs.get("antialias", True),
            ).get_BWHC()
            output_mask = rescale(
                input_mask,
                kwargs.get("factor", 2.0),
                kwargs.get("interpolation", "nearest"),
                kwargs.get("antialias", True),
            ).get_BWHC()

            return (
                output_image,
                output_mask,
            )
    ```

## Resize

Resizes an image and mask to specified dimensions

### Inputs

| Group    | Name          | Type      | Default | Extras                    |
| -------- | ------------- | --------- | ------- | ------------------------- |
| optional | image         | `IMAGE`   | None    |                           |
| optional | mask          | `MASK`    | None    |                           |
| optional | width         | `INT`     | 1024    | min=32, step=2, max=40960 |
| optional | height        | `INT`     | 1024    | min=32, step=2, max=40960 |
| optional | mode          | `LIST`    |         |                           |
| optional | interpolation | `LIST`    |         |                           |
| optional | antialias     | `BOOLEAN` | True    |                           |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |
| mask  | `MASK`  |

??? note "Pick the code in image_processing.py"

    ```python
    class Resize:
        """Resizes an image and mask to specified dimensions.

        Parameters:
            image (torch.Tensor, optional): Input image in BWHC format
            mask (torch.Tensor, optional): Input mask in BWHC format
            width (int): Target width (default: 1024)
            height (int): Target height (default: 1024)
            mode (str): Resize mode ("STRETCH", "FIT", "FILL", "ASPECT")
            interpolation (str): Interpolation method (default: "nearest")
            antialias (bool): Whether to use antialiasing (default: True)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Resized image and mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {},
                "optional": {
                    "image": ("IMAGE", {"default": None}),
                    "mask": ("MASK", {"default": None}),
                    "width": ("INT", {"default": 1024, "min": 32, "step": 2, "max": 40960}),
                    "height": ("INT", {"default": 1024, "min": 32, "step": 2, "max": 40960}),
                    "mode": (["STRETCH", "FIT", "FILL", "ASPECT"],),
                    "interpolation": (["bilinear", "nearest", "bicubic", "area"],),
                    "antialias": (
                        "BOOLEAN",
                        {"default": True},
                    ),
                },
            }

        RETURN_TYPES = (
            "IMAGE",
            "MASK",
        )
        FUNCTION = "execute"
        CATEGORY = IMAGE_PROCESSING_CAT

        def execute(self, **kwargs):
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            mode = kwargs.get("mode", "default")
            interpolation = kwargs.get("interpolation", "nearest")
            antialias = kwargs.get("antialias", True)
            image = kwargs.get("image", None)
            mask = kwargs.get("mask", None)

            input_image = (
                TensorImage.from_BWHC(image)
                if isinstance(image, torch.Tensor)
                else TensorImage(torch.zeros((1, 3, width, height)))
            )
            input_mask = (
                TensorImage.from_BWHC(mask)
                if isinstance(mask, torch.Tensor)
                else TensorImage(torch.zeros((1, 1, width, height)))
            )
            output_image = resize(input_image, width, height, mode, interpolation, antialias).get_BWHC()
            output_mask = resize(input_mask, width, height, mode, interpolation, antialias).get_BWHC()

            return (
                output_image,
                output_mask,
            )
    ```

## Cutout

Cuts out a portion of an image based on a mask

### Inputs

| Group    | Name  | Type    | Default | Extras |
| -------- | ----- | ------- | ------- | ------ |
| required | image | `IMAGE` |         |        |
| required | mask  | `MASK`  |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |
| image | `IMAGE` |

??? note "Pick the code in image_processing.py"

    ```python
    class Cutout:
        """Cuts out a portion of an image based on a mask.

        Parameters:
            image (torch.Tensor): Input image in BWHC format
            mask (torch.Tensor): Mask defining the cutout area

        Returns:
            tuple[torch.Tensor, torch.Tensor]: RGB and RGBA versions of the cutout image
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": ("IMAGE",),
                    "mask": ("MASK",),
                },
            }

        RETURN_TYPES = ("IMAGE", "IMAGE")
        RETURN_NAMES = ("rgb", "rgba")
        FUNCTION = "execute"
        CATEGORY = IMAGE_PROCESSING_CAT

        def execute(self, **kwargs):
            image = kwargs.get("image")
            mask = kwargs.get("mask")

            if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise ValueError("Either image or mask must be provided")

            tensor_image = TensorImage.from_BWHC(image)
            tensor_mask = TensorImage.from_BWHC(mask, image.device)

            image_rgb, image_rgba = cutout(tensor_image, tensor_mask)

            out_image_rgb = TensorImage(image_rgb).get_BWHC()
            out_image_rgba = TensorImage(image_rgba).get_BWHC()

            return (
                out_image_rgb,
                out_image_rgba,
            )
    ```
