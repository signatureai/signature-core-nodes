# Mask Nodes

## BaseMask

Creates a basic binary mask of specified dimensions

### Inputs

| Group    | Name   | Type   | Default | Extras        |
| -------- | ------ | ------ | ------- | ------------- |
| required | color  | `LIST` |         |               |
| required | width  | `INT`  | 1024    | min=1, step=1 |
| required | height | `INT`  | 1024    | min=1, step=1 |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class BaseMask:
        """Creates a basic binary mask of specified dimensions.

        Parameters:
            color (str): Either "white" or "black" to set mask color
            width (int): Width of the output mask (default: 1024)
            height (int): Height of the output mask (default: 1024)

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing a binary mask in BWHC format
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "color": (["white", "black"],),
                    "width": ("INT", {"default": 1024, "min": 1, "max": MAX_INT, "step": 1}),
                    "height": ("INT", {"default": 1024, "min": 1, "max": MAX_INT, "step": 1}),
                }
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, **kwargs):
            color = kwargs.get("color") or "white"
            width = kwargs.get("width") or 1024
            height = kwargs.get("height") or 1024
            if color == "white":
                mask = torch.ones(1, 1, height, width)
            else:
                mask = torch.zeros(1, 1, height, width)
            mask = TensorImage(mask).get_BWHC()
            return (mask,)
    ```

## MaskMorphology

Applies morphological operations to a mask

### Inputs

| Group    | Name        | Type   | Default | Extras        |
| -------- | ----------- | ------ | ------- | ------------- |
| required | mask        | `MASK` |         |               |
| required | operation   | `LIST` |         |               |
| required | kernel_size | `INT`  | 1       | min=1, step=2 |
| required | iterations  | `INT`  | 5       | min=1, step=1 |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskMorphology:
        """Applies morphological operations to a mask.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format
            operation (str): One of: "dilation", "erosion", "opening", "closing",
                            "gradient", "top_hat", "bottom_hat"
            kernel_size (int): Size of the morphological kernel (default: 1)
            iterations (int): Number of times to apply the operation (default: 5)

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the processed mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                    "operation": (
                        [
                            "dilation",
                            "erosion",
                            "opening",
                            "closing",
                            "gradient",
                            "top_hat",
                            "bottom_hat",
                        ],
                    ),
                    "kernel_size": (
                        "INT",
                        {"default": 1, "min": 1, "max": MAX_INT, "step": 2},
                    ),
                    "iterations": (
                        "INT",
                        {"default": 5, "min": 1, "max": MAX_INT, "step": 1},
                    ),
                }
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, **kwargs):
            mask = kwargs.get("mask")
            if not isinstance(mask, torch.Tensor):
                raise ValueError("Mask must be a tensor")
            kernel = kwargs.get("kernel_size")
            iterations = kwargs.get("iterations")
            operation = kwargs.get("operation")
            step = TensorImage.from_BWHC(mask)

            if operation == "dilation":
                output = dilation(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "erosion":
                output = erosion(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "opening":
                output = opening(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "closing":
                output = closing(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "gradient":
                output = gradient(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "top_hat":
                output = top_hat(image=step, kernel_size=kernel, iterations=iterations)
            elif operation == "bottom_hat":
                output = bottom_hat(image=step, kernel_size=kernel, iterations=iterations)
            else:
                raise ValueError("Invalid operation")
            return (output.get_BWHC(),)
    ```

## MaskBitwise

Performs bitwise operations between two masks

### Inputs

| Group    | Name   | Type   | Default | Extras |
| -------- | ------ | ------ | ------- | ------ |
| required | mask_1 | `MASK` |         |        |
| required | mask_2 | `MASK` |         |        |
| required | mode   | `LIST` |         |        |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskBitwise:
        """Performs bitwise operations between two masks.

        Parameters:
            mask_1 (torch.Tensor): First input mask in BWHC format
            mask_2 (torch.Tensor): Second input mask in BWHC format
            mode (str): One of: "and", "or", "xor", "left_shift", "right_shift"

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the resulting mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask_1": ("MASK",),
                    "mask_2": ("MASK",),
                    "mode": (["and", "or", "xor", "left_shift", "right_shift"],),
                },
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask_1: torch.Tensor, mask_2: torch.Tensor, mode: str):
            input_mask_1 = TensorImage.from_BWHC(mask_1)
            input_mask_2 = TensorImage.from_BWHC(mask_2)
            eight_bit_mask_1 = torch.tensor(input_mask_1 * 255, dtype=torch.uint8)
            eight_bit_mask_2 = torch.tensor(input_mask_2 * 255, dtype=torch.uint8)

            if mode == "and":
                result = torch.bitwise_and(eight_bit_mask_1, eight_bit_mask_2)
            elif mode == "or":
                result = torch.bitwise_or(eight_bit_mask_1, eight_bit_mask_2)
            elif mode == "xor":
                result = torch.bitwise_xor(eight_bit_mask_1, eight_bit_mask_2)
            elif mode == "left_shift":
                result = torch.bitwise_left_shift(eight_bit_mask_1, eight_bit_mask_2)
            elif mode == "right_shift":
                result = torch.bitwise_right_shift(eight_bit_mask_1, eight_bit_mask_2)
            else:
                raise ValueError("Invalid mode")

            float_result = result.float() / 255
            output_mask = TensorImage(float_result).get_BWHC()
            return (output_mask,)
    ```

## MaskDistance

Calculates the Euclidean distance between two masks

### Inputs

| Group    | Name   | Type   | Default | Extras |
| -------- | ------ | ------ | ------- | ------ |
| required | mask_0 | `MASK` |         |        |
| required | mask_1 | `MASK` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| float | `FLOAT` |

??? note "Pick the code in mask.py"

    ```python
    class MaskDistance:
        """Calculates the Euclidean distance between two masks.

        Parameters:
            mask_0 (torch.Tensor): First input mask in BWHC format
            mask_1 (torch.Tensor): Second input mask in BWHC format

        Returns:
            tuple[float]: Single-element tuple containing the distance value
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {"required": {"mask_0": ("MASK",), "mask_1": ("MASK",)}}

        RETURN_TYPES = ("FLOAT",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, **kwargs):
            mask_0 = kwargs.get("mask_0")
            mask_1 = kwargs.get("mask_1")
            if not isinstance(mask_0, torch.Tensor) or not isinstance(mask_1, torch.Tensor):
                raise ValueError("Mask must be a tensor")
            tensor1 = TensorImage.from_BWHC(mask_0)
            tensor2 = TensorImage.from_BWHC(mask_1)
            dist = torch.Tensor((tensor1 - tensor2).pow(2).sum(3).sqrt().mean())
            return (dist,)
    ```

## Mask2Trimap

Converts a mask to a trimap representation (foreground, background, unknown regions)

### Inputs

| Group    | Name                | Type   | Default | Extras         |
| -------- | ------------------- | ------ | ------- | -------------- |
| required | mask                | `MASK` |         |                |
| required | inner_min_threshold | `INT`  | 200     | min=0, max=255 |
| required | inner_max_threshold | `INT`  | 255     | min=0, max=255 |
| required | outer_min_threshold | `INT`  | 15      | min=0, max=255 |
| required | outer_max_threshold | `INT`  | 240     | min=0, max=255 |
| required | kernel_size         | `INT`  | 10      | min=1, max=100 |

### Returns

| Name   | Type     |
| ------ | -------- |
| mask   | `MASK`   |
| trimap | `TRIMAP` |

??? note "Pick the code in mask.py"

    ```python
    class Mask2Trimap:
        """Converts a mask to a trimap representation (foreground, background, unknown regions).

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format
            inner_min_threshold (int): Minimum threshold for inner region (default: 200)
            inner_max_threshold (int): Maximum threshold for inner region (default: 255)
            outer_min_threshold (int): Minimum threshold for outer region (default: 15)
            outer_max_threshold (int): Maximum threshold for outer region (default: 240)
            kernel_size (int): Size of morphological kernel (default: 10)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mask and trimap tensors
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                    "inner_min_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
                    "inner_max_threshold": ("INT", {"default": 255, "min": 0, "max": 255}),
                    "outer_min_threshold": ("INT", {"default": 15, "min": 0, "max": 255}),
                    "outer_max_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                    "kernel_size": ("INT", {"default": 10, "min": 1, "max": 100}),
                }
            }

        RETURN_TYPES = ("MASK", "TRIMAP")
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, **kwargs):
            mask = kwargs.get("mask")
            inner_min_threshold = kwargs.get("inner_min_threshold") or 200
            inner_max_threshold = kwargs.get("inner_max_threshold") or 255
            outer_min_threshold = kwargs.get("outer_min_threshold") or 15
            outer_max_threshold = kwargs.get("outer_max_threshold") or 240
            kernel_size = kwargs.get("kernel_size")

            if not isinstance(mask, torch.Tensor):
                raise ValueError("Mask must be a tensor")

            step = TensorImage.from_BWHC(mask)
            inner_mask = TensorImage(step.clone())
            inner_mask[inner_mask > (inner_max_threshold / 255.0)] = 1.0
            inner_mask[inner_mask <= (inner_min_threshold / 255.0)] = 0.0

            step = TensorImage.from_BWHC(mask)
            inner_mask = erosion(image=inner_mask, kernel_size=kernel_size, iterations=1)

            inner_mask[inner_mask != 0.0] = 1.0

            outter_mask = step.clone()
            outter_mask[outter_mask > (outer_max_threshold / 255.0)] = 1.0
            outter_mask[outter_mask <= (outer_min_threshold / 255.0)] = 0.0
            outter_mask = dilation(image=inner_mask, kernel_size=kernel_size, iterations=5)

            outter_mask[outter_mask != 0.0] = 1.0

            trimap_im = torch.zeros_like(step)
            trimap_im[outter_mask == 1.0] = 0.5
            trimap_im[inner_mask == 1.0] = 1.0
            batch_size = step.shape[0]

            trimap = torch.zeros(
                batch_size, 2, step.shape[2], step.shape[3], dtype=step.dtype, device=step.device
            )
            for i in range(batch_size):
                tar_trimap = trimap_im[i][0]
                trimap[i][1][tar_trimap == 1] = 1
                trimap[i][0][tar_trimap == 0] = 1

            output_0 = TensorImage(trimap_im).get_BWHC()
            output_1 = trimap.permute(0, 2, 3, 1)

            return (
                output_0,
                output_1,
            )
    ```

## MaskBinaryFilter

Applies binary thresholding to a mask

### Inputs

| Group    | Name      | Type    | Default | Extras                      |
| -------- | --------- | ------- | ------- | --------------------------- |
| required | mask      | `MASK`  |         |                             |
| required | threshold | `FLOAT` | 0.01    | min=0.0, max=1.0, step=0.01 |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskBinaryFilter:
        """Applies binary thresholding to a mask.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format
            threshold (float): Threshold value between 0 and 1 (default: 0.01)

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the binary mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                    "threshold": ("FLOAT", {"default": 0.01, "min": 0.00, "max": 1.00, "step": 0.01}),
                }
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask: torch.Tensor, threshold: float):
            step = TensorImage.from_BWHC(mask)
            step[step > threshold] = 1.0
            step[step <= threshold] = 0.0
            output = TensorImage(step).get_BWHC()
            return (output,)
    ```

## MaskInvert

Inverts a mask (1 becomes 0 and vice versa)

### Inputs

| Group    | Name | Type   | Default | Extras |
| -------- | ---- | ------ | ------- | ------ |
| required | mask | `MASK` |         |        |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskInvert:
        """Inverts a mask (1 becomes 0 and vice versa).

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the inverted mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                }
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask: torch.Tensor):
            step = TensorImage.from_BWHC(mask)
            step = 1.0 - step
            output = TensorImage(step).get_BWHC()
            return (output,)
    ```

## MaskGaussianBlur

Applies Gaussian blur to a mask

### Inputs

| Group    | Name         | Type      | Default | Extras |
| -------- | ------------ | --------- | ------- | ------ |
| required | image        | `MASK`    |         |        |
| required | radius       | `INT`     | 13      |        |
| required | sigma        | `FLOAT`   | 10.5    |        |
| required | interations  | `INT`     | 1       |        |
| required | only_outline | `BOOLEAN` | False   |        |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskGaussianBlur:
        """Applies Gaussian blur to a mask.

        Parameters:
            image (torch.Tensor): Input mask in BWHC format
            radius (int): Blur radius (default: 13)
            sigma (float): Blur sigma/strength (default: 10.5)
            iterations (int): Number of blur passes (default: 1)
            only_outline (bool): Whether to blur only the outline (default: False)

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the blurred mask
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "image": ("MASK",),
                    "radius": ("INT", {"default": 13}),
                    "sigma": ("FLOAT", {"default": 10.5}),
                    "interations": ("INT", {"default": 1}),
                    "only_outline": ("BOOLEAN", {"default": False}),
                }
            }

        RETURN_TYPES = ("MASK",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, image: torch.Tensor, radius, sigma, interations):
            tensor_image = TensorImage.from_BWHC(image)
            output = gaussian_blur2d(tensor_image, radius, sigma, interations).get_BWHC()
            return (output,)
    ```

## Mask2Image

Converts a single-channel mask to a 3-channel image

### Inputs

| Group    | Name | Type   | Default | Extras |
| -------- | ---- | ------ | ------- | ------ |
| required | mask | `MASK` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in mask.py"

    ```python
    class Mask2Image:
        """Converts a single-channel mask to a 3-channel image.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the converted image
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask: torch.Tensor):
            mask_tensor = TensorImage.from_BWHC(mask)
            output = mask_tensor.repeat(1, 3, 1, 1)
            output = TensorImage(output).get_BWHC()
            return (output,)
    ```

## MaskGrowWithBlur

Expands or contracts a mask with optional blur and tapering effects

### Inputs

| Group    | Name                   | Type      | Default | Extras                       |
| -------- | ---------------------- | --------- | ------- | ---------------------------- |
| required | mask                   | `MASK`    |         |                              |
| required | expand                 | `INT`     | 0       | step=1                       |
| required | incremental_expandrate | `FLOAT`   | 0.0     | min=0.0, max=100.0, step=0.1 |
| required | tapered_corners        | `BOOLEAN` | True    |                              |
| required | flip_input             | `BOOLEAN` | False   |                              |
| required | blur_radius            | `FLOAT`   | 0.0     | min=0.0, max=100, step=0.1   |
| required | lerp_alpha             | `FLOAT`   | 1.0     | min=0.0, max=1.0, step=0.01  |
| required | decay_factor           | `FLOAT`   | 1.0     | min=0.0, max=1.0, step=0.01  |

### Returns

| Name | Type   |
| ---- | ------ |
| mask | `MASK` |

??? note "Pick the code in mask.py"

    ```python
    class MaskGrowWithBlur:
        """Expands or contracts a mask with optional blur and tapering effects.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format
            expand (int): Pixels to expand (positive) or contract (negative)
            incremental_expandrate (float): Rate of expansion per iteration
            tapered_corners (bool): Whether to taper corners (default: True)
            flip_input (bool): Whether to invert input before processing (default: False)
            blur_radius (float): Radius for final blur (default: 0.0)
            lerp_alpha (float): Linear interpolation factor (default: 1.0)
            decay_factor (float): Decay factor for expansion (default: 1.0)

        Returns:
            tuple[torch.Tensor]: Single-element tuple containing the processed mask
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "mask": ("MASK",),
                    "expand": (
                        "INT",
                        {
                            "default": 0,
                            "min": -MAX_INT,
                            "max": MAX_INT,
                            "step": 1,
                        },
                    ),
                    "incremental_expandrate": (
                        "FLOAT",
                        {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    ),
                    "tapered_corners": ("BOOLEAN", {"default": True}),
                    "flip_input": ("BOOLEAN", {"default": False}),
                    "blur_radius": (
                        "FLOAT",
                        {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1},
                    ),
                    "lerp_alpha": (
                        "FLOAT",
                        {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                    ),
                    "decay_factor": (
                        "FLOAT",
                        {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                    ),
                },
            }

        CATEGORY = MASK_CAT
        RETURN_TYPES = ("MASK",)
        RETURN_NAMES = ("mask",)
        FUNCTION = "expand_mask"

        def expand_mask(self, **kwargs):
            mask = kwargs.get("mask")
            if not isinstance(mask, torch.Tensor):
                raise ValueError("Mask must be a tensor")
            expand = kwargs.get("expand")
            if not isinstance(expand, int):
                raise ValueError("Expand must be an integer")
            incremental_expandrate = kwargs.get("incremental_expandrate")
            if not isinstance(incremental_expandrate, float):
                raise ValueError("Incremental expandrate must be a float")
            tapered_corners = kwargs.get("tapered_corners")
            if not isinstance(tapered_corners, bool):
                raise ValueError("Tapered corners must be a boolean")
            flip_input = kwargs.get("flip_input")
            if not isinstance(flip_input, bool):
                raise ValueError("Flip input must be a boolean")
            blur_radius = kwargs.get("blur_radius")
            if not isinstance(blur_radius, float):
                raise ValueError("Blur radius must be a float")
            lerp_alpha = kwargs.get("lerp_alpha")
            if not isinstance(lerp_alpha, float):
                raise ValueError("Lerp alpha must be a float")
            decay_factor = kwargs.get("decay_factor")
            if not isinstance(decay_factor, float):
                raise ValueError("Decay factor must be a float")
            mask = TensorImage.from_BWHC(mask)
            alpha = lerp_alpha
            decay = decay_factor
            if flip_input:
                mask = 1.0 - mask
            c = 0 if tapered_corners else 1
            kernel = torch.tensor([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=torch.float32)
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            previous_output = None
            current_expand = expand
            for m in growmask:
                m = m.unsqueeze(0).unsqueeze(0)
                output = m.clone()

                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = morphology.erosion(output, kernel)
                    else:
                        output = morphology.dilation(output, kernel)
                if current_expand < 0:
                    current_expand -= abs(incremental_expandrate)
                else:
                    current_expand += abs(incremental_expandrate)

                output = output.squeeze(0).squeeze(0)

                if alpha < 1.0 and previous_output is not None:
                    output = alpha * output + (1 - alpha) * previous_output
                if decay < 1.0 and previous_output is not None:
                    output += decay * previous_output
                    output = output / output.max()
                previous_output = output
                out.append(output)

            if blur_radius != 0:
                kernel_size = int(4 * round(blur_radius) + 1)
                blurred = [
                    filters.gaussian_blur2d(
                        tensor.unsqueeze(0).unsqueeze(0), (kernel_size, kernel_size), (blur_radius, blur_radius)
                    ).squeeze(0)
                    for tensor in out
                ]
                blurred = torch.cat(blurred, dim=0)

                return (TensorImage(blurred).get_BWHC(),)

            return (TensorImage(torch.stack(out, dim=0)).get_BWHC(),)
    ```

## GetMaskShape

Returns the dimensions of an input mask

### Inputs

| Group    | Name | Type   | Default | Extras |
| -------- | ---- | ------ | ------- | ------ |
| required | mask | `MASK` |         |        |

### Returns

| Name   | Type     |
| ------ | -------- |
| int    | `INT`    |
| int    | `INT`    |
| int    | `INT`    |
| int    | `INT`    |
| string | `STRING` |

??? note "Pick the code in mask.py"

    ```python
    class GetMaskShape:
        """Returns the dimensions of an input mask.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format

        Returns:
            tuple[int, int, int, int, str]: Batch size, width, height, channels, and shape string
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "mask": ("MASK",),
                },
            }

        RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
        RETURN_NAMES = ("batch", "width", "height", "channels", "debug")
        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask):
            if len(mask.shape) == 3:
                return (mask.shape[0], mask.shape[2], mask.shape[1], 1, str(mask.shape))
            return (mask.shape[0], mask.shape[2], mask.shape[1], mask.shape[3], str(mask.shape))
    ```

## MaskPreview

Saves a preview of a mask as an image file

### Inputs

| Group    | Name | Type   | Default | Extras |
| -------- | ---- | ------ | ------- | ------ |
| required | mask | `MASK` |         |        |

??? note "Pick the code in mask.py"

    ```python
    class MaskPreview(SaveImage):
        """Saves a preview of a mask as an image file.

        Parameters:
            mask (torch.Tensor): Input mask in BWHC format
            filename_prefix (str): Prefix for the output filename (default: "Signature")
            prompt (Optional[str]): Optional prompt to include in metadata
            extra_pnginfo (Optional[dict]): Additional PNG metadata

        Returns:
            tuple[str, str]: Paths to the saved preview images
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
                    "mask": ("MASK",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

        FUNCTION = "execute"
        CATEGORY = MASK_CAT

        def execute(self, mask, filename_prefix="Signature", prompt=None, extra_pnginfo=None):
            preview = TensorImage.from_BWHC(mask).get_rgb_or_rgba().get_BWHC()
            return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)
    ```
