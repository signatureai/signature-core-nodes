# Utils Nodes

## Any2String

Converts any input value to its string representation

### Inputs

| Group    | Name  | Type                                  | Default | Extras |
| -------- | ----- | ------------------------------------- | ------- | ------ |
| required | value | `<ast.Name object at 0x7f0a337c1480>` |         |        |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in utils.py"

    ```python
    class Any2String:
        """Converts any input value to its string representation.

        A utility node that takes any input value and converts it to a string using Python's str() function.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "value": (any_type,),
                }
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = UTILS_CAT

        def execute(self, value):
            return (str(value),)
    ```

## Any2Image

Converts any inputs value to image format

### Inputs

| Group    | Name  | Type                                  | Default | Extras |
| -------- | ----- | ------------------------------------- | ------- | ------ |
| required | value | `<ast.Name object at 0x7f0a337c3010>` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in utils.py"

    ```python
    class Any2Image:
        """Converts any inputs value to image format.

        A utility node that converts tensor inputs to image format. Currently only supports torch.Tensor inputs.

        Raises:
            ValueError: If the input value is not a torch.Tensor.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "value": (any_type,),
                }
            }

        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "execute"
        CATEGORY = UTILS_CAT

        def execute(self, value):
            if isinstance(value, torch.Tensor):
                return (value,)
            raise ValueError(f"Unsupported type: {type(value)}")
    ```

## Any2Any

Passes through any input value unchanged

### Inputs

| Group    | Name  | Type                                  | Default | Extras |
| -------- | ----- | ------------------------------------- | ------- | ------ |
| required | value | `<ast.Name object at 0x7f0a337c0ee0>` |         |        |

??? note "Pick the code in utils.py"

    ```python
    class Any2Any:
        """Passes through any input value unchanged.

        A utility node that acts as a pass-through, returning the input value without modification.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "value": (any_type,),
                }
            }

        RETURN_TYPES = (any_type,)
        FUNCTION = "execute"
        CATEGORY = UTILS_CAT

        def execute(self, value):
            return (value,)
    ```

## RGB2HSV

Converts RGB images to HSV color space

### Inputs

| Group    | Name  | Type    | Default | Extras |
| -------- | ----- | ------- | ------- | ------ |
| required | image | `IMAGE` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in utils.py"

    ```python
    class RGB2HSV:
        """Converts RGB images to HSV color space.

        A utility node that converts RGB format images to HSV (Hue, Saturation, Value) color space.
        Expects input images in BWHC format.
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
            output = rgb_to_hsv(image_tensor).get_BWHC()
            return (output,)
    ```

## RGBHLS

Converts RGB images to HLS color space

### Inputs

| Group    | Name  | Type    | Default | Extras |
| -------- | ----- | ------- | ------- | ------ |
| required | image | `IMAGE` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in utils.py"

    ```python
    class RGBHLS:
        """Converts RGB images to HLS color space.

        A utility node that converts RGB format images to HLS (Hue, Lightness, Saturation) color space.
        Expects input images in BWHC format.
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
            output = rgb_to_hls(image_tensor).get_BWHC()
            return (output,)
    ```

## RGBA2RGB

Converts RGBA images to RGB format

### Inputs

| Group    | Name  | Type    | Default | Extras |
| -------- | ----- | ------- | ------- | ------ |
| required | image | `IMAGE` |         |        |

### Returns

| Name  | Type    |
| ----- | ------- |
| image | `IMAGE` |

??? note "Pick the code in utils.py"

    ```python
    class RGBA2RGB:
        """Converts RGBA images to RGB format.

        A utility node that converts RGBA (Red, Green, Blue, Alpha) images to RGB format.
        If the input image is already in RGB format, it will be passed through unchanged.
        Expects input images in BWHC format.
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
            if image_tensor.shape[1] == 4:
                image_tensor = rgba_to_rgb(image_tensor)
            output = image_tensor.get_BWHC()
            return (output,)
    ```

## PurgeVRAM

Cleans up VRAM by purging caches and/or unloading models

### Inputs

| Group    | Name         | Type                                  | Default | Extras |
| -------- | ------------ | ------------------------------------- | ------- | ------ |
| required | anything     | `<ast.Name object at 0x7f0a337c2530>` |         |        |
| required | purge_cache  | `BOOLEAN`                             | True    |        |
| required | purge_models | `BOOLEAN`                             | True    |        |

??? note "Pick the code in utils.py"

    ```python
    class PurgeVRAM:
        """Cleans up VRAM by purging caches and/or unloading models.

        A utility node that helps manage VRAM usage by:
        - Clearing Python garbage collector
        - Emptying PyTorch CUDA cache
        - Optionally unloading all models
        - Optionally purging system cache

        Args:
            anything: Any input value (unused, allows connection in workflow)
            purge_cache (bool): Whether to purge system and CUDA cache
            purge_models (bool): Whether to unload all models from memory
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "anything": (any_type, {}),
                    "purge_cache": ("BOOLEAN", {"default": True}),
                    "purge_models": ("BOOLEAN", {"default": True}),
                },
                "optional": {},
            }

        RETURN_TYPES = ()
        FUNCTION = "execute"
        CATEGORY = UTILS_CAT
        OUTPUT_NODE = True

        def execute(self, anything, purge_cache, purge_models):

            if purge_cache:

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            if purge_models:
                mm.unload_all_models()
                mm.soft_empty_cache(True)
            return (None,)
    ```
