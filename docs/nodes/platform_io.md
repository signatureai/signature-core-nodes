# Platform Io Nodes

## PlatformInputImage

Handles image input for the platform

### Inputs

| Group    | Name          | Type                                  | Default     | Extras         |
| -------- | ------------- | ------------------------------------- | ----------- | -------------- |
| required | title         | `STRING`                              | Input Image |                |
| required | subtype       | `LIST`                                |             |                |
| required | required      | `BOOLEAN`                             | True        |                |
| required | include_alpha | `BOOLEAN`                             | False       |                |
| required | multiple      | `BOOLEAN`                             | False       |                |
| required | value         | `STRING`                              |             |                |
| required | metadata      | `STRING`                              | {}          | multiline=True |
| optional | fallback      | `<ast.Name object at 0x7f0a337f9bd0>` |             |                |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformInputImage:
        """Handles image input for the platform.

        This class processes image inputs, supporting both single and multiple images.
        It can handle images from URLs or base64 strings and apply post-processing like alpha channel removal.

        Methods:
            execute(**kwargs): Processes the image input and returns a list of processed images.

        Raises:
            ValueError: If input values are not of the expected types or if no valid input is found.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Input Image"}),
                    "subtype": (["image", "mask"],),
                    "required": ("BOOLEAN", {"default": True}),
                    "include_alpha": ("BOOLEAN", {"default": False}),
                    "multiple": ("BOOLEAN", {"default": False}),
                    "value": ("STRING", {"default": ""}),
                    "metadata": ("STRING", {"default": "{}", "multiline": True}),
                },
                "optional": {
                    "fallback": (any_type,),
                },
            }

        RETURN_TYPES = (any_type,)
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT
        OUTPUT_IS_LIST = (True,)

        def execute(
            self,
            **kwargs,
        ):
            def post_process(output: TensorImage, include_alpha: bool) -> TensorImage:
                if include_alpha is False and output.shape[1] == 4:
                    # get alpha
                    rgb = TensorImage(output[:, :3, :, :])
                    alpha = TensorImage(output[:, -1, :, :])
                    output, _ = cutout(rgb, alpha)
                return output

            value = kwargs.get("value")
            if not isinstance(value, str):
                raise ValueError("Value must be a string")
            subtype = kwargs.get("subtype")
            if not isinstance(subtype, str):
                raise ValueError("Subtype must be a string")
            include_alpha = kwargs.get("include_alpha") or False
            if not isinstance(include_alpha, bool):
                raise ValueError("Include alpha must be a boolean")
            multiple = kwargs.get("multiple") or False
            if not isinstance(multiple, bool):
                raise ValueError("Multiple must be a boolean")
            fallback = kwargs.get("fallback")

            if "," in value:
                splited_value = value.split(",")
                value = splited_value if multiple else splited_value[0]
            else:
                value = [value] if value != "" else []
            outputs: list[TensorImage | torch.Tensor] = []
            for i, _ in enumerate(value):
                item = value[i]
                if isinstance(item, str):
                    if item != "":
                        if item.startswith("http"):
                            output = TensorImage.from_web(item)
                        else:
                            try:
                                output = TensorImage.from_base64(item)
                            except:
                                raise ValueError(f"Unsupported input format: {item}")
                        outputs.append(output)
            if len(outputs) == 0:
                if fallback is None:
                    raise ValueError("No input found")
                tensor_fallback = TensorImage.from_BWHC(fallback)
                outputs.append(tensor_fallback)
            for i, _ in enumerate(outputs):
                output = outputs[i]
                if isinstance(output, torch.Tensor):
                    output = TensorImage(output)
                if subtype == "mask":
                    outputs[i] = output.get_grayscale().get_BWHC()
                else:
                    if isinstance(output, TensorImage):
                        outputs[i] = post_process(output, include_alpha).get_BWHC()
            return (outputs,)
    ```

## PlatformInputConnector

Handles input from external connectors like Google Drive

### Inputs

| Group    | Name      | Type      | Default         | Extras         |
| -------- | --------- | --------- | --------------- | -------------- |
| required | title     | `STRING`  | Input Connector |                |
| required | subtype   | `LIST`    |                 |                |
| required | required  | `BOOLEAN` | True            |                |
| required | override  | `BOOLEAN` | False           |                |
| required | token     | `STRING`  |                 |                |
| required | mime_type | `STRING`  | image/png       |                |
| required | value     | `STRING`  |                 |                |
| required | metadata  | `STRING`  | {}              | multiline=True |

### Returns

| Name | Type   |
| ---- | ------ |
| file | `FILE` |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformInputConnector:
        """Handles input from external connectors like Google Drive.

        This class manages file downloads from external services using provided tokens and file IDs.

        Methods:
            execute(**kwargs): Downloads the specified file and returns the file data.

        Raises:
            ValueError: If input values are not of the expected types.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Input Connector"}),
                    "subtype": (["google_drive"],),
                    "required": ("BOOLEAN", {"default": True}),
                    "override": ("BOOLEAN", {"default": False}),
                    "token": ("STRING", {"default": ""}),
                    "mime_type": ("STRING", {"default": "image/png"}),
                    "value": ("STRING", {"default": ""}),
                    "metadata": ("STRING", {"default": "{}", "multiline": True}),
                },
            }

        RETURN_TYPES = ("FILE",)
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT

        def execute(
            self,
            **kwargs,
        ):
            value = kwargs.get("value")
            if not isinstance(value, str):
                raise ValueError("Value must be a string")
            token = kwargs.get("token")
            if not isinstance(token, str):
                raise ValueError("Token must be a string")
            mime_type = kwargs.get("mime_type")
            if not isinstance(mime_type, str):
                raise ValueError("Mime type must be a string")
            override = kwargs.get("override")
            if not isinstance(override, bool):
                raise ValueError("Override must be a boolean")
            connector = GoogleConnector(token=token)
            input_folder = os.path.join(BASE_COMFY_DIR, "input")
            data = connector.download(
                file_id=value, mime_type=mime_type, output_path=input_folder, override=override
            )
            return (data,)
    ```

## PlatformInputText

Handles text input for the platform

### Inputs

| Group    | Name     | Type      | Default    | Extras          |
| -------- | -------- | --------- | ---------- | --------------- |
| required | title    | `STRING`  | Input Text |                 |
| required | subtype  | `LIST`    |            |                 |
| required | required | `BOOLEAN` | True       |                 |
| required | value    | `STRING`  |            | multiline=True  |
| required | metadata | `STRING`  | {}         | multiline=True  |
| optional | fallback | `STRING`  |            | forceInput=True |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformInputText:
        """Handles text input for the platform.

        This class processes text inputs, providing a fallback option if the input is empty.

        Methods:
            execute(**kwargs): Returns the input text or the fallback if the input is empty.

        Raises:
            ValueError: If input values are not of the expected types.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Input Text"}),
                    "subtype": (["string", "positive_prompt", "negative_prompt"],),
                    "required": ("BOOLEAN", {"default": True}),
                    "value": ("STRING", {"multiline": True, "default": ""}),
                    "metadata": ("STRING", {"default": "{}", "multiline": True}),
                },
                "optional": {
                    "fallback": ("STRING", {"forceInput": True}),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT

        def execute(self, **kwargs):
            value = kwargs.get("value")
            if not isinstance(value, str):
                raise ValueError("Value must be a string")
            fallback = kwargs.get("fallback")
            if not isinstance(fallback, str):
                raise ValueError("Fallback must be a string")
            if value == "":
                value = fallback or ""
            return (value,)
    ```

## PlatformInputNumber

Handles numeric input for the platform

### Inputs

| Group    | Name     | Type      | Default      | Extras         |
| -------- | -------- | --------- | ------------ | -------------- |
| required | title    | `STRING`  | Input Number |                |
| required | subtype  | `LIST`    |              |                |
| required | required | `BOOLEAN` | True         |                |
| required | value    | `FLOAT`   | 0            |                |
| required | metadata | `STRING`  | {}           | multiline=True |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformInputNumber:
        """Handles numeric input for the platform.

        This class processes numeric inputs, supporting both integers and floats.

        Methods:
            execute(**kwargs): Returns the input number, converting it to the specified subtype.

        Raises:
            ValueError: If input values are not of the expected types.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Input Number"}),
                    "subtype": (["float", "int"],),
                    "required": ("BOOLEAN", {"default": True}),
                    "value": ("FLOAT", {"default": 0}),
                    "metadata": ("STRING", {"default": "{}", "multiline": True}),
                },
            }

        RETURN_TYPES = (any_type,)
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT

        def execute(self, **kwargs):
            value = kwargs.get("value")
            if not isinstance(value, int) and not isinstance(value, float):
                raise ValueError("Value must be a string")
            subtype = kwargs.get("subtype")
            if not isinstance(subtype, str):
                raise ValueError("Subtype must be a string")
            if subtype == "int":
                value = int(value)
            else:
                value = float(value)
            return (value,)
    ```

## PlatformInputBoolean

Handles boolean input for the platform

### Inputs

| Group    | Name     | Type      | Default       | Extras         |
| -------- | -------- | --------- | ------------- | -------------- |
| required | title    | `STRING`  | Input Boolean |                |
| required | subtype  | `LIST`    |               |                |
| required | required | `BOOLEAN` | True          |                |
| required | value    | `BOOLEAN` | False         |                |
| required | metadata | `STRING`  | {}            | multiline=True |

### Returns

| Name    | Type      |
| ------- | --------- |
| boolean | `BOOLEAN` |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformInputBoolean:
        """Handles boolean input for the platform.

        This class processes boolean inputs.

        Methods:
            execute(**kwargs): Returns the input boolean value.

        Raises:
            ValueError: If input values are not of the expected types.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Input Boolean"}),
                    "subtype": (["boolean"],),
                    "required": ("BOOLEAN", {"default": True}),
                    "value": ("BOOLEAN", {"default": False}),
                    "metadata": ("STRING", {"default": "{}", "multiline": True}),
                }
            }

        RETURN_TYPES = ("BOOLEAN",)
        RETURN_NAMES = ("boolean",)
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT

        def execute(self, **kwargs):
            value = kwargs.get("value")
            if not isinstance(value, bool):
                raise ValueError("Value must be a boolean")
            return (value,)
    ```

## PlatformOutput

Handles output for the platform

### Inputs

| Group    | Name        | Type                                  | Default      | Extras         |
| -------- | ----------- | ------------------------------------- | ------------ | -------------- |
| required | title       | `STRING`                              | Output Image |                |
| required | subtype     | `LIST`                                |              |                |
| required | metadata    | `STRING`                              |              | multiline=True |
| required | value       | `<ast.Name object at 0x7f0a337c3040>` |              |                |
| hidden   | output_path | `STRING`                              | output       |                |

??? note "Pick the code in platform_io.py"

    ```python
    class PlatformOutput:
        """Handles output for the platform.

        This class manages the output of various data types, including images, numbers, and strings.
        It supports saving images and generating thumbnails.

        Methods:
            execute(**kwargs): Processes and saves the output data, returning metadata about the saved files.

        Raises:
            ValueError: If input values are not of the expected types or if unsupported output types are provided.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "title": ("STRING", {"default": "Output Image"}),
                    "subtype": (["image", "mask", "int", "float", "string", "dict"],),
                    "metadata": ("STRING", {"default": "", "multiline": True}),
                    "value": (any_type,),
                },
                "hidden": {
                    "output_path": ("STRING", {"default": "output"}),
                },
            }

        RETURN_TYPES = ()
        OUTPUT_NODE = True
        INPUT_IS_LIST = True
        FUNCTION = "execute"
        CATEGORY = PLATFORM_IO_CAT

        def __save_outputs(self, **kwargs) -> dict | None:
            img = kwargs.get("img")
            if not isinstance(img, (torch.Tensor, TensorImage)):
                raise ValueError("Image must be a tensor or TensorImage")

            title = kwargs.get("title", "")
            if not isinstance(title, str):
                title = str(title)

            subtype = kwargs.get("subtype", "image")
            if not isinstance(subtype, str):
                subtype = str(subtype)

            thumbnail_size = kwargs.get("thumbnail_size", 1024)
            if not isinstance(thumbnail_size, int):
                try:
                    thumbnail_size = int(thumbnail_size)
                except (ValueError, TypeError):
                    thumbnail_size = 1024

            output_dir = kwargs.get("output_dir", "output")
            if not isinstance(output_dir, str):
                output_dir = str(output_dir)

            metadata = kwargs.get("metadata", "")
            if not isinstance(metadata, str):
                metadata = str(metadata)

            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"signature_{current_time_str}_{uuid7str()}.png"
            save_path = os.path.join(output_dir, file_name)
            if os.path.exists(save_path):
                file_name = f"signature_{current_time_str}_{uuid7str()}_{uuid7str()}.png"
                save_path = os.path.join(output_dir, file_name)

            output_img = img if isinstance(img, TensorImage) else TensorImage(img)

            thumbnail_img = output_img.get_resized(thumbnail_size)
            thumbnail_path = save_path.replace(".png", "_thumbnail.jpeg")
            thumbnail_file_name = file_name.replace(".png", "_thumbnail.jpeg")
            thumbnail_saved = thumbnail_img.save(thumbnail_path)

            image_saved = output_img.save(save_path)

            if image_saved and thumbnail_saved:
                return {
                    "title": title,
                    "type": subtype,
                    "metadata": metadata,
                    "value": file_name,
                    "thumbnail": thumbnail_file_name if thumbnail_saved else None,
                }

            return None

        def execute(self, **kwargs):
            title_list = kwargs.get("title")
            if not isinstance(title_list, list):
                raise ValueError("Title must be a list")
            metadata_list = kwargs.get("metadata")
            if not isinstance(metadata_list, list):
                raise ValueError("Metadata must be a list")
            subtype_list = kwargs.get("subtype")
            if not isinstance(subtype_list, list):
                raise ValueError("Subtype must be a list")
            output_path_list = kwargs.get("output_path")
            print(f"output_path_list: {output_path_list} {type(output_path_list)}")
            if not isinstance(output_path_list, list):
                output_path_list = ["output"] * len(title_list)
            value_list = kwargs.get("value")
            if not isinstance(value_list, list):
                raise ValueError("Value must be a list")
            main_subtype = subtype_list[0]
            supported_types = ["image", "mask", "int", "float", "string", "dict"]
            if main_subtype not in supported_types:
                raise ValueError(f"Unsupported output type: {main_subtype}")

            results = []
            thumbnail_size = 1024
            for idx, item in enumerate(value_list):
                title = title_list[idx]
                metadata = metadata_list[idx]
                output_dir = os.path.join(BASE_COMFY_DIR, output_path_list[idx])
                if isinstance(item, torch.Tensor):
                    if main_subtype in ["image", "mask"]:
                        tensor_images = TensorImage.from_BWHC(item.to("cpu"))
                        for img in tensor_images:
                            result = self.__save_outputs(
                                img=img,
                                title=title,
                                subtype=main_subtype,
                                thumbnail_size=thumbnail_size,
                                output_dir=output_dir,
                                metadata=metadata,
                            )
                            if result:
                                results.append(result)
                    else:
                        raise ValueError(f"Unsupported output type: {type(item)}")
                else:
                    value_json = json.dumps(item) if main_subtype == "dict" else item
                    results.append(
                        {"title": title, "type": main_subtype, "metadata": metadata, "value": value_json}
                    )
            return {"ui": {"signature_output": results}}
    ```
