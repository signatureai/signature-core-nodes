import json
import os
from datetime import datetime

import torch
from signature_core.connectors.google_connector import GoogleConnector
from signature_core.functional.transform import cutout
from signature_core.img.tensor_image import TensorImage

from .categories import PLATFROM_IO_CAT
from .shared import BASE_COMFY_DIR, any_type

from uuid_extensions import uuid7str


class PlatformInputImage:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Image"}),
                "subtype": (["image", "mask"],),
                "required": ("BOOLEAN", {"default": True}),
                "include_alpha": ("BOOLEAN", {"default": False}),
                "value": ("STRING", {"default": ""}),
                "metadata": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "fallback": (any_type,),
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT
    OUTPUT_IS_LIST = (True,)

    def apply(
        self,
        value,
        title: str,
        metadata: str,
        subtype: str,
        required: bool,
        include_alpha: bool,
        fallback=None,
    ):
        def post_process(output: TensorImage, include_alpha: bool) -> TensorImage:
            if include_alpha is False and output.shape[1] == 4:
                # get alpha
                rgb = TensorImage(output[:, :3, :, :])
                alpha = TensorImage(output[:, -1, :, :])
                output, _ = cutout(rgb, alpha)
            return output

        if "," in value:
            value = value.split(",")
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


class PlatformInputConnector:
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
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(
        self,
        value: str,
        token: str,
        mime_type: str,
        override: bool,
        title: str,
        metadata: str,
        subtype: str,
        required: bool,
    ):
        connector = GoogleConnector(token=token)
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        data = connector.download(
            file_id=value, mime_type=mime_type, output_path=input_folder, override=override
        )
        return (data,)


class PlatformInputText:
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
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: str, title: str, metadata: str, subtype: str, required: bool, fallback=None):
        if value == "":
            value = fallback or ""
        return (value,)


class PlatformInputNumber:
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
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: float, title: str, metadata: str, subtype: str, required: bool):
        if subtype == "int":
            value = int(value)
        else:
            value = float(value)
        return (value,)


class PlatformInputBoolean:
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
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value: bool, title: str, subtype: str, metadata: str, required: bool):
        return (value,)


class PlatformOutput:
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
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def __save_outputs(
        self, img, title: str, subtype: str, thumbnail_size: int, output_dir: str, metadata: str = ""
    ) -> dict | None:
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"signature_{current_time_str}_{uuid7str()}.png"
        save_path = os.path.join(output_dir, file_name)
        if os.path.exists(save_path):
            file_name = f"signature_{current_time_str}_{uuid7str()}_{uuid7str()}.png"
            save_path = os.path.join(output_dir, file_name)

        output_img = TensorImage(img)

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

    def apply(self, value, title: str, subtype: str, metadata: str = "", output_path: str = ["output"]):
        if len(subtype) == 0 or len(value) == 0:
            raise ValueError("No input found")
        main_subtype = subtype[0]
        supported_types = ["image", "mask", "int", "float", "string", "dict"]
        if main_subtype not in supported_types:
            raise ValueError(f"Unsupported output type: {subtype}")
        # ComfyUI passes output_path as a list instead of a string because of INPUT_IS_LIST=True
        output_dir = os.path.join(BASE_COMFY_DIR, output_path[0])
        results = []
        thumbnail_size = 1024
        for item in value:
            if isinstance(item, torch.Tensor):
                if main_subtype in ["image", "mask"]:
                    tensor_images = TensorImage.from_BWHC(item.to("cpu"))
                    for img in tensor_images:
                        # console.log(f"Input tensor shape {img.shape}")
                        result = self.__save_outputs(
                            img, title, main_subtype, thumbnail_size, output_dir, metadata
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


NODE_CLASS_MAPPINGS = {
    "signature_input_image": PlatformInputImage,
    "signature_input_text": PlatformInputText,
    "signature_input_number": PlatformInputNumber,
    "signature_input_boolean": PlatformInputBoolean,
    "signature_input_connector": PlatformInputConnector,
    "signature_output": PlatformOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_input_image": "SIG Input Image",
    "signature_input_text": "SIG Input Text",
    "signature_input_number": "SIG Input Number",
    "signature_input_boolean": "SIG Input Boolean",
    "signature_input_connector": "SIG Input Connector",
    "signature_output": "SIG Output",
}
