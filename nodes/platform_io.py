
from .categories import PLATFROM_IO_CAT
from .shared import BASE_COMFY_DIR, any
from signature_core.img.tensor_image import TensorImage
import torch
import os
from datetime import datetime
import json


class PlatformInputImage():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Image"}),
                "subtype": (['image', 'mask'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("STRING", {"default": ""}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                },
                "optional": {"fallback": (any,),}
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value, title:str, metadata:str, subtype: str, required:str, fallback = None):

        if value != "":
            if value.startswith("data:"):
                output = TensorImage.from_base64(value)
            elif value.startswith("http"):
                output = TensorImage.from_web(value)
            else:
                raise ValueError(f"Unsupported input type: {type(value)}")
            if subtype == "mask":
                output = output.get_grayscale()
            else:
                output = output.get_rgb_or_rgba()
            return (output.get_BWHC(),)

        if isinstance(fallback, torch.Tensor):
            return (fallback,)

        raise ValueError(f"Unsupported fallback type: {type(fallback)}")

class PlatformInputText():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Text"}),
                "subtype": (['string','positive_prompt', 'negative_prompt'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("STRING", {"multiline": True, "default": ""}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                },
            }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value:str, title:str, metadata:str, subtype: str, required:str):

        if isinstance(value, str):
            return (value,)
        else:
            raise ValueError(f"Unsupported input type: {type(value)}")


class PlatformInputNumber():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Number"}),
                "subtype": (['float','int'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("FLOAT", {"default": 0}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                },
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value:float, title:str, metadata:str, subtype: str, required:str):
        if subtype == "int":
            value = int(value)
        else:
            value = float(value)
        return (value,)


class PlatformInputSlider():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Slider"}),
                "subtype": (['float','int'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("FLOAT", {"default": 0}),
                "min_value": ("FLOAT", {"default": 0}),
                "max_value": ("FLOAT", {"default": 10}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                },
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self,
              value: float,
              min_value: float,
              max_value: float,
              title: str,
              metadata: str,
              subtype: str,
              required: str):
        if subtype == "int":
            value = max(min(int(max_value), int(value)), int(min_value))
        else:
            value = max(min(max_value, value), min_value)
        return (value,)


class PlatformOutput():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Output Image"}),
                "subtype": (['image', 'mask', 'int', 'float', 'string', 'dict'],),
                "metadata": ("STRING", {"default": "", "multiline": True}),
                "value": (any,),
                },
            }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value, title: str, subtype: str, metadata: str = ''):
        supported_types = ["image", "mask", "int", "float", "string", "dict"]
        if subtype not in supported_types:
            raise ValueError(f"Unsupported output type: {subtype}")

        output_dir = os.path.join(BASE_COMFY_DIR, 'output')
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = []
        thumbnail_size = 768
        if subtype in ["image", "mask"]:
            tensor_images = TensorImage.from_BWHC(value.to('cpu'))
            for img in tensor_images:
                random_str = str(torch.randint(0, 100000, (1,)).item())
                file_name = f"signature_{current_time_str}_{random_str}.png"
                save_path = os.path.join(output_dir, file_name)

                output_img = TensorImage(img)
                width, height = output_img.size()[-2:]

                # Resize only if either dimension is greater than 768
                if width > thumbnail_size or height > thumbnail_size:
                    thumbnail_img = output_img.get_resized(thumbnail_size)
                    thumbnail_path = save_path.replace(".png", "_thumbnail.jpeg")
                    thumbnail_saved = thumbnail_img.save(thumbnail_path)
                else:
                    thumbnail_path = save_path
                    thumbnail_saved = True

                image_saved = output_img.save(save_path)

                if image_saved and thumbnail_saved:
                    results.append({
                        "title": title,
                        "type": subtype,
                        "metadata": metadata,
                        "value": file_name,
                        "thumbnail": thumbnail_path if thumbnail_saved else None
                    })
        else:
            value_json = json.dumps(value) if subtype == "dict" else value
            results.append({
                "title": title,
                "type": subtype,
                "metadata": metadata,
                "value": value_json
            })

        return {"ui": {"signature_output": results}}


NODE_CLASS_MAPPINGS = {
    "signature_input_image": PlatformInputImage,
    "signature_input_text": PlatformInputText,
    "signature_input_number": PlatformInputNumber,
    "signature_input_slider": PlatformInputSlider,
    "signature_output": PlatformOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_input_image": "SIG Input Image",
    "signature_input_text": "SIG Input Text",
    "signature_input_number": "SIG Input Number",
    "signature_input_slider": "SIG Input Slider",
    "signature_output": "SIG Output",
}