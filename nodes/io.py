import torch
import os
import json
from .categories import IO_CAT
from .shared import BASE_COMFY_DIR
from signature_core.img.tensor_image import TensorImage


def image_array_to_tensor(x: TensorImage):
    image = x.get_BWHC()
    mask = torch.ones((x.shape[0],
                       1,
                       x.shape[2],
                       x.shape[3]),
                      dtype=torch.float32)
    if x.shape[1] == 4:
        mask = image[:, :, :, -1]
    return (image, mask, )

class ImageFromWeb():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, url: str):
        img_arr = TensorImage.from_web(url)
        return image_array_to_tensor(img_arr)

class ImageFromBase64():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"base64": ("STRING", {"default": "BASE64 HERE"})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, base64: str):
        img_arr = TensorImage.from_base64(base64)
        return image_array_to_tensor(img_arr)

class Base64FromImage():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = IO_CAT
    OUTPUT_NODE = True

    def process(self, image):
        images = TensorImage.from_BWHC(image)
        output = images.get_base64()
        return (output,)

class LoadFile():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": ('STRING', {'default': ''}),
            },
    }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, value: str):
        data = value.split('&&') if '&&' in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i in range(len(data)):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get('name', None)
                if name is None:
                    continue
                item['name'] = os.path.join(input_folder, name)
                data[i] = item

        return (data,)

class LoadFolder():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": ('STRING', {'default': ''}),
            },
    }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, value: str):
        data = value.split('&&') if '&&' in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i in range(len(data)):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get('name', None)
                if name is None:
                    continue
                item['name'] = os.path.join(input_folder, name)
                data[i] = item
        return (data,)


NODE_CLASS_MAPPINGS = {
    "signature_image_from_web": ImageFromWeb,
    "signature_image_from_base64": ImageFromBase64,
    "signature_base64_from_image": Base64FromImage,
    "signature_load_file": LoadFile,
    "signature_load_folder": LoadFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_image_from_web": "SIG Image from Web",
    "signature_image_from_base64": "SIG Image from Base64",
    "signature_base64_from_image": "SIG Base64 from Image",
    "signature_load_file": "SIG Load File",
    "signature_load_folder": "SIG Load Folder",

}