import json
import os

import torch
from signature_core.img.tensor_image import TensorImage

from .categories import FILE_CAT
from .shared import BASE_COMFY_DIR


def image_array_to_tensor(x: TensorImage):
    image = x.get_BWHC()
    mask = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=torch.float32)
    if x.shape[1] == 4:
        mask = image[:, :, :, -1]
    return (
        image,
        mask,
    )


class ImageFromWeb:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = FILE_CAT

    def execute(self, **kwargs):
        url = kwargs.get("url")
        if not isinstance(url, str):
            raise ValueError("URL must be a string")
        img_arr = TensorImage.from_web(url)
        return image_array_to_tensor(img_arr)


class ImageFromBase64:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"base64": ("STRING", {"default": "BASE64 HERE"})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = FILE_CAT

    def execute(self, **kwargs):
        base64 = kwargs.get("base64")
        if not isinstance(base64, str):
            raise ValueError("Base64 must be a string")
        img_arr = TensorImage.from_base64(base64)
        return image_array_to_tensor(img_arr)


class Base64FromImage:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = FILE_CAT
    OUTPUT_NODE = True

    def execute(self, **kwargs):
        image = kwargs.get("image")
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        images = TensorImage.from_BWHC(image)
        output = images.get_base64()
        return (output,)


class FileLoader:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "execute"
    CATEGORY = FILE_CAT

    def execute(self, **kwargs):
        value = kwargs.get("value")
        if not isinstance(value, str):
            raise ValueError("Value must be a string")
        data = value.split("&&") if "&&" in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i, _ in enumerate(data):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get("name", None)
                if name is None:
                    continue
                item["name"] = os.path.join(input_folder, name)
                data[i] = item

        return (data,)


class FolderLoader:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "execute"
    CATEGORY = FILE_CAT

    def execute(self, **kwargs):
        value = kwargs.get("value")
        if not isinstance(value, str):
            raise ValueError("Value must be a string")
        data = value.split("&&") if "&&" in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i, _ in enumerate(data):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get("name", None)
                if name is None:
                    continue
                item["name"] = os.path.join(input_folder, name)
                data[i] = item
        return (data,)


class File2ImageList:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "files": ("FILE", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = FILE_CAT
    OUTPUT_IS_LIST = (True,)

    def execute(self, **kwargs):
        files = kwargs.get("files")
        if not isinstance(files, list):
            raise ValueError("Files must be a list")
        images_list = []
        for file in files:
            mimetype = file["type"]
            extension = file["name"].lower().split(".")[-1]
            possible_extensions = ["png", "jpg", "jpeg", "tiff", "tif", "bmp"]
            if mimetype.startswith("image") and extension in possible_extensions:
                images_list.append(TensorImage.from_local(file["name"]).get_BWHC())

        return (images_list,)


class File2List:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "files": ("FILE", {"default": ""}),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "execute"
    CATEGORY = FILE_CAT

    def execute(self, **kwargs):
        files = kwargs.get("files")
        if not isinstance(files, list):
            raise ValueError("Files must be a list")
        return (files,)
