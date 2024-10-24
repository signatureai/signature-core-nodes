import json

from .categories import DATA_CAT
from .shared import any_type


class JsonToDict:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "json_str": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("DICT",)
    FUNCTION = "execute"
    CATEGORY = DATA_CAT

    def execute(self, **kwargs):
        json_str = kwargs.get("json_str")
        if not isinstance(json_str, str):
            raise ValueError("Json string must be a string")
        json_dict = json.loads(json_str)
        return (json_dict,)


class DictToJson:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "dict": ("DICT",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = DATA_CAT

    def execute(self, **kwargs):
        json_dict = kwargs.get("dict")
        json_str = json.dumps(json_dict)
        return (json_str,)


class GetImageListItem:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "images": ("IMAGE",),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = "IMAGE"
    RETURN_NAMES = "image"
    FUNCTION = "execute"
    CATEGORY = DATA_CAT

    def execute(self, **kwargs):
        images = kwargs.get("images")
        index = kwargs.get("index")
        if not isinstance(index, int):
            raise ValueError("Index must be an integer")
        if not isinstance(images, list):
            raise ValueError("Images must be a list")
        images = images[index]
        index = kwargs.get("index")
        image = images[index]
        return (image,)


class GetListItem:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "list": ("LIST",),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("item", "value_type")
    FUNCTION = "execute"
    CATEGORY = DATA_CAT

    def execute(self, **kwargs):
        list_obj = kwargs.get("list")
        index = kwargs.get("index")
        if not isinstance(index, int):
            raise ValueError("Index must be an integer")
        if not isinstance(list_obj, list):
            raise ValueError("List must be a list")
        item = list_obj[index]
        item_type = type(item).__name__
        return (item, item_type)


class GetDictValue:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "dict": ("DICT",),
                "key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (any_type, "STRING")
    RETURN_NAMES = ("value", "value_type")
    FUNCTION = "execute"
    CATEGORY = DATA_CAT

    def execute(self, **kwargs):
        dict_obj = kwargs.get("dict")
        key = kwargs.get("key")
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not isinstance(dict_obj, dict):
            raise ValueError("Dict must be a dictionary")
        value = dict_obj.get(key)
        value_type = type(value).__name__
        return (value, value_type)
