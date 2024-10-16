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
    FUNCTION = "process"
    CATEGORY = DATA_CAT

    def process(self, **kwargs):
        json_str = kwargs.get("json_str")
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
    FUNCTION = "process"
    CATEGORY = DATA_CAT

    def process(self, **kwargs):
        json_dict = kwargs.get("dict")
        json_str = json.dumps(json_dict)
        return (json_str,)


class GetListValue:
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
    FUNCTION = "process"
    CATEGORY = DATA_CAT

    def process(self, **kwargs):
        list_obj = kwargs.get("list")
        index = kwargs.get("index")
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
    FUNCTION = "process"
    CATEGORY = DATA_CAT

    def process(self, **kwargs):
        dict_obj = kwargs.get("dict")
        key = kwargs.get("key")
        value = dict_obj.get(key)
        value_type = type(value).__name__
        return (value, value_type)


NODE_CLASS_MAPPINGS = {
    "signature_json_to_dict": JsonToDict,
    "signature_dict_to_json": DictToJson,
    "signature_get_dict_value": GetDictValue,
    "signature_get_list_value": GetListValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_json_to_dict": "SIG Json2Dict",
    "signature_dict_to_json": "SIG Dict2Json",
    "signature_get_dict_value": "SIG Get Dict Value",
    "signature_get_list_value": "SIG Get List Value",
}
