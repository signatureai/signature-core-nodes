import json

from .categories import DATA_CAT
from .shared import any_type


class JsonToDict:
    """Converts a JSON string to a Python dictionary.

    This class parses a JSON-formatted string and converts it into a Python dictionary.

    Methods:
        execute(**kwargs): Parses the JSON string and returns the resulting dictionary.

    Args:
        json_str (str): The JSON string to be converted.

    Returns:
        tuple: A tuple containing the resulting dictionary.

    Raises:
        ValueError: If the input is not a string.
    """

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
    """Converts a Python dictionary to a JSON string.

    This class serializes a Python dictionary into a JSON-formatted string.

    Methods:
        execute(**kwargs): Serializes the dictionary and returns the resulting JSON string.

    Args:
        dict (dict): The dictionary to be converted.

    Returns:
        tuple: A tuple containing the resulting JSON string.
    """

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
    """Retrieves an image from a list by index.

    This class accesses a list of images and retrieves the image at the specified index.

    Methods:
        execute(**kwargs): Returns the image at the specified index.

    Args:
        images (list): The list of images.
        index (int): The index of the image to retrieve.

    Returns:
        tuple: A tuple containing the retrieved image.

    Raises:
        ValueError: If the index is not an integer or if images is not a list.
    """

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
    """Retrieves an item from a list by index and returns its type.

    This class accesses a list and retrieves the item at the specified index, also returning the item's type.

    Methods:
        execute(**kwargs): Returns the item and its type.

    Args:
        list (list): The list to access.
        index (int): The index of the item to retrieve.

    Returns:
        tuple: A tuple containing the item and its type as a string.

    Raises:
        ValueError: If the index is not an integer or if the list is not a list.
    """

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
            raise ValueError("Input must be a list")
        item = list_obj[index]
        item_type = type(item).__name__
        return (item, item_type)


class GetDictValue:
    """Retrieves a value from a dictionary by key and returns its type.

    This class accesses a dictionary and retrieves the value
    associated with the specified key, also returning the value's type.

    Methods:
        execute(**kwargs): Returns the value and its type.

    Args:
        dict (dict): The dictionary to access.
        key (str): The key of the value to retrieve.

    Returns:
        tuple: A tuple containing the value and its type as a string.

    Raises:
        ValueError: If the key is not a string or if the dict is not a dictionary.
    """

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
