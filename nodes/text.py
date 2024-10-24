import re

import torch

from .categories import TEXT_CAT
from .shared import any_type


class TextPreview:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "text": (any_type,),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text = kwargs.get("text", [])
        text_string = ""
        for t in text:
            if t is None:
                continue
            if text_string != "":
                text_string += "\n"
            text_string += str(t.shape) if isinstance(t, torch.Tensor) else str(t)
        return {"ui": {"text": [text_string]}, "result": (text_string,)}


class TextCase:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "case": (["lower", "upper", "capitalize", "title"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text = kwargs.get("text") or ""
        case = kwargs.get("case") or "lower"
        result = text
        if case == "lower":
            result = text.lower()
        if case == "upper":
            result = text.upper()
        if case == "capitalize":
            result = text.capitalize()
        if case == "title":
            result = text.title()
        return (result,)


class TextTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "trim_type": (["both", "left", "right"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text = kwargs.get("text") or ""
        trim_type = kwargs.get("trim_type") or "both"
        if trim_type == "both":
            return (text.strip(),)
        if trim_type == "left":
            return (text.lstrip(),)
        if trim_type == "right":
            return (text.rstrip(),)
        return (text,)


class TextSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "delimiter": ("STRING", {"default": " "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT
    OUTPUT_IS_LIST = (True,)

    def execute(self, **kwargs):
        text = kwargs.get("text", "")
        delimiter = kwargs.get("delimiter", " ")
        return (text.split(delimiter),)


class TextRegexReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "pattern": ("STRING", {"default": ""}),
                "replacement": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text = kwargs.get("text", "")
        pattern = kwargs.get("pattern", "")
        replacement = kwargs.get("replacement", "")
        return (re.sub(pattern, replacement, text),)


class TextFindReplace:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "find": ("STRING", {"default": ""}),
                "replace": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text = kwargs.get("text") or ""
        find = kwargs.get("find") or ""
        replace = kwargs.get("replace") or ""
        return (text.replace(find, replace),)


class TextConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": ""}),
                "text2": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = TEXT_CAT

    def execute(self, **kwargs):
        text1 = kwargs.get("text1", "")
        text2 = kwargs.get("text2", "")
        return (text1 + text2,)
