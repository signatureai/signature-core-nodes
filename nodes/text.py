import re

import torch

from .categories import TEXT_CAT
from .shared import any_type


class TextPreview:
    """Processes and generates a preview of text inputs, supporting both strings and tensors.

    This node takes a list of text inputs and generates a formatted preview string. For tensor inputs,
    it includes shape information in the preview. The node is designed to handle multiple input types
    and provide a consistent preview format.

    Args:
        text (Any): A list of text inputs that can be strings, tensors, or other objects that can be
            converted to strings.

    Returns:
        dict: A dictionary containing:
            - ui (dict): UI-specific data with the preview text under the 'text' key
            - result (tuple): A tuple containing the generated preview string

    Notes:
        - Tensor inputs are displayed with their shape information
        - Multiple inputs are separated by newlines
        - None values are skipped in the preview generation
    """

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
    """Transforms text case according to specified formatting rules.

    A utility node that provides various case transformation options for input text, including
    lowercase, uppercase, capitalization, and title case conversion.

    Args:
        text (str): The input text to be transformed. Required.
        case (str): The case transformation to apply. Must be one of:
            - 'lower': Convert text to lowercase
            - 'upper': Convert text to uppercase
            - 'capitalize': Capitalize the first character
            - 'title': Convert text to title case

    Returns:
        tuple[str]: A single-element tuple containing the transformed text.

    Notes:
        - Empty input text will result in an empty string output
        - The transformation preserves any existing spacing and special characters
    """

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
    """Removes whitespace from text according to specified trimming rules.

    A utility node that trims whitespace from text input, offering options to remove whitespace
    from the beginning, end, or both sides of the text.

    Args:
        text (str): The input text to be trimmed. Required.
        trim_type (str): The type of trimming to apply. Must be one of:
            - 'both': Trim whitespace from both ends
            - 'left': Trim whitespace from the start
            - 'right': Trim whitespace from the end

    Returns:
        tuple[str]: A single-element tuple containing the trimmed text.

    Notes:
        - Whitespace includes spaces, tabs, and newlines
        - Empty input text will result in an empty string output
    """

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
    """Splits text into a list of segments using a specified delimiter.

    A utility node that divides input text into multiple segments based on a delimiter,
    creating a list of substrings.

    Args:
        text (str): The input text to be split. Required.
        delimiter (str): The character or string to use as the splitting point. Defaults to space.

    Returns:
        tuple[list[str]]: A single-element tuple containing a list of split text segments.

    Notes:
        - Empty input text will result in a list with one empty string
        - If the delimiter is not found, the result will be a single-element list
        - Consecutive delimiters will result in empty strings in the output list
    """

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
    """Performs pattern-based text replacement using regular expressions.

    A powerful text processing node that uses regex patterns to find and replace text patterns,
    supporting complex pattern matching and replacement operations.

    Args:
        text (str): The input text to process. Required.
        pattern (str): The regular expression pattern to match. Required.
        replacement (str): The string to use as replacement for matched patterns. Required.

    Returns:
        tuple[str]: A single-element tuple containing the processed text.

    Notes:
        - Invalid regex patterns will cause errors
        - Empty pattern or replacement strings are allowed
        - Supports all Python regex syntax including groups and backreferences
    """

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
    """Performs simple text replacement without regex support.

    A straightforward text processing node that replaces all occurrences of a substring with
    another substring, using exact matching.

    Args:
        text (str): The input text to process. Defaults to empty string.
        find (str): The substring to search for. Defaults to empty string.
        replace (str): The substring to replace matches with. Defaults to empty string.

    Returns:
        tuple[str]: A single-element tuple containing the processed text.

    Notes:
        - Case-sensitive matching
        - All occurrences of the 'find' string will be replaced
        - Empty strings for any parameter are handled safely
    """

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
    """Combines two text strings into a single string.

    A basic text manipulation node that joins two input strings together in sequence,
    without any separator between them.

    Args:
        text1 (str): The first text string to concatenate. Defaults to empty string.
        text2 (str): The second text string to concatenate. Defaults to empty string.

    Returns:
        tuple[str]: A single-element tuple containing the concatenated text.

    Notes:
        - No separator is added between the strings
        - Empty strings are handled safely
        - The result will be the direct combination of text1 followed by text2
    """

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
