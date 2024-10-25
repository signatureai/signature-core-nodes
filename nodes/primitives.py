from .categories import PRIMITIVES_CAT


class Float:
    """Represents a floating-point number input.

    This class provides a node for handling floating-point number inputs with
    specified default, minimum, maximum, and step values.

    Methods:
        execute(value): Returns the input floating-point value as a tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = PRIMITIVES_CAT

    def execute(self, value):
        return (value,)


class Int:
    """Represents an integer input.

    This class provides a node for handling integer inputs with specified default, minimum, maximum, and step values.

    Methods:
        execute(value): Returns the input integer value as a tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": (
                    "INT",
                    {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = PRIMITIVES_CAT

    def execute(self, value):
        return (value,)


class String:
    """Represents a single-line string input.

    This class provides a node for handling single-line string inputs with a specified default value.

    Methods:
        execute(value): Returns the input string value as a tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = PRIMITIVES_CAT

    def execute(self, value):
        return (value,)


class StringMultiline:
    """Represents a multi-line string input.

    This class provides a node for handling multi-line string inputs with a specified default value.

    Methods:
        execute(value): Returns the input multi-line string value as a tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "execute"
    CATEGORY = PRIMITIVES_CAT

    def execute(self, value):
        return (value,)


class Boolean:
    """Represents a boolean input.

    This class provides a node for handling boolean inputs with a specified default value.

    Methods:
        execute(value): Returns the input boolean value as a tuple.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = PRIMITIVES_CAT

    def execute(self, value):
        return (value,)
