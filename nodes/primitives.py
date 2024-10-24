from .categories import PRIMITIVES_CAT


class Float:
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
