from .categories import PRIMITIVES


class Float:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "process"
    CATEGORY = PRIMITIVES

    def process(self, value):
        return (value,)


class Int:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "process"
    CATEGORY = PRIMITIVES

    def process(self, value):
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
    FUNCTION = "process"
    CATEGORY = PRIMITIVES

    def process(self, value):
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
    FUNCTION = "process"
    CATEGORY = PRIMITIVES

    def process(self, value):
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
    FUNCTION = "process"
    CATEGORY = PRIMITIVES

    def process(self, value):
        return (value,)


NODE_CLASS_MAPPINGS = {
    "signature_float": Float,
    "signature_int": Int,
    "signature_string": String,
    "signature_string_multiline": StringMultiline,
    "signature_boolean": Boolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_float": "SIG Float",
    "signature_int": "SIG Int",
    "signature_string": "SIG String",
    "signature_string_multiline": "SIG String Multiline",
    "signature_boolean": "SIG Boolean",
}
