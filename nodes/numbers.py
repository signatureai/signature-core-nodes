import random

from .categories import NUMBERS_CAT


class IntClamp:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": (
                    "INT",
                    {
                        "default": 0,
                        "forceInput": True,
                        "min": -18446744073709551615,
                        "max": 18446744073709551615,
                    },
                ),
                "min_value": (
                    "INT",
                    {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 1},
                ),
                "max_value": (
                    "INT",
                    {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, number: int, min_value: int, max_value: int):
        if number < min_value:
            return (min_value,)
        if number > max_value:
            return (max_value,)
        return (number,)


class FloatClamp:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": (
                    "FLOAT",
                    {
                        "default": 0,
                        "forceInput": True,
                        "min": -18446744073709551615.0,
                        "max": 18446744073709551615.0,
                    },
                ),
                "min_value": (
                    "FLOAT",
                    {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},
                ),
                "max_value": (
                    "FLOAT",
                    {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, number: float, min_value: float, max_value: float):
        if number < min_value:
            return (min_value,)
        if number > max_value:
            return (max_value,)
        return (number,)


class Float2Int:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": ("FLOAT", {"default": 0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, number: float):
        return (int(number),)


class Int2Float:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": ("INT", {"default": 0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, number: int):
        return (float(number),)


class IntOperator:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "left": ("FLOAT", {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},),
                "right": ("FLOAT", {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},),
                "operator": (["+", "-", "*", "/", "%"],),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, left: int, right: int, operator: str):
        if operator == "+":
            return (left + right,)
        if operator == "-":
            return (left - right,)
        if operator == "*":
            return (left * right,)
        if operator == "/":
            return (left / right,)

        raise ValueError(f"Unsupported operator: {operator}")


class FloatOperator:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "left": (
                    "FLOAT",
                    {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},
                ),
                "right": (
                    "FLOAT",
                    {"default": 0.0, "min": -0184467440737095516150.0, "max": 18446744073709551615.0, "step": 0.01},
                ),
                "operator": (["+", "-", "*", "/", "%"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, left: float, right: float, operator: str):
        if operator == "+":
            return (left + right,)
        if operator == "-":
            return (left - right,)
        if operator == "*":
            return (left * right,)
        if operator == "/":
            return (left / right,)

        raise ValueError(f"Unsupported operator: {operator}")


class IntMinMax:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "a": ("INT", {"default": 0, "forceInput": True}),
                "b": ("INT", {"default": 0, "forceInput": True}),
                "mode": (["min", "max"],),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, a: int, b: int, mode: str):
        if mode == "min":
            return (min(a, b),)
        if mode == "max":
            return (max(a, b),)
        raise ValueError(f"Unsupported mode: {mode}")


class FloatMinMax:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "b": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "mode": (["min", "max"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, a: float, b: float, mode: str):
        if mode == "min":
            return (min(a, b),)
        if mode == "max":
            return (max(a, b),)
        raise ValueError(f"Unsupported mode: {mode}")


class RandomNumber:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {}}

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    @staticmethod
    def get_random():
        result = random.randint(0, 18446744073709551615)
        return (
            result,
            float(result),
        )

    def process(self):
        return RandomNumber.get_random()

    @classmethod
    def IS_CHANGED(cls):  # type: ignore
        return RandomNumber.get_random()


NODE_CLASS_MAPPINGS = {
    "signature_int2float": Int2Float,
    "signature_int_minmax": IntMinMax,
    "signature_int_clamp": IntClamp,
    "signature_int_operator": IntOperator,
    "signature_float2int": Float2Int,
    "signature_float_minmax": FloatMinMax,
    "signature_float_clamp": FloatClamp,
    "signature_float_operator": FloatOperator,
    "signature_random_number": RandomNumber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_int2float": "SIG Int2Float",
    "signature_int_minmax": "SIG Int MinMax",
    "signature_int_clamp": "SIG Int Clamp",
    "signature_int_operator": "SIG Int Operator",
    "signature_float2int": "SIG Float2Int",
    "signature_float_minmax": "SIG Float MinMax",
    "signature_float_clamp": "SIG Float Clamp",
    "signature_float_operator": "SIG Float Operator",
    "signature_random_number": "SIG Random Number",
}
