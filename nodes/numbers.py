import random

from .categories import NUMBERS_CAT
from .shared import MAX_FLOAT, MAX_INT


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
                        "min": -MAX_INT,
                        "max": MAX_INT,
                    },
                ),
                "min_value": (
                    "INT",
                    {"default": 0, "min": -MAX_INT, "max": MAX_INT, "step": 1},
                ),
                "max_value": (
                    "INT",
                    {"default": 0, "min": -MAX_INT, "max": MAX_INT, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, int):
            raise ValueError("Number must be an integer")
        min_value = kwargs.get("min_value")
        if not isinstance(min_value, int):
            raise ValueError("Min value must be an integer")
        max_value = kwargs.get("max_value")
        if not isinstance(max_value, int):
            raise ValueError("Max value must be an integer")
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
                        "min": -MAX_FLOAT,
                        "max": MAX_FLOAT,
                    },
                ),
                "min_value": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
                "max_value": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, float):
            raise ValueError("Number must be a float")
        min_value = kwargs.get("min_value")
        if not isinstance(min_value, float):
            raise ValueError("Min value must be a float")
        max_value = kwargs.get("max_value")
        if not isinstance(max_value, float):
            raise ValueError("Max value must be a float")

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

    def process(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, float):
            raise ValueError("Number must be a float")
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

    def process(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, int):
            raise ValueError("Number must be an integer")
        return (float(number),)


class IntOperator:
    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "left": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
                "right": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
                "operator": (["+", "-", "*", "/"],),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, **kwargs):
        left = kwargs.get("left")
        if not isinstance(left, float):
            raise ValueError("Left must be a float")
        right = kwargs.get("right")
        if not isinstance(right, float):
            raise ValueError("Right must be a float")
        operator = kwargs.get("operator")
        if not isinstance(operator, str):
            raise ValueError("Operator must be a string")
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
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
                "right": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.001},
                ),
                "operator": (["+", "-", "*", "/"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, **kwargs):
        left = kwargs.get("left")
        if not isinstance(left, float):
            raise ValueError("Left must be a float")
        right = kwargs.get("right")
        if not isinstance(right, float):
            raise ValueError("Right must be a float")
        operator = kwargs.get("operator")
        if not isinstance(operator, str):
            raise ValueError("Operator must be a string")
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

    def process(self, **kwargs):
        a = kwargs.get("a")
        b = kwargs.get("b")
        if not isinstance(a, int):
            raise ValueError("A must be an integer")
        if not isinstance(b, int):
            raise ValueError("B must be an integer")
        mode = kwargs.get("mode")
        if not isinstance(mode, str):
            raise ValueError("Mode must be a string")
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
                "a": ("FLOAT", {"default": 0, "forceInput": True}),
                "b": ("FLOAT", {"default": 0, "forceInput": True}),
                "mode": (["min", "max"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    def process(self, **kwargs):
        a = kwargs.get("a")
        b = kwargs.get("b")
        if not isinstance(a, float):
            raise ValueError("A must be a float")
        if not isinstance(b, float):
            raise ValueError("B must be a float")
        mode = kwargs.get("mode")
        if not isinstance(mode, str):
            raise ValueError("Mode must be a string")
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
        result = random.randint(0, MAX_INT)
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
    "signature_int_minmax": "SIG IntMinMax",
    "signature_int_clamp": "SIG IntClamp",
    "signature_int_operator": "SIG IntOperator",
    "signature_float2int": "SIG Float2Int",
    "signature_float_minmax": "SIG FloatMinMax",
    "signature_float_clamp": "SIG FloatClamp",
    "signature_float_operator": "SIG FloatOperator",
    "signature_random_number": "SIG RandomNumber",
}
