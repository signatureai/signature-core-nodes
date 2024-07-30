from .categories import NUMBERS_CAT
import random
class IntClamp():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "number": ("INT", {"default": 0, "forceInput": True, "min": -18446744073709551615, "max": 18446744073709551615}),
            "min": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            "max": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            }}
    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, number: int, min: int, max: int):
        if number < min:
            return (min,)
        if number > max:
            return (max,)
        return (number,)

class FloatClamp():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "number": ("FLOAT", {"default": 0, "forceInput": True, "min": -18446744073709551615, "max": 18446744073709551615}),
            "min": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            "max": ("FLOAT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615}),
            }}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, number: float, min: float, max: float):
        if number < min:
            return (min,)
        if number > max:
            return (max,)
        return (number,)

class Float2Int():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "number": ("FLOAT", {"default": 0, "forceInput": True}),
            }}
    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, number: float):
        return (int(number),)

class Int2Float():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "number": ("INT", {"default": 0, "forceInput": True}),
            }}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, number: int):
        return (float(number),)

class IntOperator():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "left": ("INT", {"default": 0}),
            "right": ("INT", {"default": 0}),
            "operator": (['+', '-', '*', '/'],),
            }}
    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, left: int, right: int, operator: str):
        if operator == "+":
            return (left + right,)
        elif operator == "-":
            return (left - right,)
        elif operator == "*":
            return (left * right,)
        elif operator == "/":
            return (left / right,)

        raise ValueError(f"Unsupported operator: {operator}")

class FloatOperator():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "left": ("FLOAT", {"default": 0}),
            "right": ("FLOAT", {"default": 0}),
            "operator": (['+', '-', '*', '/'],),
            }}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, left: float, right: float, operator: str):
        if operator == "+":
            return (left + right,)
        elif operator == "-":
            return (left - right,)
        elif operator == "*":
            return (left * right,)
        elif operator == "/":
            return (left / right,)

        raise ValueError(f"Unsupported operator: {operator}")

class IntMinMax():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "a": ("INT", {"default": 0, "forceInput": True}),
            "b": ("INT", {"default": 0, "forceInput": True}),
            "mode": (['min', 'max'],),
            }}
    RETURN_TYPES = ("INT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, a: int, b: int, mode: str):
        if mode == "min":
            return (min(a, b),)
        if mode == "max":
            return (max(a, b),)
        raise ValueError(f"Unsupported mode: {mode}")

class FloatMinMax():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "a": ("FLOAT", {"default": 0, "forceInput": True}),
            "b": ("FLOAT", {"default": 0, "forceInput": True}),
            "mode": (['min', 'max'],),
            }}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT
    def process(self, a: float, b: float, mode: str):
        if mode == "min":
            return (min(a, b),)
        if mode == "max":
            return (max(a, b),)
        raise ValueError(f"Unsupported mode: {mode}")

class RandomNumber():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {}}
    RETURN_TYPES = ("INT", "FLOAT", )
    FUNCTION = "process"
    CATEGORY = NUMBERS_CAT

    @staticmethod
    def get_random():
        result = random.randint(0, 99999999)
        return (result, float(result), )

    def process(self):
        return RandomNumber.get_random()

    @classmethod
    def IS_CHANGED(s): # type: ignore
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
    "signature_random_number": RandomNumber
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
    "signature_random_number": "SIG Random Number"
}