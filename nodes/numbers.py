from .categories import NUMBERS_CAT
import random
class IntClamp():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "number": ("INT", {"default": 0, "forceInput": True}),
            "min": ("INT", {"default": 0}),
            "max": ("INT", {"default": 0}),
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
            "number": ("FLOAT", {"default": 0, "forceInput": True}),
            "min": ("FLOAT", {"default": 0}),
            "max": ("FLOAT", {"default": 0}),
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
    "Signature Int2Float": Int2Float,
    "Signature IntMinMax": IntMinMax,
    "Signature IntClamp": IntClamp,
    "Signature IntOperator": IntOperator,
    "Signature Float2Int": Float2Int,
    "Signature FloatMinMax": FloatMinMax,
    "Signature FloatClamp": FloatClamp,
    "Signature FloatOperator": FloatOperator,
    "Signature RandomNumber": RandomNumber
}