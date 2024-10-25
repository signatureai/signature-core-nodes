import ast
import math
import operator as op
import random

from .categories import NUMBERS_CAT
from .shared import MAX_FLOAT, MAX_INT


class IntClamp:
    """Clamps an integer within a specified range.

    This class ensures that an integer input is clamped between a minimum and maximum value.

    Methods:
        execute(**kwargs): Returns the clamped integer value.

    Raises:
        ValueError: If the input values are not integers.
    """

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
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Clamps a float within a specified range.

    This class ensures that a float input is clamped between a minimum and maximum value.

    Methods:
        execute(**kwargs): Returns the clamped float value.

    Raises:
        ValueError: If the input values are not floats.
    """

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
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
                "max_value": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Converts a float to an integer.

    This class converts a floating-point number to an integer by truncating the decimal part.

    Methods:
        execute(**kwargs): Returns the integer representation of the float.

    Raises:
        ValueError: If the input value is not a float.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": ("FLOAT", {"default": 0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, float):
            raise ValueError("Number must be a float")
        return (int(number),)


class Int2Float:
    """Converts an integer to a float.

    This class converts an integer to a floating-point number.

    Methods:
        execute(**kwargs): Returns the float representation of the integer.

    Raises:
        ValueError: If the input value is not an integer.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "number": ("INT", {"default": 0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
        number = kwargs.get("number")
        if not isinstance(number, int):
            raise ValueError("Number must be an integer")
        return (float(number),)


class IntOperator:
    """Performs arithmetic operations on two floats and returns an integer.

    This class supports basic arithmetic operations (+, -, *, /) on two float inputs
    and returns the result as an integer.

    Methods:
        execute(**kwargs): Returns the result of the arithmetic operation.

    Raises:
        ValueError: If the input values are not floats or if the operator is unsupported.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "left": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
                "right": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
                "operator": (["+", "-", "*", "/"],),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Performs arithmetic operations on two floats.

    This class supports basic arithmetic operations (+, -, *, /) on two float inputs.

    Methods:
        execute(**kwargs): Returns the result of the arithmetic operation.

    Raises:
        ValueError: If the input values are not floats or if the operator is unsupported.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "left": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
                "right": (
                    "FLOAT",
                    {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01},
                ),
                "operator": (["+", "-", "*", "/"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Finds the minimum or maximum of two integers.

    This class returns either the minimum or maximum of two integer inputs based on the specified mode.

    Methods:
        execute(**kwargs): Returns the minimum or maximum integer.

    Raises:
        ValueError: If the input values are not integers or if the mode is unsupported.
    """

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
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Finds the minimum or maximum of two floats.

    This class returns either the minimum or maximum of two float inputs based on the specified mode.

    Methods:
        execute(**kwargs): Returns the minimum or maximum float.

    Raises:
        ValueError: If the input values are not floats or if the mode is unsupported.
    """

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
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):
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
    """Generates a random integer and its float representation.

    This class generates a random integer within a specified range and provides its float representation.

    Methods:
        execute(): Returns a tuple containing the random integer and its float representation.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {"required": {}}

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    @staticmethod
    def get_random():
        result = random.randint(0, MAX_INT)
        return (
            result,
            float(result),
        )

    def execute(self):
        return RandomNumber.get_random()

    @classmethod
    def IS_CHANGED(cls):  # type: ignore
        return RandomNumber.get_random()


class MathOperator:
    """Evaluates mathematical expressions using variables and operators.

    This class evaluates mathematical expressions that can include variables (a, b, c, d) and a variety of operators.

    Methods:
        execute(**kwargs): Returns the result of the evaluated expression as both an integer and a float.

    Raises:
        ValueError: If the expression contains unsupported operations or invalid syntax.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "optional": {
                "a": ("FLOAT", {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01}),
                "b": ("FLOAT", {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01}),
                "c": ("FLOAT", {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01}),
                "d": ("FLOAT", {"default": 0, "min": -MAX_FLOAT, "max": MAX_FLOAT, "step": 0.01}),
            },
            "required": {
                "value": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "INT",
        "FLOAT",
    )
    FUNCTION = "execute"
    CATEGORY = NUMBERS_CAT

    def execute(self, **kwargs):

        a = kwargs.get("a") or 0.0
        b = kwargs.get("b") or 0.0
        c = kwargs.get("c") or 0.0
        d = kwargs.get("d") or 0.0
        value = kwargs.get("value") or ""
        # trim value
        value = value.strip()

        def safe_xor(x, y):
            if isinstance(x, float) or isinstance(y, float):
                # Convert to integers if either operand is a float
                return float(int(x) ^ int(y))
            return op.xor(x, y)

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.Mod: op.mod,
            ast.Eq: op.eq,
            ast.NotEq: op.ne,
            ast.Lt: op.lt,
            ast.LtE: op.le,
            ast.Gt: op.gt,
            ast.GtE: op.ge,
            ast.And: lambda x, y: x and y,
            ast.Or: lambda x, y: x or y,
            ast.Not: op.not_,
            ast.BitXor: safe_xor,  # Use the safe_xor function
        }

        op_functions = {
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len,
        }

        def eval_(node):
            if isinstance(node, ast.Num):  # number
                return node.n
            if isinstance(node, ast.Name):  # variable
                if node.id == "a":
                    return a
                if node.id == "b":
                    return b
                if node.id == "c":
                    return c
                if node.id == "d":
                    return d
            if isinstance(node, ast.BinOp):  # <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))  # type: ignore
            if isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_(node.operand))  # type: ignore
            if isinstance(node, ast.Compare):  # comparison operators
                left = eval_(node.left)
                for operator, comparator in zip(node.ops, node.comparators):
                    if not operators[type(operator)](left, eval_(comparator)):  # type: ignore
                        return 0
                return 1
            if isinstance(node, ast.BoolOp):  # boolean operators (And, Or)
                values = [eval_(value) for value in node.values]
                return operators[type(node.op)](*values)  # type: ignore
            if isinstance(node, ast.Call):  # custom function
                if node.func.id in op_functions:  # type: ignore
                    args = [eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)  # type: ignore
            if isinstance(node, ast.Subscript):  # indexing or slicing
                value = eval_(node.value)
                if isinstance(node.slice, ast.Constant):
                    return value[node.slice.value]
                return 0
            return 0

        result = eval_(ast.parse(value, mode="eval").body)

        if math.isnan(result):  # type: ignore
            result = 0.0

        return (
            round(result),  # type: ignore
            result,
        )  # type: ignore
