import torch

from .categories import LOGIC_CAT
from .shared import any_type


class LogicSwitch:
    """Switches between two values based on a boolean condition.

    This class returns one of two values depending on the boolean condition provided.

    Methods:
        execute(**kwargs): Returns the 'true' value if the condition is True, otherwise returns the 'false' value.

    Args:
        condition (bool): The condition to evaluate.
        true: The value to return if the condition is True.
        false: The value to return if the condition is False.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": False, "forceInput": True}),
                "true": (any_type,),
                "false": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "execute"
    CATEGORY = LOGIC_CAT

    def execute(self, **kwargs):
        condition = kwargs.get("condition") or False
        true_val = kwargs.get("true")
        false_val = kwargs.get("false")

        output = true_val if condition else false_val
        return (output,)


class LogicCompare:
    """Compares two values using a specified operator.

    This class compares two input values using either 'equal' or 'not_equal' operators and returns a boolean result.

    Methods:
        execute(**kwargs): Returns True if the comparison is successful, otherwise False.

    Args:
        input_a: The first value to compare.
        input_b: The second value to compare.
        operator (str): The comparison operator ('equal' or 'not_equal').

    Raises:
        ValueError: If any input is None or if the operator is invalid.
    """

    @classmethod
    def INPUT_TYPES(cls):  # type: ignore
        return {
            "required": {
                "input_a": (any_type,),
                "input_b": (any_type,),
                "operator": (["equal", "not_equal"],),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = LOGIC_CAT

    def execute(self, **kwargs):
        input_a = kwargs.get("input_a")
        input_b = kwargs.get("input_b")
        operator = kwargs.get("operator")

        if input_a is None or input_b is None or operator is None:
            raise ValueError("All inputs are required")

        def safe_compare(a, b, tensor_op, primitive_op):
            # Handle None values
            if a is None or b is None:
                return a is b

            # If types are different, they're never equal
            if not isinstance(a, type(b)) and not isinstance(b, type(a)):
                return False

            # Now we know both types are compatible
            if isinstance(a, torch.Tensor):
                if a.shape != b.shape:
                    # Reshape tensors to 1D for comparison
                    a = a.reshape(-1)
                    b = b.reshape(-1)
                    # If sizes still don't match, compare only the overlapping part
                    min_size = min(a.size(0), b.size(0))
                    a = a[:min_size]
                    b = b[:min_size]
                return tensor_op(a, b)

            return primitive_op(a, b)

        operator_map = {
            "equal": lambda a, b: safe_compare(a, b, torch.eq, lambda x, y: x == y),
            "not_equal": lambda a, b: safe_compare(a, b, torch.ne, lambda x, y: x != y),
        }

        if operator not in operator_map:
            raise ValueError(
                f"Invalid operator: {operator}. Valid operators are: {', '.join(operator_map.keys())}"
            )

        output = operator_map[operator](input_a, input_b)

        # Handle different output types
        if isinstance(output, torch.Tensor):
            output = output.all().item()
        elif isinstance(output, (list, tuple)):
            output = all(output)

        return (bool(output),)
