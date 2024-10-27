import torch

from .categories import LOGIC_CAT
from .shared import any_type


class LogicSwitch:
    """Switches between two input values based on a boolean condition.

    A logic gate that selects between two inputs of any type based on a boolean condition. When the
    condition is True, it returns the 'true' value; otherwise, it returns the 'false' value. This node
    is useful for creating conditional workflows and dynamic value selection.

    Args:
        condition (bool): The boolean condition that determines which value to return.
            Defaults to False if not provided.
        true (Any): The value to return when the condition is True. Can be of any type.
        false (Any): The value to return when the condition is False. Can be of any type.

    Returns:
        tuple[Any]: A single-element tuple containing either the 'true' or 'false' value based on
            the condition.

    Notes:
        - The node accepts inputs of any type, making it versatile for different data types
        - Both 'true' and 'false' values must be provided
        - The condition is automatically cast to boolean, with None being treated as False
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
    """Compares two values using equality operators and handles various data types.

    A comparison node that evaluates two inputs using either equality or inequality operators.
    Supports comparison of primitive types, tensors, and mixed data types with special handling for
    shape mismatches in tensors.

    Args:
        input_a (Any): First value for comparison. Can be of any type.
        input_b (Any): Second value for comparison. Can be of any type.
        operator (str): The comparison operator to use. Must be one of:
            - 'equal': Tests if inputs are equal
            - 'not_equal': Tests if inputs are not equal

    Returns:
        tuple[bool]: A single-element tuple containing the boolean result of the comparison.

    Raises:
        ValueError: If any input is None or if the operator is not one of the valid options.

    Notes:
        - Handles tensor comparisons with automatic shape adjustment
        - Different types are always considered unequal
        - For tensors with different shapes:
            * Tensors are flattened to 1D
            * Only the overlapping portions are compared
            * All elements must satisfy the condition for True result
        - None values are handled specially: None equals None, but nothing else
        - Lists and tuples are compared element-wise with all elements must match
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
