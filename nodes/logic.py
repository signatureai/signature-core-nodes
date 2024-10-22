import torch

from .categories import LOGIC_CAT
from .shared import any_type


class LogicSwitch:
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
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                # Convert to tensors if not already
                a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
                b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b

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


NODE_CLASS_MAPPINGS = {
    "signature_logic_compare": LogicCompare,
    "signature_logic_switch": LogicSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "signature_logic_compare": "SIG LogicCompare",
    "signature_logic_switch": "SIG LogicSwitch",
}
