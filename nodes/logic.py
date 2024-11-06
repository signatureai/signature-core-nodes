import torch
from comfy_execution.graph_utils import GraphBuilder, is_link  # type: ignore

from .categories import LOGIC_CAT
from .shared import ByPassTypeTuple, any_type

MAX_FLOW_NUM = 10


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
            if a is None or b is None:
                return a is b

            if not isinstance(a, type(b)) and not isinstance(b, type(a)):
                return False

            if isinstance(a, torch.Tensor):
                if a.shape != b.shape:
                    a = a.reshape(-1)
                    b = b.reshape(-1)
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


class WhileLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {},
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"init_value_{i}"] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + [f"value_{i}" for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "execute"

    CATEGORY = LOGIC_CAT + "/Loops"

    def execute(self, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get(f"init_value_{i}", None))
        return tuple(["stub"] + values)


class WhileLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "end_loop": ("BOOLEAN", {}),
            },
            "optional": {},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"init_value_{i}"] = (any_type,)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(f"value_{i}" for i in range(MAX_FLOW_NUM)))
    FUNCTION = "execute"

    CATEGORY = LOGIC_CAT + "/Loops"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for _, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def execute(self, flow, end_loop, dynprompt=None, unique_id=None, **kwargs):
        if end_loop:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get(f"init_value_{i}", None))
            return tuple(values)

        # We want to loop
        if dynprompt is not None:
            _ = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        graph = GraphBuilder()
        for node_id in contained:
            if dynprompt is not None:
                original_node = dynprompt.get_node(node_id)
                node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
                node.set_override_display_id(node_id)
        for node_id in contained:
            if dynprompt is not None:
                original_node = dynprompt.get_node(node_id)
                node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
                for k, v in original_node["inputs"].items():
                    if is_link(v) and v[0] in contained:
                        parent = graph.lookup_node(v[0])
                        node.set_input(k, parent.out(v[1]))
                    else:
                        node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = f"init_value_{i}"
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }
