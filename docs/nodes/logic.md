# Logic Nodes

## LogicSwitch

Switches between two values based on a boolean condition.

This class returns one of two values depending on the boolean condition provided.

Methods: execute(\*\*kwargs): Returns the 'true' value if the condition is True,
otherwise returns the 'false' value.

Args: condition (bool): The condition to evaluate. true: The value to return if the
condition is True. false: The value to return if the condition is False.

::: nodes.logic.LogicSwitch

## LogicCompare

Compares two values using a specified operator.

This class compares two input values using either 'equal' or 'not_equal' operators and
returns a boolean result.

Methods: execute(\*\*kwargs): Returns True if the comparison is successful, otherwise
False.

Args: input_a: The first value to compare. input_b: The second value to compare.
operator (str): The comparison operator ('equal' or 'not_equal').

Raises: ValueError: If any input is None or if the operator is invalid.

### Return Types

- `BOOLEAN`

::: nodes.logic.LogicCompare
