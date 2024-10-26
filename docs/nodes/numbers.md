# Numbers Nodes

## IntClamp

Clamps an integer within a specified range.

This class ensures that an integer input is clamped between a minimum and maximum value.

Methods: execute(\*\*kwargs): Returns the clamped integer value.

Raises: ValueError: If the input values are not integers.

### Return Types

- `INT`

::: nodes.numbers.IntClamp

## FloatClamp

Clamps a float within a specified range.

This class ensures that a float input is clamped between a minimum and maximum value.

Methods: execute(\*\*kwargs): Returns the clamped float value.

Raises: ValueError: If the input values are not floats.

### Return Types

- `FLOAT`

::: nodes.numbers.FloatClamp

## Float2Int

Converts a float to an integer.

This class converts a floating-point number to an integer by truncating the decimal
part.

Methods: execute(\*\*kwargs): Returns the integer representation of the float.

Raises: ValueError: If the input value is not a float.

### Return Types

- `INT`

::: nodes.numbers.Float2Int

## Int2Float

Converts an integer to a float.

This class converts an integer to a floating-point number.

Methods: execute(\*\*kwargs): Returns the float representation of the integer.

Raises: ValueError: If the input value is not an integer.

### Return Types

- `FLOAT`

::: nodes.numbers.Int2Float

## IntOperator

Performs arithmetic operations on two floats and returns an integer.

This class supports basic arithmetic operations (+, -, \*, /) on two float inputs and
returns the result as an integer.

Methods: execute(\*\*kwargs): Returns the result of the arithmetic operation.

Raises: ValueError: If the input values are not floats or if the operator is
unsupported.

### Return Types

- `INT`

::: nodes.numbers.IntOperator

## FloatOperator

Performs arithmetic operations on two floats.

This class supports basic arithmetic operations (+, -, \*, /) on two float inputs.

Methods: execute(\*\*kwargs): Returns the result of the arithmetic operation.

Raises: ValueError: If the input values are not floats or if the operator is
unsupported.

### Return Types

- `FLOAT`

::: nodes.numbers.FloatOperator

## IntMinMax

Finds the minimum or maximum of two integers.

This class returns either the minimum or maximum of two integer inputs based on the
specified mode.

Methods: execute(\*\*kwargs): Returns the minimum or maximum integer.

Raises: ValueError: If the input values are not integers or if the mode is unsupported.

### Return Types

- `INT`

::: nodes.numbers.IntMinMax

## FloatMinMax

Finds the minimum or maximum of two floats.

This class returns either the minimum or maximum of two float inputs based on the
specified mode.

Methods: execute(\*\*kwargs): Returns the minimum or maximum float.

Raises: ValueError: If the input values are not floats or if the mode is unsupported.

### Return Types

- `FLOAT`

::: nodes.numbers.FloatMinMax

## RandomNumber

Generates a random integer and its float representation.

This class generates a random integer within a specified range and provides its float
representation.

Methods: execute(): Returns a tuple containing the random integer and its float
representation.

### Return Types

- `INT`
- `FLOAT`

::: nodes.numbers.RandomNumber

## MathOperator

Evaluates mathematical expressions using variables and operators.

This class evaluates mathematical expressions that can include variables (a, b, c, d)
and a variety of operators.

Methods: execute(\*\*kwargs): Returns the result of the evaluated expression as both an
integer and a float.

Raises: ValueError: If the expression contains unsupported operations or invalid syntax.

### Return Types

- `INT`
- `FLOAT`

::: nodes.numbers.MathOperator
