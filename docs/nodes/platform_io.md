# Platform_Io Nodes

## PlatformInputImage

Handles image input for the platform.

This class processes image inputs, supporting both single and multiple images. It can
handle images from URLs or base64 strings and apply post-processing like alpha channel
removal.

Methods: execute(\*\*kwargs): Processes the image input and returns a list of processed
images.

Raises: ValueError: If input values are not of the expected types or if no valid input
is found.

::: nodes.platform_io.PlatformInputImage

## PlatformInputConnector

Handles input from external connectors like Google Drive.

This class manages file downloads from external services using provided tokens and file
IDs.

Methods: execute(\*\*kwargs): Downloads the specified file and returns the file data.

Raises: ValueError: If input values are not of the expected types.

### Return Types

- `FILE`

::: nodes.platform_io.PlatformInputConnector

## PlatformInputText

Handles text input for the platform.

This class processes text inputs, providing a fallback option if the input is empty.

Methods: execute(\*\*kwargs): Returns the input text or the fallback if the input is
empty.

Raises: ValueError: If input values are not of the expected types.

### Return Types

- `STRING`

::: nodes.platform_io.PlatformInputText

## PlatformInputNumber

Handles numeric input for the platform.

This class processes numeric inputs, supporting both integers and floats.

Methods: execute(\*\*kwargs): Returns the input number, converting it to the specified
subtype.

Raises: ValueError: If input values are not of the expected types.

::: nodes.platform_io.PlatformInputNumber

## PlatformInputBoolean

Handles boolean input for the platform.

This class processes boolean inputs.

Methods: execute(\*\*kwargs): Returns the input boolean value.

Raises: ValueError: If input values are not of the expected types.

### Return Types

- `BOOLEAN`

::: nodes.platform_io.PlatformInputBoolean

## PlatformOutput

Handles output for the platform.

This class manages the output of various data types, including images, numbers, and
strings. It supports saving images and generating thumbnails.

Methods: execute(\*\*kwargs): Processes and saves the output data, returning metadata
about the saved files.

Raises: ValueError: If input values are not of the expected types or if unsupported
output types are provided.

::: nodes.platform_io.PlatformOutput
