# Data Nodes

## JsonToDict

Converts a JSON string to a Python dictionary.

This class parses a JSON-formatted string and converts it into a Python dictionary.

Methods: execute(\*\*kwargs): Parses the JSON string and returns the resulting
dictionary.

Args: json_str (str): The JSON string to be converted.

Returns: tuple: A tuple containing the resulting dictionary.

Raises: ValueError: If the input is not a string.

### Return Types

- `DICT`

::: nodes.data.JsonToDict

## DictToJson

Converts a Python dictionary to a JSON string.

This class serializes a Python dictionary into a JSON-formatted string.

Methods: execute(\*\*kwargs): Serializes the dictionary and returns the resulting JSON
string.

Args: dict (dict): The dictionary to be converted.

Returns: tuple: A tuple containing the resulting JSON string.

### Return Types

- `STRING`

::: nodes.data.DictToJson

## GetImageListItem

Retrieves an image from a list by index.

This class accesses a list of images and retrieves the image at the specified index.

Methods: execute(\*\*kwargs): Returns the image at the specified index.

Args: images (list): The list of images. index (int): The index of the image to
retrieve.

Returns: tuple: A tuple containing the retrieved image.

Raises: ValueError: If the index is not an integer or if images is not a list.

### Return Types

- `I`
- `M`
- `A`
- `G`
- `E`

::: nodes.data.GetImageListItem

## GetListItem

Retrieves an item from a list by index and returns its type.

This class accesses a list and retrieves the item at the specified index, also returning
the item's type.

Methods: execute(\*\*kwargs): Returns the item and its type.

Args: list (list): The list to access. index (int): The index of the item to retrieve.

Returns: tuple: A tuple containing the item and its type as a string.

Raises: ValueError: If the index is not an integer or if the list is not a list.

::: nodes.data.GetListItem

## GetDictValue

Retrieves a value from a dictionary by key and returns its type.

This class accesses a dictionary and retrieves the value associated with the specified
key, also returning the value's type.

Methods: execute(\*\*kwargs): Returns the value and its type.

Args: dict (dict): The dictionary to access. key (str): The key of the value to
retrieve.

Returns: tuple: A tuple containing the value and its type as a string.

Raises: ValueError: If the key is not a string or if the dict is not a dictionary.

::: nodes.data.GetDictValue
