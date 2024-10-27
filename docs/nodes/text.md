# Text Nodes

## TextPreview

Generates a preview of text inputs

### Inputs

| Group    | Name | Type                                  | Default | Extras |
| -------- | ---- | ------------------------------------- | ------- | ------ |
| required | text | `<ast.Name object at 0x7f0a3379bb50>` |         |        |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextPreview:
        """Generates a preview of text inputs.

        This class takes a list of text inputs and generates a single string preview.
        If the input is a torch.Tensor, it includes the tensor's shape in the preview.

        Returns:
            dict: A dictionary containing the preview text under the 'ui' key and the result as a tuple.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "text": (any_type,),
                },
            }

        INPUT_IS_LIST = True
        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        OUTPUT_NODE = True
        OUTPUT_IS_LIST = (True,)

        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text = kwargs.get("text", [])
            text_string = ""
            for t in text:
                if t is None:
                    continue
                if text_string != "":
                    text_string += "\n"
                text_string += str(t.shape) if isinstance(t, torch.Tensor) else str(t)
            return {"ui": {"text": [text_string]}, "result": (text_string,)}
    ```

## TextCase

Changes the case of a given text

### Inputs

| Group    | Name | Type     | Default | Extras          |
| -------- | ---- | -------- | ------- | --------------- |
| required | text | `STRING` |         | forceInput=True |
| required | case | `LIST`   |         |                 |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextCase:
        """Changes the case of a given text.

        This class provides functionality to convert text to lower, upper, capitalize, or title case.

        Args:
            text (str): The input text to be transformed.
            case (str): The case transformation to apply ('lower', 'upper', 'capitalize', 'title').

        Returns:
            tuple: The transformed text.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "text": ("STRING", {"forceInput": True}),
                    "case": (["lower", "upper", "capitalize", "title"],),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text = kwargs.get("text") or ""
            case = kwargs.get("case") or "lower"
            result = text
            if case == "lower":
                result = text.lower()
            if case == "upper":
                result = text.upper()
            if case == "capitalize":
                result = text.capitalize()
            if case == "title":
                result = text.title()
            return (result,)
    ```

## TextTrim

Trims whitespace from text

### Inputs

| Group    | Name      | Type     | Default | Extras          |
| -------- | --------- | -------- | ------- | --------------- |
| required | text      | `STRING` |         | forceInput=True |
| required | trim_type | `LIST`   |         |                 |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextTrim:
        """Trims whitespace from text.

        This class trims whitespace from the left, right, or both sides of the input text.

        Args:
            text (str): The input text to be trimmed.
            trim_type (str): The type of trim to apply ('both', 'left', 'right').

        Returns:
            tuple: The trimmed text.
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "text": ("STRING", {"forceInput": True}),
                    "trim_type": (["both", "left", "right"],),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text = kwargs.get("text") or ""
            trim_type = kwargs.get("trim_type") or "both"
            if trim_type == "both":
                return (text.strip(),)
            if trim_type == "left":
                return (text.lstrip(),)
            if trim_type == "right":
                return (text.rstrip(),)
            return (text,)
    ```

## TextSplit

Splits text into a list based on a delimiter

### Inputs

| Group    | Name      | Type     | Default | Extras          |
| -------- | --------- | -------- | ------- | --------------- |
| required | text      | `STRING` |         | forceInput=True |
| required | delimiter | `STRING` |         |                 |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextSplit:
        """Splits text into a list based on a delimiter.

        This class splits the input text into a list of strings using the specified delimiter.

        Args:
            text (str): The input text to be split.
            delimiter (str): The delimiter to use for splitting the text.

        Returns:
            tuple: A list of split text segments.
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "text": ("STRING", {"forceInput": True}),
                    "delimiter": ("STRING", {"default": " "}),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT
        OUTPUT_IS_LIST = (True,)

        def execute(self, **kwargs):
            text = kwargs.get("text", "")
            delimiter = kwargs.get("delimiter", " ")
            return (text.split(delimiter),)
    ```

## TextRegexReplace

Performs regex-based text replacement

### Inputs

| Group    | Name        | Type     | Default | Extras          |
| -------- | ----------- | -------- | ------- | --------------- |
| required | text        | `STRING` |         | forceInput=True |
| required | pattern     | `STRING` |         |                 |
| required | replacement | `STRING` |         |                 |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextRegexReplace:
        """Performs regex-based text replacement.

        This class uses regular expressions to find and replace patterns in the input text.

        Args:
            text (str): The input text to be processed.
            pattern (str): The regex pattern to search for.
            replacement (str): The string to replace the pattern with.

        Returns:
            tuple: The text after regex replacement.
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "text": ("STRING", {"forceInput": True}),
                    "pattern": ("STRING", {"default": ""}),
                    "replacement": ("STRING", {"default": ""}),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text = kwargs.get("text", "")
            pattern = kwargs.get("pattern", "")
            replacement = kwargs.get("replacement", "")
            return (re.sub(pattern, replacement, text),)
    ```

## TextFindReplace

Finds and replaces text

### Inputs

| Group    | Name    | Type     | Default | Extras |
| -------- | ------- | -------- | ------- | ------ |
| required | text    | `STRING` |         |        |
| required | find    | `STRING` |         |        |
| required | replace | `STRING` |         |        |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextFindReplace:
        """Finds and replaces text.

        This class finds a specified substring in the input text and replaces it with another substring.

        Args:
            text (str): The input text to be processed.
            find (str): The substring to find.
            replace (str): The substring to replace with.

        Returns:
            tuple: The text after find and replace.
        """

        @classmethod
        def INPUT_TYPES(cls):  # type: ignore
            return {
                "required": {
                    "text": ("STRING", {"default": ""}),
                    "find": ("STRING", {"default": ""}),
                    "replace": ("STRING", {"default": ""}),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text = kwargs.get("text") or ""
            find = kwargs.get("find") or ""
            replace = kwargs.get("replace") or ""
            return (text.replace(find, replace),)
    ```

## TextConcatenate

Concatenates two text strings

### Inputs

| Group    | Name  | Type     | Default | Extras |
| -------- | ----- | -------- | ------- | ------ |
| required | text1 | `STRING` |         |        |
| required | text2 | `STRING` |         |        |

### Returns

| Name   | Type     |
| ------ | -------- |
| string | `STRING` |

??? note "Pick the code in text.py"

    ```python
    class TextConcatenate:
        """Concatenates two text strings.

        This class concatenates two input text strings into a single string.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            tuple: The concatenated text.
        """

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "text1": ("STRING", {"default": ""}),
                    "text2": ("STRING", {"default": ""}),
                },
            }

        RETURN_TYPES = ("STRING",)
        FUNCTION = "execute"
        CATEGORY = TEXT_CAT

        def execute(self, **kwargs):
            text1 = kwargs.get("text1", "")
            text2 = kwargs.get("text2", "")
            return (text1 + text2,)
    ```
