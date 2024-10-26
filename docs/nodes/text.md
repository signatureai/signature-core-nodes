# Text Nodes

## TextPreview

Generates a preview of text inputs.

This class takes a list of text inputs and generates a single string preview. If the
input is a torch.Tensor, it includes the tensor's shape in the preview.

Returns: dict: A dictionary containing the preview text under the 'ui' key and the
result as a tuple.

### Return Types

- `STRING`

::: nodes.text.TextPreview

## TextCase

Changes the case of a given text.

This class provides functionality to convert text to lower, upper, capitalize, or title
case.

Args: text (str): The input text to be transformed. case (str): The case transformation
to apply ('lower', 'upper', 'capitalize', 'title').

Returns: tuple: The transformed text.

### Return Types

- `STRING`

::: nodes.text.TextCase

## TextTrim

Trims whitespace from text.

This class trims whitespace from the left, right, or both sides of the input text.

Args: text (str): The input text to be trimmed. trim_type (str): The type of trim to
apply ('both', 'left', 'right').

Returns: tuple: The trimmed text.

### Return Types

- `STRING`

::: nodes.text.TextTrim

## TextSplit

Splits text into a list based on a delimiter.

This class splits the input text into a list of strings using the specified delimiter.

Args: text (str): The input text to be split. delimiter (str): The delimiter to use for
splitting the text.

Returns: tuple: A list of split text segments.

### Return Types

- `STRING`

::: nodes.text.TextSplit

## TextRegexReplace

Performs regex-based text replacement.

This class uses regular expressions to find and replace patterns in the input text.

Args: text (str): The input text to be processed. pattern (str): The regex pattern to
search for. replacement (str): The string to replace the pattern with.

Returns: tuple: The text after regex replacement.

### Return Types

- `STRING`

::: nodes.text.TextRegexReplace

## TextFindReplace

Finds and replaces text.

This class finds a specified substring in the input text and replaces it with another
substring.

Args: text (str): The input text to be processed. find (str): The substring to find.
replace (str): The substring to replace with.

Returns: tuple: The text after find and replace.

### Return Types

- `STRING`

::: nodes.text.TextFindReplace

## TextConcatenate

Concatenates two text strings.

This class concatenates two input text strings into a single string.

Args: text1 (str): The first text string. text2 (str): The second text string.

Returns: tuple: The concatenated text.

### Return Types

- `STRING`

::: nodes.text.TextConcatenate
