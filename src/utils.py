from typing import List, Union


def extract_primary_tag(tag_input: Union[str, List[str]]) -> str:
    """Extracts the first string tag if the input is a list, otherwise returns

    the raw string safely.

    Args:
        tag_input (Union[str, List[str]]): The tag field extracted from the
          dataset.

    Returns:
        str: The cleaned primary scalar tag.
    """
    if isinstance(tag_input, list) and len(tag_input) > 0:
        return str(tag_input[0])
    return str(tag_input)
