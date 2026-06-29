from typing import List, Union

# Global constant for strict label ordering across evaluation and visualization
SPOILER_LABELS = ["phrase", "passage", "multi"]


def extract_primary_tag(tag_input: Union[str, List[str]]) -> str:
    """Extracts the first string tag if the input is a list, otherwise returns
    the raw string safely. Returns an empty string if the input is empty or None.

    Args:
        tag_input (Union[str, List[str]]): The tag field extracted from the
          dataset.

    Returns:
        str: The cleaned primary scalar tag.
    """
    if not tag_input:
        return ""
    if isinstance(tag_input, list):
        return str(tag_input[0]) if len(tag_input) > 0 else ""
    return str(tag_input)
