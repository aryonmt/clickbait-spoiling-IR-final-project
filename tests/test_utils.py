from src.utils import extract_primary_tag


def test_extract_primary_tag_list():
    """Verifies that extract_primary_tag extracts the first item from lists."""
    assert extract_primary_tag(["phrase"]) == "phrase"
    assert extract_primary_tag(["multi", "phrase"]) == "multi"


def test_extract_primary_tag_str():
    """Verifies that extract_primary_tag preserves single string values."""
    assert extract_primary_tag("passage") == "passage"


def test_extract_primary_tag_empty():
    """Verifies that extract_primary_tag handles empty structures gracefully."""
    assert extract_primary_tag([]) == ""
