import os

import pytest

from src.data_loader import JSONLLoader


def test_jsonl_loader_success():
    """Verifies data ingestion and sanity check operations."""
    fixture_path = os.path.join("tests", "fixtures", "tiny_train.jsonl")
    loader = JSONLLoader(fixture_path)
    df = loader.load_data()
    assert len(df) == 4
    assert "uuid" in df.columns
    loader.run_sanity_checks()


def test_jsonl_loader_file_not_found():
    """Ensures file loader raises expected exception for missing paths."""
    loader = JSONLLoader("non_existent_file.jsonl")
    with pytest.raises(FileNotFoundError):
        loader.load_data()
