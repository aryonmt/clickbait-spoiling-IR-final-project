from datasets import Dataset

from src.qa_preprocessor import QAPreprocessor


def test_prepare_train_features():
    """Tests the qa_preprocessor span indexing and fallback logic."""
    # Using a super lightweight random architecture with custom stride for fast testing
    preprocessor = QAPreprocessor(
        model_name="hf-internal-testing/tiny-random-RoBERTa",
        max_length=128,
        doc_stride=16,  # Explicitly set low stride to satisfy small test models constraints
    )

    examples = {
        "postText": [["What a day"]],
        "targetParagraphs": [["It was sunny.", "Indeed."]],
        "spoiler": [["sunny"]],
        "tags": [["phrase"]],
        "uuid": ["test-uuid-0"],
    }
    dataset = Dataset.from_dict(examples)
    tokenized = dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=dataset.column_names,
    )

    assert "start_positions" in tokenized.column_names
    assert "end_positions" in tokenized.column_names
    assert len(tokenized["start_positions"]) == 1
