from datasets import Dataset

from src.qa_preprocessor import QAPreprocessor, find_answer_span


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


def test_find_answer_span_exact():
    """Tests if exact matching works correctly."""
    assert find_answer_span("this is a test context", "test") == (10, 14)


def test_find_answer_span_normalized():
    """Tests quotation and white-space normalization robustness."""
    assert find_answer_span("she said “hello  world”", '"hello world"') == (9, 23)
    assert find_answer_span("the price was $5.00", " $5.00 ") == (14, 19)
    assert find_answer_span("empty spaces", "") is None
