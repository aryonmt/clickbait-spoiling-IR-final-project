import numpy as np
from transformers import AutoTokenizer

from src.qa_inference import predict_best_span


class MockModelOutput:
    """Mock container for sequence model logits."""

    def __init__(self, start_logits: np.ndarray, end_logits: np.ndarray):
        import torch

        self.start_logits = torch.tensor(start_logits)
        self.end_logits = torch.tensor(end_logits)


def test_predict_best_span_sliding_window():
    """Ensures predict_best_span correctly filters context boundaries and length."""
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-RoBERTa")
    question = "Where is Paris?"
    context = "Paris is the capital of France. It is a beautiful city in Europe."

    # Dynamically locate valid context token indexes for this tokenizer instance
    encodings = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=32,
        stride=8,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )
    sequence_ids = encodings.sequence_ids(0)
    context_indices = [idx for idx, s_id in enumerate(sequence_ids) if s_id == 1]

    # Guarantee we have valid context tokens to assert
    assert len(context_indices) >= 2
    s_mock = context_indices[0]
    e_mock = context_indices[1]  # Adjacent context tokens

    class MockModel:
        """Mock QA model mimicking logit outputs inside context sequence limits."""

        def __call__(self, input_ids, attention_mask):
            num_windows = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            start = np.zeros((num_windows, seq_len))
            end = np.zeros((num_windows, seq_len))

            # Place mock high logit scores dynamically at valid context positions
            start[:, s_mock] = 10.0
            end[:, e_mock] = 10.0
            return MockModelOutput(start, end)

        def eval(self):
            pass

    best_span = predict_best_span(
        model=MockModel(),
        tokenizer=tokenizer,
        question=question,
        context=context,
        max_length=32,
        doc_stride=8,
        max_answer_length=5,
        device="cpu",
    )

    assert isinstance(best_span, dict)
    assert "text" in best_span
    assert best_span["score"] == 20.0  # Must be exactly 10.0 + 10.0
    assert len(best_span["text"]) > 0
