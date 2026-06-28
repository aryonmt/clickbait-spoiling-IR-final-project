from transformers import AutoTokenizer


class QAPreprocessor:
    """Preprocesses Clickbait Spoiling datasets into standardized extractive

    Question-Answering token spans.
    """

    def __init__(self, model_name: str, max_length: int = 512, doc_stride: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.doc_stride = doc_stride

    def prepare_train_features(self, examples):
        # Flatten posts and target paragraphs into single strings
        questions = [
            " ".join(q) if isinstance(q, list) else str(q) for q in examples["postText"]
        ]

        contexts = []
        for paras in examples["targetParagraphs"]:
            contexts.append(" ".join(paras) if isinstance(paras, list) else str(paras))

        # Tokenize with truncation and padding, keeping overflows via sliding window
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Get the original sample index
            sample_index = sample_mapping[i]
            context = contexts[sample_index]

            # Extract spoiler list and reconstruct primary target text
            spoilers = examples["spoiler"][sample_index]
            answer_text = (
                spoilers[0]
                if (isinstance(spoilers, list) and len(spoilers) > 0)
                else ""
            )

            # Locate character span in text
            start_char = context.find(answer_text) if answer_text else -1

            if start_char == -1:
                # If answer isn't explicitly found, anchor to CLS token (unanswerable fallback)
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                end_char = start_char + len(answer_text)
                sequence_ids = (
                    tokenized_examples.series_toc_ids(i)
                    if hasattr(tokenized_examples, "series_toc_ids")
                    else []
                )
                # Fallback to manual sequence identification
                sequence_ids = [
                    1 if id is not None and id > 0 else 0
                    for id in tokenized_examples.sequence_ids(i)
                ]

                # Find token span limits
                token_start_index = 0
                while (
                    token_start_index < len(offsets)
                    and sequence_ids[token_start_index] != 1
                ):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer window falls completely inside this token slice
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while (
                        token_end_index >= 0 and offsets[token_end_index][1] >= end_char
                    ):
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
