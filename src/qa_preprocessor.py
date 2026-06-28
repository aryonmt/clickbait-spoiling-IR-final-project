from transformers import AutoTokenizer

from src.logging_setup import setup_logger
from src.utils import extract_primary_tag


class QAPreprocessor:
    """Preprocesses Clickbait Spoiling datasets into standardized extractive
    Question-Answering token spans with active fallback monitoring.
    """

    def __init__(self, model_name: str, max_length: int = 512, doc_stride: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.logger = setup_logger("qa_preprocessor")

        # Diagnostics metrics for span extraction analysis
        self.total_processed = 0
        self.fallback_count = 0
        self.tag_fallbacks = {"phrase": 0, "passage": 0, "multi": 0}
        self.tag_totals = {"phrase": 0, "passage": 0, "multi": 0}

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

        # Safe extraction of diagnostic fields to handle missing test-time columns
        uuids = examples.get("uuid", ["unknown"] * len(questions))
        raw_tags = examples.get("tags", [["phrase"]] * len(questions))

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Get the original sample index
            sample_index = sample_mapping[i]
            context = contexts[sample_index]

            # Extract spoiler list and reconstruct primary target text
            spoilers = (
                examples["spoiler"][sample_index] if "spoiler" in examples else []
            )
            answer_text = (
                spoilers[0]
                if (isinstance(spoilers, list) and len(spoilers) > 0)
                else ""
            )

            # Extract labels for type-specific monitoring
            tag = extract_primary_tag(raw_tags[sample_index])
            self.tag_totals[tag] = self.tag_totals.get(tag, 0) + 1
            self.total_processed += 1

            # Locate character span in text
            start_char = context.find(answer_text) if answer_text else -1

            if start_char == -1:
                # Anchor to CLS token (unanswerable fallback)
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

                self.fallback_count += 1
                self.tag_fallbacks[tag] = self.tag_fallbacks.get(tag, 0) + 1
                self.logger.warning(
                    f"Spoiler text not found in context for uuid: {uuids[sample_index]}. "
                    f"Falling back to CLS index. Tag: {tag}"
                )
            else:
                end_char = start_char + len(answer_text)

                # Fetch sequence IDs to identify context tokens safely (Removed dead series_toc_ids block)
                sequence_ids = [
                    1 if val is not None and val > 0 else 0
                    for val in tokenized_examples.sequence_ids(i)
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

                    self.fallback_count += 1
                    self.tag_fallbacks[tag] = self.tag_fallbacks.get(tag, 0) + 1
                    self.logger.warning(
                        f"Answer span outside active token window for uuid: {uuids[sample_index]}. "
                        f"Falling back to CLS index. Tag: {tag}"
                    )
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

        # Print structured statistical evaluation inside logs
        if self.total_processed > 0:
            phrase_rate = (
                self.tag_fallbacks["phrase"] / max(1, self.tag_totals.get("phrase", 0))
            ) * 100
            passage_rate = (
                self.tag_fallbacks["passage"]
                / max(1, self.tag_totals.get("passage", 0))
            ) * 100
            multi_rate = (
                self.tag_fallbacks["multi"] / max(1, self.tag_totals.get("multi", 0))
            ) * 100

            self.logger.info(
                f"Data preprocessing chunk complete. Statistics:\n"
                f"  - Total processed: {self.total_processed}\n"
                f"  - Overall CLS Fallbacks: {self.fallback_count} ({self.fallback_count / self.total_processed * 100:.2f}%)\n"
                f"  - Phrase fallbacks: {self.tag_fallbacks['phrase']}/{self.tag_totals.get('phrase', 0)} ({phrase_rate:.2f}%)\n"
                f"  - Passage fallbacks: {self.tag_fallbacks['passage']}/{self.tag_totals.get('passage', 0)} ({passage_rate:.2f}%)\n"
                f"  - Multi fallbacks: {self.tag_fallbacks['multi']}/{self.tag_totals.get('multi', 0)} ({multi_rate:.2f}%)"
            )

        return tokenized_examples
