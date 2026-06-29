from typing import List

import nltk

# Ensure both punkt and punkt_tab are safely downloaded for newer NLTK versions
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class RetrievalSpoilerGenerator:
    """Generates multi-type spoilers by dynamically retrieving and ranking sentences

    from the article based on lexical similarity to the clickbait post.
    """

    @staticmethod
    def _compute_jaccard(tokens_a: set, tokens_b: set) -> float:
        """Calculates Jaccard Similarity between two sets of tokens."""
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def generate_multi_spoiler(
        cls, post_text: str, paragraphs: List[str], top_k: int = 3
    ) -> List[str]:
        """Selects top-k sentences across all paragraphs ranked by lexical
        similarity to the clickbait post, for multi-type spoiler generation.

        Args:
            post_text: The clickbait headline string.
            paragraphs: All paragraphs from the source article.
            top_k: Number of highest ranking sentences to extract.

        Returns:
            List[str]: Extracted top-k spoiler sentences.
        """
        if not paragraphs:
            return []

        # Tokenize post into lower-cased words for uniform matching
        post_words = set(post_text.lower().split())
        if not post_words:
            return [paragraphs[0]] if paragraphs else []

        scored_sentences = []
        sentence_idx = 0

        # Break all paragraphs into sentences and score each
        for para_idx, para in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(para)
            for sent in sentences:
                sent_words = set(sent.lower().split())
                score = cls._compute_jaccard(post_words, sent_words)

                # Keep track of sentence, its score, and original order
                scored_sentences.append(
                    {
                        "text": sent.strip(),
                        "score": score,
                        "para_idx": para_idx,
                        "order": sentence_idx,
                    }
                )
                sentence_idx += 1

        # Rank by score descending, then maintain original document order for top sentences
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        top_sentences = scored_sentences[:top_k]

        # Sort the selected sentences back to their original document order to maintain coherence
        top_sentences.sort(key=lambda x: x["order"])

        return [item["text"] for item in top_sentences]
