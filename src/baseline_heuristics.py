from typing import List

import nltk

# Ensure both punkt and punkt_tab are safely downloaded for newer NLTK versions
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


class HeuristicSpoilerGenerator:
    """Generates spoilers using rule-based heuristics based on predicted spoiler

    types and empirical length constraints.
    """

    @staticmethod
    def _extract_heuristic(paragraphs: List[str], tag: str) -> List[str]:
        """Applies heuristic rules on target paragraphs to extract a baseline

        spoiler.

        Args:
            paragraphs (List[str]): List of paragraphs from the article.
            tag (str): Predicted tag ('phrase', 'passage', or 'multi').

        Returns:
            List[str]: Extracted baseline spoiler tokens/phrases.
        """
        if not paragraphs or not isinstance(paragraphs, list):
            return [""]

        first_para = paragraphs[0]

        if tag == "phrase":
            # Target phrase length: ~3-5 words from the very beginning of the article
            words = first_para.split()
            return [" ".join(words[:5])]

        elif tag == "passage":
            # Target passage length: return the complete first paragraph
            return [first_para]

        elif tag == "multi":
            # Target multi length: combine the first sentences of the first 3 paragraphs
            extracted_sentences = []
            for para in paragraphs[:3]:
                sentences = nltk.sent_tokenize(para)
                if sentences:
                    extracted_sentences.append(sentences[0])
            return extracted_sentences

        return [first_para]

    def generate(self, df: List[str], predicted_tags: List[str]) -> List[List[str]]:
        """Orchestrates heuristic generation across the entire dataset.

        Args:
            df (pd.DataFrame): Input dataset.
            predicted_tags (List[str]): Predicted classes from Task 1.

        Returns:
            List[List[str]]: Generated list of spoilers.
        """
        print("Generating spoilers using heuristic rules...")
        generated_spoilers = []
        for idx, row in df.iterrows():
            paragraphs = row["targetParagraphs"]
            pred_tag = predicted_tags[idx]
            spoiler = self._extract_heuristic(paragraphs, pred_tag)
            generated_spoilers.append(spoiler)
        return generated_spoilers
