from typing import List

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import SPOILER_LABELS

# Silent download of METEOR dependencies to ensure smooth execution
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.translate.meteor_score import meteor_score


class AdvancedEvaluationSuite:
    """Comprehensive evaluation metrics suite for both challenge tasks."""

    @staticmethod
    def evaluate_task1(y_true: List[str], y_pred: List[str]) -> None:
        """Computes and displays classification metrics for Task 1.

        Args:
            y_true (List[str]): Ground truth labels.
            y_pred (List[str]): Model predictions.
        """
        print("\n=== TASK 1: CLASSIFICATION PERFORMANCE ===")
        print(classification_report(y_true, y_pred, labels=SPOILER_LABELS, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=SPOILER_LABELS))

    @staticmethod
    def evaluate_task2(
        ground_truths: List[List[str]], predictions: List[List[str]]
    ) -> None:
        """Computes BLEU, ROUGE-L, and METEOR scores for the generated spoilers.

        Args:
            ground_truths (List[List[str]]): Target human spoilers.
            predictions (List[List[str]]): Generated baseline spoilers.
        """
        print("\n=== TASK 2: TEXT GENERATION PERFORMANCE ===")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleu_scores = []
        rouge_l_scores = []
        meteor_scores = []

        for gt, pred in zip(ground_truths, predictions):
            gt_text = " ".join(gt).lower() if isinstance(gt, list) else str(gt).lower()
            pred_text = (
                " ".join(pred).lower() if isinstance(pred, list) else str(pred).lower()
            )

            # Calculate ROUGE-L
            rouge_score = scorer.score(gt_text, pred_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)

            # Calculate BLEU-4
            gt_tokens = gt_text.split()
            pred_tokens = pred_text.split()

            if not pred_tokens or not gt_tokens:
                bleu_scores.append(0.0)
                meteor_scores.append(0.0)
                continue

            score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
            bleu_scores.append(score)

            # Calculate METEOR using standard NLTK interface
            m_score = meteor_score([gt_tokens], pred_tokens)
            meteor_scores.append(m_score)

        print(f"Mean BLEU Score: {np.mean(bleu_scores):.4f}")
        print(f"Mean ROUGE-L Score: {np.mean(rouge_l_scores):.4f}")
        print(f"Mean METEOR Score: {np.mean(meteor_scores):.4f}")

        # Compute semantic BERTScore if available
        try:
            import bert_score

            references_str = [
                " ".join(gt) if isinstance(gt, list) else str(gt)
                for gt in ground_truths
            ]
            candidates_str = [
                " ".join(pred) if isinstance(pred, list) else str(pred)
                for pred in predictions
            ]

            if references_str and candidates_str:
                _, _, F1 = bert_score.score(
                    candidates_str, references_str, lang="en", verbose=False
                )
                mean_bert = float(F1.mean().item())
                print(f"Mean BERTScore:    {mean_bert:.4f}")
        except ImportError:
            print(
                "[INFO] 'bert-score' library not installed. Skipping semantic BERTScore display."
            )
