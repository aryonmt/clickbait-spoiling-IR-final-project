from typing import List

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import SPOILER_LABELS


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

    # Task 2 methods remain unchanged...

    @staticmethod
    def evaluate_task2(
        ground_truths: List[List[str]], predictions: List[List[str]]
    ) -> None:
        """Computes BLEU and ROUGE-L scores for the generated spoilers.

        Args:
            ground_truths (List[List[str]]): Target human spoilers.
            predictions (List[List[str]]): Generated baseline spoilers.
        """
        print("\n=== TASK 2: TEXT GENERATION PERFORMANCE ===")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        smoother = SmoothingFunction().method1

        bleu_scores = []
        rouge_l_scores = []

        for gt, pred in zip(ground_truths, predictions):
            gt_text = " ".join(gt).lower()
            pred_text = " ".join(pred).lower()

            # Calculate ROUGE-L
            rouge_score = scorer.score(gt_text, pred_text)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_score)

            # Calculate BLEU-4 (with word tokens)
            gt_tokens = gt_text.split()
            pred_tokens = pred_text.split()

            # Avoid mathematical errors on empty strings
            if not pred_tokens or not gt_tokens:
                bleu_scores.append(0.0)
                continue

            score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
            bleu_scores.append(score)

        print(f"Mean BLEU Score: {np.mean(bleu_scores):.4f}")
        print(f"Mean ROUGE-L Score: {np.mean(rouge_l_scores):.4f}")
