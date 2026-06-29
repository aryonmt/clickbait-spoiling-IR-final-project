import argparse
import os

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from src.baseline_heuristics import HeuristicSpoilerGenerator
from src.data_loader import JSONLLoader
from src.retrieval_generator import RetrievalSpoilerGenerator
from src.utils import extract_primary_tag


def evaluate_predictions(ground_truths: list, predictions: list):
    """Computes BLEU and ROUGE-L mean scores."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    bleu_scores = []
    rouge_l_scores = []

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
            continue

        score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother)
        bleu_scores.append(score)

    return np.mean(bleu_scores), np.mean(rouge_l_scores)


def main():
    parser = argparse.ArgumentParser(
        description="A/B Test Multi-type Spoiler Generation Strategies"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/validation.jsonl",
        help="Path to validation data",
    )
    args = parser.parse_args()

    if not os.path.exists(args.val_path):
        print(f"[ERROR] Validation file not found at: {args.val_path}")
        return

    # 1. Load Validation Data
    loader = JSONLLoader(args.val_path)
    df = loader.load_data()

    # 2. Filter strictly for 'multi' tag cases
    multi_df = df[df["tags"].apply(extract_primary_tag) == "multi"].copy()
    total_cases = len(multi_df)
    print(f"\nEvaluating {total_cases} multi-type spoiler cases...")

    ground_truths = multi_df["spoiler"].tolist()

    # Strategy A: Baseline Heuristics (first sentences of top 3 paragraphs)
    predicted_spoilers_a = []
    for _, row in multi_df.iterrows():
        paragraphs = row["targetParagraphs"]
        spoilers_a = HeuristicSpoilerGenerator._extract_heuristic(paragraphs, "multi")
        predicted_spoilers_a.append(spoilers_a)

    # Strategy B: New Retrieval Spoiler Generator (Top-3 lexically ranked sentences)
    predicted_spoilers_b = []
    for _, row in multi_df.iterrows():
        post_text = (
            " ".join(row["postText"])
            if isinstance(row["postText"], list)
            else str(row["postText"])
        )
        paragraphs = row["targetParagraphs"]
        spoilers_b = RetrievalSpoilerGenerator.generate_multi_spoiler(
            post_text, paragraphs, top_k=3
        )
        predicted_spoilers_b.append(spoilers_b)

    # 3. Evaluate Metrics
    bleu_a, rouge_a = evaluate_predictions(ground_truths, predicted_spoilers_a)
    bleu_b, rouge_b = evaluate_predictions(ground_truths, predicted_spoilers_b)

    print("\n==========================================")
    print("A/B TEST REPORT: MULTI-TYPE GENERATION STRATEGY")
    print("==========================================")
    print("Strategy A: Baseline Heuristics")
    print(f"  - Mean BLEU:    {bleu_a:.4f}")
    print(f"  - Mean ROUGE-L: {rouge_a:.4f}")
    print("------------------------------------------")
    print("Strategy B: New Retrieval Spoiler Generator (Jaccard Rank)")
    print(f"  - Mean BLEU:    {bleu_b:.4f}")
    print(f"  - Mean ROUGE-L: {rouge_b:.4f}")
    print("==========================================")

    # Conclusion
    if bleu_b > bleu_a or rouge_b > rouge_a:
        improvement_bleu = ((bleu_b - bleu_a) / max(0.01, bleu_a)) * 100
        improvement_rouge = ((rouge_b - rouge_a) / max(0.01, rouge_a)) * 100
        print("[SUCCESS] Strategy B outperforms Strategy A!")
        print(f"  - BLEU Relative Improvement:    +{improvement_bleu:.2f}%")
        print(f"  - ROUGE-L Relative Improvement: +{improvement_rouge:.2f}%")
    else:
        print("[WARNING] Strategy B did not outperform Strategy A on these metrics.")


if __name__ == "__main__":
    main()
