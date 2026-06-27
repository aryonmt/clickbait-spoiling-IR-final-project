import argparse

import pandas as pd

from src.baseline_heuristics import HeuristicSpoilerGenerator
from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite
from src.inference_transformer import TransformerInferencePipeline
from src.utils import extract_primary_tag


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using trained Transformer for Task 1"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation data asset"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to trained model checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="run_transformer_baseline.jsonl",
        help="Output artifact path",
    )
    args = parser.parse_args()

    try:
        val_loader = JSONLLoader(file_path=args.val_path)
        val_df = val_loader.load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Step 1: Generate high-quality tags using our 61.82% Transformer model
    inference_pipeline = TransformerInferencePipeline(
        checkpoint_dir=args.checkpoint_dir
    )
    transformer_tags = inference_pipeline.predict(val_df)

    # Step 2: Feed these premium tags into the Task 2 Heuristic generator
    generator = HeuristicSpoilerGenerator()
    predicted_spoilers = generator.generate(val_df, transformer_tags)

    # Step 3: Run full suite evaluation
    y_true_tags = val_df["tags"].apply(extract_primary_tag).tolist()
    ground_truth_spoilers = val_df["spoiler"].tolist()

    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, transformer_tags)
    AdvancedEvaluationSuite.evaluate_task2(ground_truth_spoilers, predicted_spoilers)

    # Step 4: Export standardized submission artifact
    print(f"\nSaving integrated pipeline execution file to {args.output_path}...")
    output_records = []
    for idx, row in val_df.iterrows():
        record = {
            "uuid": row["uuid"],
            "spoilerType": transformer_tags[idx],
            "spoiler": predicted_spoilers[idx],
        }
        output_records.append(record)

    output_df = pd.DataFrame(output_records)
    output_df.to_json(args.output_path, orient="records", lines=True)
    print("[SUCCESS] Transformer-driven pipeline run completed safely.")


if __name__ == "__main__":
    main()
