import os

import pandas as pd

from src.baseline_classifier import BaselineSpoilerClassifier
from src.baseline_heuristics import HeuristicSpoilerGenerator
from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite


def main():
    # File paths setup
    train_path = os.path.join("data", "train.jsonl")
    val_path = os.path.join("data", "validation.jsonl")
    output_path = "run.jsonl"

    # Step 2.1: Data Loading
    try:
        train_loader = JSONLLoader(train_path)
        train_df = train_loader.load_data()

        val_loader = JSONLLoader(val_path)
        val_df = val_loader.load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Ensure both train.jsonl and validation.jsonl are in the 'data/' folder.")
        return

    # Step 2.3: Task 1 Baseline Classifier
    classifier = BaselineSpoilerClassifier()
    classifier.fit(train_df)
    predicted_tags = classifier.predict(val_df)

    # Step 2.4: Task 2 Heuristic Generator
    generator = HeuristicSpoilerGenerator()
    predicted_spoilers = generator.generate(val_df, predicted_tags)

    # Step 2.5: Run Automated Evaluation Suite
    y_true_tags = (
        val_df["tags"].apply(lambda x: x[0] if isinstance(x, list) else x).tolist()
    )
    ground_truth_spoilers = val_df["spoiler"].tolist()

    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, predicted_tags)
    AdvancedEvaluationSuite.evaluate_task2(ground_truth_spoilers, predicted_spoilers)

    # Step 2.6: Formulating output artifact in SemEval compliant structure
    print(f"\nSaving final baseline execution file to {output_path}...")
    output_records = []
    for idx, row in val_df.iterrows():
        record = {
            "uuid": row["uuid"],
            "spoilerType": predicted_tags[idx],
            "spoiler": predicted_spoilers[idx],
        }
        output_records.append(record)

    output_df = pd.DataFrame(output_records)
    output_df.to_json(output_path, orient="records", lines=True)
    print("[SUCCESS] Baseline run accomplished safely.")


if __name__ == "__main__":
    main()
