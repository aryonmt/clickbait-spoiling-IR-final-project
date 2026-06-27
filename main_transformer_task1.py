import argparse

from src.data_loader import JSONLLoader
from src.transformer_classifier import TransformerSpoilerClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Encapsulated Training Framework for Clickbait Challenge Task 1"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training data asset",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to validation data asset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="HuggingFace model architecture blueprint",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate constraints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_task1",
        help="Directory to save checkpoint weights",
    )
    args = parser.parse_args()

    # Step 1: Secure Data Ingestion via object-oriented loaders
    try:
        train_loader = JSONLLoader(file_path=args.train_path)
        train_df = train_loader.load_data()

        val_loader = JSONLLoader(file_path=args.val_path)
        val_df = val_loader.load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] Ingestion crashed: {e}")
        return

    # Step 2: Trigger encapsulated modeling architecture
    classifier = TransformerSpoilerClassifier(model_name=args.model_name, lr=args.lr)
    classifier.train_pipeline(
        train_df=train_df, val_df=val_df, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
