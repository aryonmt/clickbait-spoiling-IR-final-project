import argparse

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.transformer_classifier import TransformerSpoilerClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Encapsulated Training Framework for Clickbait Challenge Task 1"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training data asset"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation data asset"
    )
    parser.add_argument(
        "--model_name", type=str, help="HuggingFace model architecture blueprint"
    )
    parser.add_argument("--lr", type=float, help="Learning rate constraints")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save checkpoint weights"
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    args = parser.parse_args()

    # Load and map command-line arguments to global PipelineConfig
    config = PipelineConfig()
    if args.model_name:
        config.task1_model_name = args.model_name
    if args.lr:
        config.task1_lr = args.lr
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.task1_epochs = args.epochs

    try:
        train_loader = JSONLLoader(file_path=args.train_path)
        train_df = train_loader.load_data()

        val_loader = JSONLLoader(file_path=args.val_path)
        val_df = val_loader.load_data()
    except FileNotFoundError as e:
        print(f"[ERROR] Ingestion crashed: {e}")
        return

    # Initialize classifier utilizing the structured configuration
    classifier = TransformerSpoilerClassifier(config=config)
    classifier.train_pipeline(train_df=train_df, val_df=val_df)


if __name__ == "__main__":
    main()
