import os

from src.data_loader import JSONLLoader
from src.eda_analysis import TextLengthAnalyzer


def main():
    # Define file paths
    # Assuming data is downloaded inside a folder named 'data' in the root directory
    train_path = os.path.join("data", "train.jsonl")

    # Step 1.1: Data Ingestion & Sanity Checks
    try:
        loader = JSONLLoader(file_path=train_path)
        train_df = loader.load_data()
        loader.run_sanity_checks()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(
            "Please ensure you have placed 'training.jsonl' inside the 'data/'"
            " directory."
        )
        return

    # Step 1.2: Sequence Length Distribution Analysis
    analyzer = TextLengthAnalyzer(df=train_df)
    analyzer.process_lengths()
    analyzer.display_statistics()
    analyzer.plot_distributions()


if __name__ == "__main__":
    main()
