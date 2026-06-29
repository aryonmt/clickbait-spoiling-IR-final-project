import os

from src.data_loader import JSONLLoader
from src.eda_analysis import TextLengthAnalyzer


def main():
    # Fixed naming alignment to point to the real file name
    train_path = os.path.join("data", "train.jsonl")

    try:
        loader = JSONLLoader(file_path=train_path)
        train_df = loader.load_data()
        loader.run_sanity_checks()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(
            "Please ensure you have placed 'train.jsonl' inside the 'data/' directory."
        )
        return

    analyzer = TextLengthAnalyzer(df=train_df, output_dir="reports")
    analyzer.process_lengths()
    analyzer.display_statistics()
    analyzer.plot_distributions()
    analyzer.analyze_class_balance()
    analyzer.analyze_spoiler_overlap()
    analyzer.analyze_positional_bias()
    analyzer.analyze_tag_specific_lengths()
    analyzer.analyze_oracle_retrieval_bound()


if __name__ == "__main__":
    main()
