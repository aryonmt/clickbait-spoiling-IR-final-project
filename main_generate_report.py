import argparse

from src.visualization_suite import EvaluationReportBuilder


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Evaluation Report")
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to predictions JSONL file",
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports", help="Root directory for reports"
    )
    args = parser.parse_args()

    # Orchestrate visual reports generation
    builder = EvaluationReportBuilder(
        predictions_path=args.predictions_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
    )
    builder.build_report()


if __name__ == "__main__":
    main()
