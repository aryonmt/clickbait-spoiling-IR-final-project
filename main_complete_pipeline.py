import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite
from src.utils import extract_primary_tag


class EndToEndClickbaitPipeline:
    def __init__(self, task1_dir: str, task2_dir: str, max_length: int = 512):
        print("[INFO] Initializing End-to-End Deep Transformer Pipeline...")
        # Load Task 1 (Classifier)
        self.t1_tokenizer = AutoTokenizer.from_pretrained(task1_dir)
        self.t1_model = AutoModelForSequenceClassification.from_pretrained(task1_dir)
        self.t1_id_to_label = {0: "phrase", 1: "passage", 2: "multi"}
        self.t1_trainer = Trainer(
            model=self.t1_model, processing_class=self.t1_tokenizer
        )

        # Load Task 2 (QA Extractor)
        self.t2_tokenizer = AutoTokenizer.from_pretrained(task2_dir)
        self.t2_model = AutoModelForQuestionAnswering.from_pretrained(task2_dir)

        self.max_length = max_length

    def predict_pipeline(self, df: pd.DataFrame):
        # --- STEP 1: Task 1 Classification ---
        processed_t1 = pd.DataFrame()
        processed_t1["post"] = df["postText"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
        processed_t1["article"] = df["targetParagraphs"].apply(
            lambda x: " ".join(x[:8]) if isinstance(x, list) else ""
        )
        processed_t1["label"] = 0

        t1_ds = Dataset.from_pandas(processed_t1)
        t1_tokenized = t1_ds.map(
            lambda x: self.t1_tokenizer(
                x["post"], x["article"], truncation=True, max_length=self.max_length
            ),
            batched=True,
            verbose=False,
        )

        print("Executing Task 1: Predicting Spoiler Types...")
        t1_preds = self.t1_trainer.predict(t1_tokenized)
        predicted_t1_ids = np.argmax(t1_preds.predictions, axis=-1)
        predicted_tags = [self.t1_id_to_label[idx] for idx in predicted_t1_ids]

        # --- STEP 2: Task 2 Extractive QA ---
        print("Executing Task 2: Extracting Precise Spoiler Spans...")
        predicted_spoilers = []

        self.t2_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t2_model.to(device)

        for idx, row in df.iterrows():
            question = (
                " ".join(row["postText"])
                if isinstance(row["postText"], list)
                else str(row["postText"])
            )
            context = (
                " ".join(row["targetParagraphs"])
                if isinstance(row["targetParagraphs"], list)
                else str(row["targetParagraphs"])
            )

            inputs = self.t2_tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.t2_model(**inputs)

            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()

            start_idx = int(np.argmax(start_logits))
            end_idx = int(np.argmax(end_logits))

            # Reconstruct string from token span safely
            input_ids = inputs["input_ids"][0].cpu().numpy()
            if end_idx >= start_idx:
                extracted_tokens = input_ids[start_idx : end_idx + 1]
                pred_text = self.t2_tokenizer.decode(
                    extracted_tokens, skip_special_tokens=True
                ).strip()
            else:
                pred_text = ""

            # Fallback optimization if QA model returns empty string
            if not pred_text and len(row["targetParagraphs"]) > 0:
                pred_text = row["targetParagraphs"][0]

            predicted_spoilers.append([pred_text] if pred_text else [])

        return predicted_tags, predicted_spoilers


def main():
    parser = argparse.ArgumentParser(
        description="Run complete dual-transformer neural pipeline evaluation"
    )
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--t1_checkpoint", type=str, required=True)
    parser.add_argument("--t2_checkpoint", type=str, required=True)
    parser.add_argument(
        "--output_path", type=str, default="final_complete_submission.jsonl"
    )
    args = parser.parse_args()

    val_df = JSONLLoader(args.val_path).load_data()
    pipeline = EndToEndClickbaitPipeline(
        task1_dir=args.t1_checkpoint, task2_dir=args.t2_checkpoint
    )

    pred_tags, pred_spoilers = pipeline.predict_pipeline(val_df)

    # Full Suite Evaluation
    y_true_tags = val_df["tags"].apply(extract_primary_tag).tolist()
    ground_truth_spoilers = val_df["spoiler"].tolist()

    print("\n=== FINAL DISSERTATION SYSTEM REPORT ===")
    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, pred_tags)
    AdvancedEvaluationSuite.evaluate_task2(ground_truth_spoilers, pred_spoilers)

    # Save Output
    output_records = [
        {"uuid": r["uuid"], "spoilerType": pred_tags[i], "spoiler": pred_spoilers[i]}
        for i, r in val_df.iterrows()
    ]
    pd.DataFrame(output_records).to_json(args.output_path, orient="records", lines=True)
    print(f"[SUCCESS] Deep integrated run file saved securely at {args.output_path}")


if __name__ == "__main__":
    main()
