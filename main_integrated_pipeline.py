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

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.evaluation import AdvancedEvaluationSuite
from src.logging_setup import setup_logger
from src.qa_inference import predict_best_span
from src.retrieval_generator import RetrievalSpoilerGenerator
from src.utils import extract_primary_tag

# Set up integrated log tracking
logger = setup_logger("integrated_pipeline")


class ClickbaitIntegratedPipeline:
    """Orchestrates combined inference using sequential Task 1 sequence classification

    and Task 2 token span extraction.
    """

    def __init__(self, task1_dir: str, task2_dir: str, config: PipelineConfig):
        logger.info("Initializing Clickbait Spoiler Integrated Pipeline...")
        self.config = config
        self.max_length = config.task2_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Task 1 Model
        logger.info(f"Loading Task 1 classification weights from: {task1_dir}")
        self.t1_tokenizer = AutoTokenizer.from_pretrained(task1_dir)
        self.t1_model = AutoModelForSequenceClassification.from_pretrained(task1_dir)
        self.t1_model.to(self.device)
        self.t1_model.eval()
        self.t1_id_to_label = {0: "phrase", 1: "passage", 2: "multi"}

        # Load Task 2 QA Model
        logger.info(f"Loading Task 2 extractive QA weights from: {task2_dir}")
        self.t2_tokenizer = AutoTokenizer.from_pretrained(task2_dir)
        self.t2_model = AutoModelForQuestionAnswering.from_pretrained(task2_dir)
        self.t2_model.to(self.device)
        self.t2_model.eval()

    def run_inference(self, df: pd.DataFrame):
        """Executes the complete integrated evaluation flow across the input dataframe.

        Args:
            df (pd.DataFrame): Input dataframe containing challenge samples.

        Returns:
            tuple: (predicted_tags, predicted_spoilers, confidence_scores) lists.
        """
        # --- Task 1: Predict Spoiler Types ---
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
        )

        logger.info("Running Task 1 Model Inference...")
        trainer = Trainer(model=self.t1_model, processing_class=self.t1_tokenizer)
        t1_preds = trainer.predict(t1_tokenized)

        # Calculate softmax probabilities to extract confidence metrics (Phase 4)
        logits_tensor = torch.tensor(t1_preds.predictions)
        probs = torch.nn.functional.softmax(logits_tensor, dim=-1).numpy()

        predicted_t1_ids = np.argmax(t1_preds.predictions, axis=-1)
        predicted_tags = [self.t1_id_to_label[idx] for idx in predicted_t1_ids]

        # Save confidence score of the predicted class for each prediction
        confidence_scores = [
            float(probs[i, idx]) for i, idx in enumerate(predicted_t1_ids)
        ]

        # --- Task 2: Robust Spoiler Span Extraction ---
        logger.info("Running Task 2 Model Inference with Safe Boundary Mapping...")
        predicted_spoilers = []

        for idx, row in df.iterrows():
            pred_tag = predicted_tags[idx]

            if pred_tag == "multi":
                # Route 'multi' classes to custom lexically ranked Retrieval Spoiler Generator
                post_text = (
                    " ".join(row["postText"])
                    if isinstance(row["postText"], list)
                    else str(row["postText"])
                )
                paragraphs = row["targetParagraphs"]
                spoiler = RetrievalSpoilerGenerator.generate_multi_spoiler(
                    post_text=post_text,
                    paragraphs=paragraphs,
                    top_k=self.config.task2_multi_top_k,
                    method=self.config.task2_multi_method,
                )
            else:
                # Route 'phrase' and 'passage' classes to Transformer QA Extractor using n-best sliding window
                paragraphs = row["targetParagraphs"]
                context_str = (
                    " ".join(paragraphs)
                    if isinstance(paragraphs, list)
                    else str(paragraphs)
                )
                post_str = (
                    " ".join(row["postText"])
                    if isinstance(row["postText"], list)
                    else str(row["postText"])
                )

                # Fetch max answer limits dynamically based on expected type class (phrase vs passage)
                max_ans_len = 10 if pred_tag == "phrase" else 150

                span_result = predict_best_span(
                    model=self.t2_model,
                    tokenizer=self.t2_tokenizer,
                    question=post_str,
                    context=context_str,
                    max_length=self.max_length,
                    doc_stride=self.config.task2_doc_stride,
                    max_answer_length=max_ans_len,
                    device=self.device,
                )
                pred_text = span_result["text"]

                # Safe fallback to the first paragraph if the QA extraction returns empty
                if not pred_text and len(row["targetParagraphs"]) > 0:
                    pred_text = row["targetParagraphs"][0]

                spoiler = [pred_text] if pred_text else []

            predicted_spoilers.append(spoiler)

        return predicted_tags, predicted_spoilers, confidence_scores


def main():
    parser = argparse.ArgumentParser(
        description="Run integrated clickbait spoiler prediction pipeline"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--t1_checkpoint",
        type=str,
        required=True,
        help="Path to Task 1 classifier weights",
    )
    parser.add_argument(
        "--t2_checkpoint",
        type=str,
        required=True,
        help="Path to Task 2 QA extractor weights",
    )
    parser.add_argument(
        "--output_path", type=str, default="final_integrated_submission.jsonl"
    )
    args = parser.parse_args()

    logger.info("Loading validation data...")
    val_df = JSONLLoader(args.val_path).load_data()

    # Initialize configuration with dynamically defined parameters
    config = PipelineConfig()

    pipeline = ClickbaitIntegratedPipeline(
        task1_dir=args.t1_checkpoint, task2_dir=args.t2_checkpoint, config=config
    )

    pred_tags, pred_spoilers, confidence_scores = pipeline.run_inference(val_df)

    # Multi-metric baseline performance scoring
    y_true_tags = val_df["tags"].apply(extract_primary_tag).tolist()
    ground_truth_spoilers = val_df["spoiler"].tolist()

    logger.info("Computing integrated pipeline metrics...")
    AdvancedEvaluationSuite.evaluate_task1(y_true_tags, pred_tags)
    AdvancedEvaluationSuite.evaluate_task2(ground_truth_spoilers, pred_spoilers)

    # Save standardized submission artifact for evaluations with confidence metrics
    logger.info(f"Saving final pipeline output file to {args.output_path}...")
    output_records = [
        {
            "uuid": row["uuid"],
            "spoilerType": pred_tags[i],
            "spoiler": pred_spoilers[i],
            "confidence": confidence_scores[i],
        }
        for i, row in val_df.iterrows()
    ]
    pd.DataFrame(output_records).to_json(args.output_path, orient="records", lines=True)
    logger.info("Pipeline executed successfully.")


if __name__ == "__main__":
    main()
