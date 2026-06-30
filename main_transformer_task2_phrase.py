import argparse
import os

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.qa_preprocessor import QAPreprocessor


def compute_qa_metrics(eval_pred):
    """Computes exact match and token-level F1 on token positions directly."""
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    pred_start = np.argmax(start_logits, axis=-1)
    pred_end = np.argmax(end_logits, axis=-1)

    exact_match = 0
    f1_scores = []
    answerable_exact_match = 0
    answerable_f1_scores = []
    total = len(start_positions)

    for i in range(total):
        s_true, e_true = start_positions[i], end_positions[i]
        s_pred, e_pred = pred_start[i], pred_end[i]

        is_em = s_true == s_pred and e_true == e_pred
        if is_em:
            exact_match += 1

        true_span = set(range(s_true, e_true + 1))
        pred_span = set(range(s_pred, e_pred + 1)) if e_pred >= s_pred else set()

        if not true_span and not pred_span:
            f1 = 1.0
        elif not true_span or not pred_span:
            f1 = 0.0
        else:
            intersection = true_span.intersection(pred_span)
            if not intersection:
                f1 = 0.0
            else:
                precision = len(intersection) / len(pred_span)
                recall = len(intersection) / len(true_span)
                f1 = (2 * precision * recall) / (precision + recall)

        f1_scores.append(f1)

        if s_true != 0:
            answerable_f1_scores.append(f1)
            if is_em:
                answerable_exact_match += 1

    mean_answerable_f1 = np.mean(answerable_f1_scores) if answerable_f1_scores else 0.0
    mean_answerable_em = (
        answerable_exact_match / len(answerable_f1_scores)
        if answerable_f1_scores
        else 0.0
    )

    return {
        "exact_match": exact_match / total,
        "token_f1": np.mean(f1_scores),
        "answerable_exact_match": mean_answerable_em,
        "answerable_token_f1": mean_answerable_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer model for Phrase Spoilers Extraction"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to train JSONL file"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument("--model_name", type=str, help="Base QA model override")
    parser.add_argument("--output_dir", type=str, help="Output directory override")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.model_name:
        config.task2_phrase_model_name = args.model_name
    if args.output_dir:
        config.task2_phrase_output_dir = args.output_dir
    if args.lr:
        config.task2_lr = args.lr
    if args.epochs:
        config.task2_phrase_epochs = args.epochs

    set_seed(config.seed)

    # Load and filter dataset for PHRASE only
    print("Loading and filtering phrase datasets...")
    train_loader = JSONLLoader(args.train_path).filter_by_tag("phrase")
    val_loader = JSONLLoader(args.val_path).filter_by_tag("phrase")

    train_dataset = Dataset.from_pandas(train_loader.df)
    val_dataset = Dataset.from_pandas(val_loader.df)

    # Preprocess with type-specific max answer length
    preprocessor = QAPreprocessor(
        model_name=config.task2_phrase_model_name,
        max_length=config.task2_max_length,
        doc_stride=config.task2_doc_stride,
    )

    train_tokenized = train_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = val_dataset.map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    preprocessor.export_stats(config.task2_phrase_output_dir)

    model = AutoModelForQuestionAnswering.from_pretrained(
        config.task2_phrase_model_name
    )

    # Phase 2.4: Configure training arguments with strict save_total_limit
    training_args = TrainingArguments(
        output_dir=config.task2_passage_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.task2_lr,
        per_device_train_batch_size=config.task2_batch_size,
        per_device_eval_batch_size=config.task2_batch_size,
        num_train_epochs=config.task2_passage_epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="answerable_token_f1",
        greater_is_better=True,
        save_total_limit=1,  # Keep only the single best checkpoint to save disk space
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DefaultDataCollator(),
        processing_class=preprocessor.tokenizer,
        compute_metrics=compute_qa_metrics,
    )

    print("\n=== LAUNCHING EXTRACTIVE TRANSFORMER PASSAGE QA TRAIN SEQUENCE ===")
    trainer.train()

    print(
        f"Saving optimized Passage QA model weights to {config.task2_passage_output_dir}..."
    )
    trainer.save_model(config.task2_passage_output_dir)

    # Clean intermediate checkpoint folders
    import glob
    import shutil

    for folder in glob.glob(
        os.path.join(config.task2_passage_output_dir, "checkpoint-*")
    ):
        shutil.rmtree(folder, ignore_errors=True)
        print(f"[CLEANUP] Pruned intermediate checkpoint: {folder}")

    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
