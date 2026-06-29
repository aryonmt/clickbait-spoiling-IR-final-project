import argparse

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.qa_preprocessor import QAPreprocessor


def compute_qa_metrics(eval_pred):
    """Computes exact match and token-level F1 on token positions directly.

    Args:
        eval_pred: Evaluated predictions containing (logits, labels).

    Returns:
        dict: Calculated Exact Match (EM) and token-level F1 scores.
    """
    start_logits, end_logits = eval_pred.predictions
    start_positions, end_positions = eval_pred.label_ids

    pred_start = np.argmax(start_logits, axis=-1)
    pred_end = np.argmax(end_logits, axis=-1)

    exact_match = 0
    f1_scores = []
    total = len(start_positions)

    for i in range(total):
        s_true, e_true = start_positions[i], end_positions[i]
        s_pred, e_pred = pred_start[i], pred_end[i]

        # 1. Exact Match (Token boundary overlap)
        if s_true == s_pred and e_true == e_pred:
            exact_match += 1

        # 2. Token-level F1 calculation
        true_span = set(range(s_true, e_true + 1))
        pred_span = set(range(s_pred, e_pred + 1)) if e_pred >= s_pred else set()

        if not true_span and not pred_span:
            f1_scores.append(1.0)
            continue

        intersection = true_span.intersection(pred_span)
        if not intersection:
            f1_scores.append(0.0)
            continue

        precision = len(intersection) / len(pred_span)
        recall = len(intersection) / len(true_span)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return {
        "exact_match": exact_match / total,
        "token_f1": np.mean(f1_scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Transformer model for Task 2 (Spoiler Text Extraction)"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to train JSONL file"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument("--model_name", type=str, help="Base QA model from HF Hub")
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save QA checkpoints"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    args = parser.parse_args()

    config = PipelineConfig()
    if args.model_name:
        config.task2_model_name = args.model_name
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.lr:
        config.task2_lr = args.lr
    if args.epochs:
        config.task2_epochs = args.epochs

    # Load Datasets
    print("Loading datasets for QA Extraction...")
    train_df = JSONLLoader(args.train_path).load_data()
    val_df = JSONLLoader(args.val_path).load_data()

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Prepare features utilizing configurations
    print(f"Tokenizing and mapping text spans using: {config.task2_model_name}")
    preprocessor = QAPreprocessor(
        model_name=config.task2_model_name,
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

    model = AutoModelForQuestionAnswering.from_pretrained(config.task2_model_name)

    # Select best model using real token_f1 performance metrics
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.task2_lr,
        per_device_train_batch_size=config.task2_batch_size,
        per_device_eval_batch_size=config.task2_batch_size,
        num_train_epochs=config.task2_epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Changed to token-F1
        greater_is_better=True,  # Higher token-F1 is better
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
        compute_metrics=compute_qa_metrics,  # Pass the QA metrics function
    )

    print("\n=== LAUNCHING EXTRACTIVE TRANSFORMER QA TRAIN SEQUENCE ===")
    trainer.train()

    print(f"Saving optimized Task 2 QA model weights to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
