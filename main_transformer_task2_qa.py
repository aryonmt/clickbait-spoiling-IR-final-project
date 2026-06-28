import argparse

from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.data_loader import JSONLLoader
from src.qa_preprocessor import QAPreprocessor


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
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepset/roberta-base-squad2",
        help="Base QA model from HF Hub",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_qa",
        help="Directory to save QA checkpoints",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    # 1. Load Datasets
    print("Loading datasets for QA Extraction...")
    train_df = JSONLLoader(args.train_path).load_data()
    val_df = JSONLLoader(args.val_path).load_data()

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 2. Map spans natively via token grids
    print(f"Tokenizing and mapping text spans using: {args.model_name}")
    preprocessor = QAPreprocessor(model_name=args.model_name)

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

    # 3. Load pre-configured QA Architecture
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # 4. Enforce strict training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,  # Speed up training via Kaggle T4 GPU
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DefaultDataCollator(),
        processing_class=preprocessor.tokenizer,
    )

    print("\n=== LAUNCHING EXTRACTIVE TRANSFORMER QA TRAIN SEQUENCE ===")
    trainer.train()

    print(f"Saving optimized Task 2 QA model weights to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
