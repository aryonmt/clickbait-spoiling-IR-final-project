import argparse

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

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.task2_lr,
        per_device_train_batch_size=config.task2_batch_size,
        per_device_eval_batch_size=config.task2_batch_size,
        num_train_epochs=config.task2_epochs,
        weight_decay=config.task2_weight_decay,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
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
    )

    print("\n=== LAUNCHING EXTRACTIVE TRANSFORMER QA TRAIN SEQUENCE ===")
    trainer.train()

    print(f"Saving optimized Task 2 QA model weights to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    print("[SUCCESS] Training execution routine complete.")


if __name__ == "__main__":
    main()
