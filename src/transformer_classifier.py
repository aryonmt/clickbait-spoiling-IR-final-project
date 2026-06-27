import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.utils import extract_primary_tag

# Fix 1: Enforce global reproducibility from the start
set_seed(42)


class ClickbaitWeightedTrainer(Trainer):
    """Custom Trainer overriding compute_loss to handle class imbalance via

    Weighted Cross-Entropy.
    """

    def __init__(self, class_weights: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
            self.args.device
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Calculates loss applying the computed historical class weights."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class TransformerSpoilerClassifier:
    """A fully modular, encapsulated pipeline for training and evaluating

    Transformer-based sequence classifiers.
    """

    def __init__(
        self, model_name: str = "roberta-base", max_length: int = 512, lr: float = 1e-5
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.lr = lr
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.label_to_id = {"phrase": 0, "passage": 1, "multi": 2}
        self.id_to_label = {0: "phrase", 1: "passage", 2: "multi"}

    def _prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Processes dataframe fields into tokenized text-pairs using standard

        utilities.
        """
        processed_df = pd.DataFrame()
        processed_df["post"] = df["postText"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x)
        )
        processed_df["article"] = df["targetParagraphs"].apply(
            lambda x: " ".join(x[:8]) if isinstance(x, list) else ""
        )
        processed_df["label"] = (
            df["tags"].apply(extract_primary_tag).map(self.label_to_id)
        )

        dataset = Dataset.from_pandas(processed_df)

        def tokenize_func(examples):
            return self.tokenizer(
                examples["post"],
                examples["article"],
                truncation=True,
                max_length=self.max_length,
            )

        return dataset.map(tokenize_func, batched=True)

    def _calculate_class_weights(self, df: pd.DataFrame) -> list:
        """Computes inverse frequency class weights to address training dataset

        imbalance.
        """
        labels = df["tags"].apply(extract_primary_tag).map(self.label_to_id).values
        class_counts = np.bincount(labels, minlength=3)
        total_samples = len(labels)

        weights = total_samples / (3.0 * class_counts)
        return weights.tolist()

    def train_pipeline(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str
    ):
        """Executes full tokenization, model initialization, and full-precision

        weighted training loop.
        """
        print(f"Mapping and processing partitions for: {self.model_name}")
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)

        class_weights = self._calculate_class_weights(train_df)
        print(f"[INFO] Computed Dynamic Class Weights: {class_weights}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )

        def compute_metrics(eval_pred):
            metric = evaluate.load("f1")
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(
                predictions=predictions, references=labels, average="macro"
            )

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.lr,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,  # Fix 5: Fine-grained tracking to observe early divergence
            fp16=False,  # Safely disabled to prevent unscaling defects
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            report_to="none",
        )

        trainer = ClickbaitWeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        print(f"\n=== LAUNCHING TRANSFORMER TRAIN SEQUENCE: {self.model_name} ===")
        trainer.train()

        print("\n=== RUNNING CONVERGENCE EVALUATION ===")
        eval_metrics = trainer.evaluate()
        print(eval_metrics)
        return eval_metrics
