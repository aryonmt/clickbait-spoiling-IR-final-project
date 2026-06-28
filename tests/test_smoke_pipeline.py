import os

from src.data_loader import JSONLLoader
from src.transformer_classifier import TransformerSpoilerClassifier


def test_smoke_task1_training(tmpdir):
    """Executes end-to-end classification training on tiny dataset to verify stability."""
    train_path = os.path.join("tests", "fixtures", "tiny_train.jsonl")
    val_path = os.path.join("tests", "fixtures", "tiny_val.jsonl")

    train_df = JSONLLoader(train_path).load_data()
    val_df = JSONLLoader(val_path).load_data()

    # Use lightweight model to prevent actual training workloads
    classifier = TransformerSpoilerClassifier(
        model_name="hf-internal-testing/tiny-random-RoBERTa", max_length=64, lr=1e-5
    )

    output_dir = str(tmpdir.mkdir("smoke_output"))
    classifier.train_pipeline(train_df=train_df, val_df=val_df, output_dir=output_dir)

    assert os.path.exists(output_dir)
