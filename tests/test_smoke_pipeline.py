import os

from src.config import PipelineConfig
from src.data_loader import JSONLLoader
from src.transformer_classifier import TransformerSpoilerClassifier


def test_smoke_task1_training(tmpdir):
    """Executes end-to-end classification training on tiny dataset to verify stability."""
    train_path = os.path.join("tests", "fixtures", "tiny_train.jsonl")
    val_path = os.path.join("tests", "fixtures", "tiny_val.jsonl")

    train_df = JSONLLoader(train_path).load_data()
    val_df = JSONLLoader(val_path).load_data()

    # Configure custom pipeline settings for fast isolated tests
    config = PipelineConfig(
        task1_model_name="hf-internal-testing/tiny-random-RoBERTa",
        task1_max_length=64,
        task1_lr=1e-5,
        output_dir=str(tmpdir.mkdir("smoke_output")),
        task1_ignore_mismatched_sizes=True,  # Allowed strictly under mock test structures
    )

    classifier = TransformerSpoilerClassifier(config=config)
    classifier.train_pipeline(train_df=train_df, val_df=val_df)


def test_regression_baseline_snapshot():
    """Regression snapshot test to ensure baseline validation files run safely."""
    val_path = os.path.join("tests", "fixtures", "tiny_val.jsonl")
    loader = JSONLLoader(val_path)
    df = loader.load_data()

    assert len(df) > 0
    assert "uuid" in df.columns
    assert "targetParagraphs" in df.columns
