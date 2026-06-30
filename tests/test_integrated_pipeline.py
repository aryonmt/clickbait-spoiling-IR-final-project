from unittest.mock import MagicMock, patch

from main_integrated_pipeline import ClickbaitIntegratedPipeline
from src.config import PipelineConfig


@patch("main_integrated_pipeline.AutoTokenizer.from_pretrained")
@patch("main_integrated_pipeline.AutoModelForSeq2SeqLM.from_pretrained")
@patch("main_integrated_pipeline.AutoModelForQuestionAnswering.from_pretrained")
@patch("main_integrated_pipeline.AutoModelForSequenceClassification.from_pretrained")
@patch("os.path.exists")
def test_integrated_pipeline_seq2seq_wiring(
    mock_exists,
    mock_t1_from_pretrained,
    mock_t2_from_pretrained,
    mock_seq2seq_from_pretrained,
    mock_tokenizer_from_pretrained,
):
    """Verifies that the integrated pipeline correctly wires and instantiates

    the Seq2Seq model when task2_multi_strategy is set to 'seq2seq'.
    """
    # Configure path.exists to return True for results directories
    mock_exists.side_effect = lambda path: True if "results_multi" in path else False

    # Configure mock models
    mock_seq2seq_model = MagicMock()
    mock_seq2seq_from_pretrained.return_value = mock_seq2seq_model

    config = PipelineConfig(task2_multi_strategy="seq2seq")

    pipeline = ClickbaitIntegratedPipeline(
        task1_dir="mock_t1",
        phrase_dir="mock_phrase",
        passage_dir="mock_passage",
        config=config,
    )

    assert pipeline.multi_strategy == "seq2seq"
    assert pipeline.multi_model is mock_seq2seq_model
