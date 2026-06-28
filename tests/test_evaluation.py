from src.evaluation import AdvancedEvaluationSuite


def test_evaluate_task1(capsys):
    """Verifies classification reporting prints outputs correctly."""
    y_true = ["phrase", "passage", "multi"]
    y_pred = ["phrase", "passage", "phrase"]
    AdvancedEvaluationSuite.evaluate_task1(y_true, y_pred)
    captured = capsys.readouterr()
    assert "CLASSIFICATION PERFORMANCE" in captured.out


def test_evaluate_task2(capsys):
    """Verifies BLEU/ROUGE metrics output formatting handles text inputs correctly."""
    ground_truths = [["the movie was great"], ["sunny day"]]
    predictions = [["the movie was great"], ["cloudy day"]]
    AdvancedEvaluationSuite.evaluate_task2(ground_truths, predictions)
    captured = capsys.readouterr()
    assert "TEXT GENERATION PERFORMANCE" in captured.out
    assert "Mean BLEU Score" in captured.out
