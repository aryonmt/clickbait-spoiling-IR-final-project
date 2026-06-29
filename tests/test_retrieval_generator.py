from src.retrieval_generator import RetrievalSpoilerGenerator


def test_generate_multi_spoiler():
    """Verifies that RetrievalSpoilerGenerator extracts logically relevant sentences."""
    post_text = "What is the capital of France?"
    paragraphs = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy.",
    ]
    spoiler = RetrievalSpoilerGenerator.generate_multi_spoiler(
        post_text, paragraphs, top_k=1
    )
    assert len(spoiler) == 1
    assert "Paris" in spoiler[0]
