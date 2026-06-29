from src.retrieval_generator import RetrievalSpoilerGenerator


def test_generate_multi_spoiler():
    """Verifies that RetrievalSpoilerGenerator extracts logically relevant sentences."""
    post_text = "What is the capital of France?"
    paragraphs = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy.",
    ]
    # Test Strategy 1: Jaccard
    spoiler_j = RetrievalSpoilerGenerator.generate_multi_spoiler(
        post_text, paragraphs, top_k=1, method="jaccard"
    )
    assert len(spoiler_j) == 1
    assert "Paris" in spoiler_j[0]

    # Test Strategy 2: TF-IDF Cosine Similarity
    spoiler_t = RetrievalSpoilerGenerator.generate_multi_spoiler(
        post_text, paragraphs, top_k=1, method="tfidf"
    )
    assert len(spoiler_t) == 1
    assert "Paris" in spoiler_t[0]
