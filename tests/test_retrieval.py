from types import SimpleNamespace

from app.retrieval import reciprocal_rank_fusion, tokenize


def test_tokenize_keeps_technical_identifiers():
    assert "c++" in tokenize("C++ and HTTP/2")
    assert "http/2" in tokenize("C++ and HTTP/2")


def test_reciprocal_rank_fusion_rewards_documents_found_by_both_retrievers():
    both = SimpleNamespace(id=1, content="same in both")
    semantic_only = SimpleNamespace(id=2, content="semantic only")
    keyword_only = SimpleNamespace(id=3, content="keyword only")

    results = reciprocal_rank_fusion(
        semantic_results=[(both, 0.1), (semantic_only, 0.2)],
        keyword_results=[both, keyword_only],
    )

    assert results[0][0].id == 1
    assert len(results) == 3
    assert results[0][1] > results[1][1]
