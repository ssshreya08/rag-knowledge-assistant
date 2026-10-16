import re
from collections import defaultdict
from typing import Iterable

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk


def tokenize(text: str) -> list[str]:
    """Basic tokenizer for BM25; keeps identifiers like C++, HTTP/2 and e-mail words."""
    return re.findall(r"[a-zA-Z0-9_+#./-]+", text.lower())


def semantic_search(db: Session, query_embedding: list[float], limit: int) -> list[tuple[Chunk, float]]:
    distance_expr = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    rows = db.execute(
        select(Chunk, distance_expr)
        .order_by(distance_expr.asc())
        .limit(limit)
    ).all()
    return [(row[0], float(row[1])) for row in rows]


def bm25_search(db: Session, query: str, limit: int) -> list[Chunk]:
    """Build a small-corpus BM25 index from stored chunks and retrieve keyword matches.

    This intentionally favors simplicity for a portfolio project. For a very large corpus,
    move lexical retrieval into a dedicated search engine or a PostgreSQL full-text index.
    """
    chunks = db.scalars(select(Chunk)).all()
    if not chunks:
        return []
    tokenized_corpus = [tokenize(chunk.content) for chunk in chunks]
    scores = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    ranked_scores = scores.get_scores(query_tokens)
    ranked_indices = sorted(range(len(chunks)), key=lambda index: float(ranked_scores[index]), reverse=True)
    return [chunks[index] for index in ranked_indices[:limit] if ranked_scores[index] > 0]


def reciprocal_rank_fusion(
    semantic_results: list[tuple[Chunk, float]],
    keyword_results: list[Chunk],
    rrf_constant: int = 60,
) -> list[tuple[Chunk, float, float | None]]:
    """Fuse semantic and BM25 result lists using reciprocal-rank fusion (RRF).

    Returns tuples of (chunk, fused_score, semantic_cosine_distance_if_available).
    RRF scores are ranking signals, not calibrated probabilities.
    """
    fused_scores: dict[int, float] = defaultdict(float)
    chunks_by_id: dict[int, Chunk] = {}
    semantic_distances: dict[int, float] = {}

    for rank, (chunk, distance) in enumerate(semantic_results, start=1):
        fused_scores[chunk.id] += 1.0 / (rrf_constant + rank)
        chunks_by_id[chunk.id] = chunk
        semantic_distances[chunk.id] = distance

    for rank, chunk in enumerate(keyword_results, start=1):
        fused_scores[chunk.id] += 1.0 / (rrf_constant + rank)
        chunks_by_id[chunk.id] = chunk

    ordered_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [
        (chunks_by_id[chunk_id], fused_scores[chunk_id], semantic_distances.get(chunk_id))
        for chunk_id in ordered_ids
    ]
