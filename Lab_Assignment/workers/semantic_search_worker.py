"""Semantic-style worker over the Day08 corpus."""

from __future__ import annotations

import time

from Lab_Assignment.models import SearchHit
from Lab_Assignment.utils import cosine_similarity, hashing_embedding, lexical_overlap_score, load_corpus
from Lab_Assignment.workers.base import BaseWorker


class SemanticSearchWorker(BaseWorker):
    """Dense retrieval worker using lightweight local embeddings."""

    name = "semantic_search_worker"

    def run(self, query: str, top_k: int = 8):
        start = time.perf_counter()
        if top_k <= 0:
            return self.result(start, notes="top_k <= 0")

        query_embedding = hashing_embedding(query)
        hits: list[SearchHit] = []
        for chunk in load_corpus():
            content = chunk.get("content", "")
            dense_score = cosine_similarity(query_embedding, hashing_embedding(content))
            overlap_score = lexical_overlap_score(query, content)
            score = max(0.0, 0.72 * dense_score + 0.28 * overlap_score)
            if score <= 0:
                continue
            metadata = dict(chunk.get("metadata", {}))
            metadata["retrieval_mode"] = "semantic_hashing"
            hits.append(
                SearchHit(
                    content=content,
                    score=float(score),
                    metadata=metadata,
                    worker=self.name,
                    signals={"dense": float(dense_score), "overlap": float(overlap_score)},
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return self.result(start, hits[:top_k], notes="Local dense-style retrieval over Day08 chunks")

