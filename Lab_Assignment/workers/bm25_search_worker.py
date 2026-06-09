"""BM25 worker over the Day08 corpus."""

from __future__ import annotations

import time

from Lab_Assignment.models import SearchHit
from Lab_Assignment.utils import SimpleBM25, load_corpus, tokenize
from Lab_Assignment.workers.base import BaseWorker


class BM25SearchWorker(BaseWorker):
    """Sparse keyword retrieval worker."""

    name = "bm25_search_worker"

    def run(self, query: str, top_k: int = 8):
        start = time.perf_counter()
        if top_k <= 0:
            return self.result(start, notes="top_k <= 0")

        corpus = load_corpus()
        if not corpus:
            return self.result(start, notes="No Day08 corpus available")

        bm25 = SimpleBM25([tokenize(item.get("content", "")) for item in corpus])
        scores = bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

        hits: list[SearchHit] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = corpus[index]
            metadata = dict(chunk.get("metadata", {}))
            metadata["retrieval_mode"] = "bm25"
            hits.append(
                SearchHit(
                    content=chunk.get("content", ""),
                    score=score,
                    metadata=metadata,
                    worker=self.name,
                    signals={"bm25": score},
                )
            )
            if len(hits) >= top_k:
                break

        return self.result(start, hits, notes="BM25 keyword retrieval over Day08 chunks")

