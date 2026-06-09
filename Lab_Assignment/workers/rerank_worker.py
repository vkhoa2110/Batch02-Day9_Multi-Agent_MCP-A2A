"""Rerank worker for merging worker outputs."""

from __future__ import annotations

import time
from collections import defaultdict

from Lab_Assignment.models import SearchHit
from Lab_Assignment.utils import bounded_score, lexical_overlap_score, normalize_text
from Lab_Assignment.workers.base import BaseWorker


class RerankWorker(BaseWorker):
    """Merge duplicate hits and rerank them with overlap plus worker agreement."""

    name = "rerank_worker"

    def _domain_hint(self, query: str, content: str) -> float:
        """Small Day08 drug-law hint to separate similar criminal articles."""
        query_norm = normalize_text(query)
        content_norm = normalize_text(content)
        hints = [
            ("tang tru", "dieu 249"),
            ("mua ban", "dieu 251"),
            ("van chuyen", "dieu 250"),
            ("to chuc su dung", "dieu 255"),
        ]
        score = 0.0
        for phrase, article in hints:
            if phrase in query_norm and article in content_norm:
                score += 0.14
            elif phrase in query_norm and phrase not in content_norm:
                score -= 0.04
            elif phrase not in query_norm and article in content_norm:
                score -= 0.04
        return score

    def run(self, query: str, hits: list[SearchHit], top_k: int = 5):
        start = time.perf_counter()
        if not hits:
            return self.result(start, notes="No candidate hits to rerank")

        buckets: dict[str, SearchHit] = {}
        worker_votes: dict[str, set[str]] = defaultdict(set)
        best_original: dict[str, float] = defaultdict(float)

        for hit in hits:
            key = hit.key()
            worker_votes[key].add(hit.worker)
            best_original[key] = max(best_original[key], hit.score)
            if key not in buckets or hit.score > buckets[key].score:
                buckets[key] = SearchHit(
                    content=hit.content,
                    score=hit.score,
                    metadata=dict(hit.metadata),
                    worker=self.name,
                    signals=dict(hit.signals),
                )
            else:
                buckets[key].signals.update(hit.signals)

        reranked: list[SearchHit] = []
        for key, hit in buckets.items():
            overlap = lexical_overlap_score(query, hit.content)
            original = bounded_score(best_original[key])
            agreement = min(1.0, len(worker_votes[key]) / 2)
            domain_hint = self._domain_hint(query, hit.content)
            final_score = min(1.0, max(0.0, 0.50 * overlap + 0.30 * original + 0.20 * agreement + domain_hint))
            metadata = dict(hit.metadata)
            metadata["workers"] = sorted(worker_votes[key])
            metadata["original_score"] = round(best_original[key], 4)
            reranked.append(
                SearchHit(
                    content=hit.content,
                    score=float(final_score),
                    metadata=metadata,
                    worker=self.name,
                    signals={**hit.signals, "overlap": overlap, "agreement": agreement, "domain_hint": domain_hint},
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return self.result(start, reranked[:top_k], notes="Reranked by overlap, original score, and worker agreement")
