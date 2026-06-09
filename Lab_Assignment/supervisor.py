"""Supervisor agent that coordinates Day08 RAG workers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from Lab_Assignment.models import SearchHit, WorkerResult
from Lab_Assignment.utils import source_label
from Lab_Assignment.workers.answer_generator_worker import AnswerGeneratorWorker
from Lab_Assignment.workers.bm25_search_worker import BM25SearchWorker
from Lab_Assignment.workers.rerank_worker import RerankWorker
from Lab_Assignment.workers.semantic_search_worker import SemanticSearchWorker


class SupervisorAgent:
    """Supervisor-Workers implementation for the Day08 RAG pipeline.

    Flow:
    1. Supervisor receives the question.
    2. SemanticSearchWorker and BM25SearchWorker run in parallel.
    3. RerankWorker merges and reranks retrieved chunks.
    4. AnswerGeneratorWorker creates the final cited answer.
    """

    def __init__(self) -> None:
        self.semantic_worker = SemanticSearchWorker()
        self.bm25_worker = BM25SearchWorker()
        self.rerank_worker = RerankWorker()
        self.answer_worker = AnswerGeneratorWorker()

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        retrieval_top_k = max(top_k * 2, 6)
        worker_results: list[WorkerResult] = []
        candidate_hits: list[SearchHit] = []

        retrieval_workers = [self.semantic_worker, self.bm25_worker]
        with ThreadPoolExecutor(max_workers=len(retrieval_workers)) as executor:
            future_map = {
                executor.submit(worker.run, query, retrieval_top_k): worker.name
                for worker in retrieval_workers
            }
            for future in as_completed(future_map):
                worker_name = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = WorkerResult(worker_name=worker_name, error=str(exc))
                worker_results.append(result)
                candidate_hits.extend(result.hits)

        rerank_result = self.rerank_worker.run(query, candidate_hits, top_k=top_k)
        worker_results.append(rerank_result)

        answer_result = self.answer_worker.run(query, rerank_result.hits, top_k=top_k)
        worker_results.append(answer_result)

        total_ms = (time.perf_counter() - started) * 1000
        return {
            "question": query,
            "answer": answer_result.answer,
            "sources": [hit.to_dict() for hit in rerank_result.hits],
            "latency_ms": round(total_ms, 2),
            "worker_trace": [result.to_dict() for result in worker_results],
            "architecture": "Supervisor -> parallel Semantic/BM25 Workers -> Rerank Worker -> Answer Worker",
        }


def format_report(result: dict[str, Any]) -> str:
    lines = [
        "# Supervisor-Workers Day08 RAG Result",
        "",
        f"Architecture: {result['architecture']}",
        f"Latency: {result['latency_ms']:.2f} ms",
        "",
        "## Answer",
        "",
        result["answer"],
        "",
        "## Worker Trace",
        "",
    ]

    for worker in result["worker_trace"]:
        status = "error" if worker.get("error") else "ok"
        lines.append(
            f"- {worker['worker_name']}: {status}, "
            f"{worker['hit_count']} hits, {worker['elapsed_ms']:.2f} ms"
        )
        if worker.get("notes"):
            lines.append(f"  Notes: {worker['notes']}")
        if worker.get("error"):
            lines.append(f"  Error: {worker['error']}")

    lines.extend(["", "## Top Sources", ""])
    for index, source in enumerate(result.get("sources", [])[:5], 1):
        label = source_label(source, f"source-{index}")
        metadata = source.get("metadata", {})
        workers = ", ".join(metadata.get("workers", [])) or source.get("worker", "")
        lines.append(
            f"{index}. {label} | score={source.get('score', 0):.3f} | "
            f"type={metadata.get('type', 'unknown')} | workers={workers}"
        )
    return "\n".join(lines)

