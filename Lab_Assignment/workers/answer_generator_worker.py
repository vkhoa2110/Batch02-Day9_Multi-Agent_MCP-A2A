"""Extractive answer worker with citations."""

from __future__ import annotations

import time

from Lab_Assignment.models import SearchHit
from Lab_Assignment.utils import source_label, snippet
from Lab_Assignment.workers.base import BaseWorker


def reorder_for_llm(hits: list[SearchHit]) -> list[SearchHit]:
    """Place strong evidence at the beginning and near the end."""
    if len(hits) <= 2:
        return list(hits)
    front = list(hits[0::2])
    back = list(reversed(hits[1::2]))
    return front + back


class AnswerGeneratorWorker(BaseWorker):
    """Generate a citation-heavy extractive answer from retrieved chunks."""

    name = "answer_generator_worker"

    def run(self, query: str, hits: list[SearchHit], top_k: int = 5):
        start = time.perf_counter()
        if not hits:
            answer = (
                "Không đủ bằng chứng trong corpus Day08 hiện tại để trả lời câu hỏi. "
                "Supervisor không tự suy đoán ngoài nguồn đã truy xuất."
            )
            return self.result(start, answer=answer, notes="No evidence")

        selected = reorder_for_llm(hits[:top_k])
        evidence_lines = []
        for index, hit in enumerate(selected[:3], 1):
            label = source_label(hit.to_dict(), f"source-{index}")
            evidence_lines.append(f"{index}. {snippet(hit.content)} [{label}]")

        answer = "\n".join(
            [
                f"Câu hỏi: {query}",
                "",
                "Trả lời dựa trên các nguồn Day08 đã truy xuất:",
                *evidence_lines,
                "",
                "Kết luận: các đoạn trên là bằng chứng liên quan nhất do Supervisor tổng hợp từ các worker.",
                "Nếu cần kết luận pháp lý chính xác hơn, cần kiểm tra trực tiếp văn bản gốc được trích dẫn.",
            ]
        )
        return self.result(start, hits=selected[:top_k], answer=answer, notes="Extractive answer with citations")

