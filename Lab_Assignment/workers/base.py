"""Base worker class."""

from __future__ import annotations

import time

from Lab_Assignment.models import SearchHit, WorkerResult


class BaseWorker:
    name = "base_worker"

    def result(
        self,
        start_time: float,
        hits: list[SearchHit] | None = None,
        answer: str = "",
        notes: str = "",
        error: str = "",
    ) -> WorkerResult:
        return WorkerResult(
            worker_name=self.name,
            hits=hits or [],
            answer=answer,
            elapsed_ms=(time.perf_counter() - start_time) * 1000,
            notes=notes,
            error=error,
        )

