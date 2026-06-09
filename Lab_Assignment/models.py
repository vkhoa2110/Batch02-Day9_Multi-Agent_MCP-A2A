"""Shared data models for the Supervisor-Workers RAG assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """One retrieved chunk produced by a worker."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    worker: str = ""
    signals: dict[str, float] = field(default_factory=dict)

    def key(self) -> str:
        source = self.metadata.get("source") or self.metadata.get("path")
        chunk_index = self.metadata.get("chunk_index")
        if source is not None and chunk_index is not None:
            return f"{source}:{chunk_index}"
        return " ".join(self.content.split()).lower()[:220]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": dict(self.metadata),
            "worker": self.worker,
            "signals": dict(self.signals),
        }

    @classmethod
    def from_dict(cls, item: dict[str, Any], worker: str) -> "SearchHit":
        return cls(
            content=str(item.get("content", "")),
            score=float(item.get("score", 0.0)),
            metadata=dict(item.get("metadata", {})),
            worker=worker,
            signals={worker: float(item.get("score", 0.0))},
        )


@dataclass
class WorkerResult:
    """Traceable result returned by one worker."""

    worker_name: str
    hits: list[SearchHit] = field(default_factory=list)
    answer: str = ""
    elapsed_ms: float = 0.0
    notes: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_name": self.worker_name,
            "hit_count": len(self.hits),
            "hits": [hit.to_dict() for hit in self.hits],
            "answer": self.answer,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "notes": self.notes,
            "error": self.error,
        }

