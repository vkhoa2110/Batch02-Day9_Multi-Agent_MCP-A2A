"""Local RAG helpers adapted from the Day08 pipeline.

The assignment intentionally avoids external API keys. It loads the copied
Day08 markdown/vector data from Lab_Assignment/data first and can fall back to
the sibling Day08 repo when this folder is used on the same machine.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = ROOT_DIR / "data" / "standardized"
LOCAL_VECTOR_INDEX = ROOT_DIR / "data" / "vectorstore" / "local_index.json"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "with",
    "la", "va", "ve", "cua", "cho", "cac", "co", "duoc", "trong",
    "theo", "mot", "nhung", "nay", "do", "khi", "tu", "tai", "den",
    "voi", "nguon", "noi", "dung",
}


def discover_day08_root() -> Path | None:
    env_root = os.getenv("DAY08_RAG_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            ROOT_DIR.parent.parent / "Day08_RAG_pipeline_cohort2",
            Path("D:/DSA/Day08_RAG_pipeline_cohort2"),
        ]
    )
    for candidate in candidates:
        if (candidate / "data" / "standardized").exists():
            return candidate
    return None


def standardized_dir() -> Path:
    if LOCAL_DATA_DIR.exists() and list(LOCAL_DATA_DIR.rglob("*.md")):
        return LOCAL_DATA_DIR
    day08_root = discover_day08_root()
    if day08_root:
        return day08_root / "data" / "standardized"
    return LOCAL_DATA_DIR


def vector_index_path() -> Path:
    if LOCAL_VECTOR_INDEX.exists():
        return LOCAL_VECTOR_INDEX
    day08_root = discover_day08_root()
    if day08_root:
        candidate = day08_root / "data" / "vectorstore" / "local_index.json"
        if candidate.exists():
            return candidate
    return LOCAL_VECTOR_INDEX


def normalize_text(text: str) -> str:
    text = text.lower().replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(normalize_text(text))
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _extract_metadata_line(content: str, prefix: str) -> str:
    for line in content.splitlines()[:12]:
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def load_markdown_documents(base_dir: Path | None = None) -> list[dict[str, Any]]:
    base_dir = base_dir or standardized_dir()
    documents: list[dict[str, Any]] = []
    if not base_dir.exists():
        return documents

    for md_file in sorted(base_dir.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(base_dir)
        doc_type = "legal" if "legal" in rel_path.parts else "news"
        metadata: dict[str, Any] = {
            "source": md_file.name,
            "path": str(rel_path).replace("\\", "/"),
            "type": doc_type,
        }
        url = _extract_metadata_line(content, "**URL:**")
        if url:
            metadata["url"] = url
        documents.append({"content": content, "metadata": metadata})
    return documents


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue
        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(paragraph), step):
            chunk = paragraph[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        current = ""
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for index, content in enumerate(chunk_text(document.get("content", ""))):
            metadata = dict(document.get("metadata", {}))
            metadata["chunk_index"] = index
            chunks.append({"content": content, "metadata": metadata})
    return chunks


def load_vector_index(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or vector_index_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus() -> list[dict[str, Any]]:
    vector_chunks = load_vector_index()
    if vector_chunks:
        return vector_chunks
    return chunk_documents(load_markdown_documents())


def hashing_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    counts = Counter(tokenize(text))
    vector = [0.0] * dim
    for token, count in counts.items():
        vector[hash(token) % dim] += float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[i] * right[i] for i in range(size))


def lexical_overlap_score(query: str, content: str) -> float:
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    c_counts = Counter(tokenize(content))
    unique = set(q_tokens)
    hits = sum(1 for token in unique if c_counts.get(token, 0) > 0)
    density = sum(min(c_counts.get(token, 0), 3) for token in unique)
    return min(1.0, (hits / len(unique)) * 0.75 + (density / (len(q_tokens) * 3)) * 0.25)


class SimpleBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.avgdl = sum(len(doc) for doc in tokenized_corpus) / self.doc_count if self.doc_count else 0.0
        self.doc_freq: Counter[str] = Counter()
        for document in tokenized_corpus:
            self.doc_freq.update(set(document))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for document in self.tokenized_corpus:
            term_freq = Counter(document)
            doc_len = len(document) or 1
            score = 0.0
            for token in query_tokens:
                freq = term_freq.get(token, 0)
                if not freq:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def bounded_score(value: float) -> float:
    return value / (abs(value) + 1.0) if value else 0.0


def source_label(item: dict[str, Any], fallback: str = "local corpus") -> str:
    metadata = item.get("metadata", {})
    return str(metadata.get("source") or metadata.get("path") or fallback)


def snippet(text: str, max_len: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rsplit(" ", 1)[0] + "..."

