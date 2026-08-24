from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


@dataclass(frozen=True)
class Chunk:
    source: str
    ordinal: int
    text: str
    embedding: list[float]


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed for {route}: {exc}") from exc

    def embed(self, texts: Sequence[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        result = self._post("/api/embed", {"model": model, "input": list(texts)})
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError("Ollama returned an invalid embeddings response")
        return embeddings

    def generate(self, prompt: str, model: str) -> str:
        result = self._post(
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False},
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise OllamaError("Ollama returned an invalid generation response")
        return response.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must be non-empty and have equal dimensions")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    if denominator == 0:
        return 0.0
    return sum(x * y for x, y in zip(left, right)) / denominator


def ingest_files(
    paths: Iterable[Path], client: OllamaClient, embedding_model: str, size: int, overlap: int
) -> list[Chunk]:
    pending: list[tuple[str, int, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        pending.extend((str(path), ordinal, chunk) for ordinal, chunk in enumerate(chunk_text(text, size, overlap)))
    embeddings = client.embed([item[2] for item in pending], embedding_model)
    return [Chunk(source, ordinal, text, embedding) for (source, ordinal, text), embedding in zip(pending, embeddings)]


def save_index(path: Path, chunks: Sequence[Chunk], embedding_model: str) -> None:
    payload = {"version": 1, "embedding_model": embedding_model, "chunks": [asdict(chunk) for chunk in chunks]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path) -> tuple[str, list[Chunk]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("embedding_model"), str):
        raise ValueError("Unsupported or invalid index")
    return payload["embedding_model"], [Chunk(**item) for item in payload.get("chunks", [])]


def retrieve(chunks: Sequence[Chunk], query_embedding: Sequence[float], top_k: int) -> list[tuple[float, Chunk]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scored = [(cosine_similarity(query_embedding, chunk.embedding), chunk) for chunk in chunks]
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]


def answer_prompt(question: str, matches: Sequence[tuple[float, Chunk]]) -> str:
    context = "\n\n".join(
        f"[{number}] source={chunk.source} chunk={chunk.ordinal}\n{chunk.text}"
        for number, (_, chunk) in enumerate(matches, start=1)
    )
    return (
        "Answer the question using only the supplied context. "
        "If the context is insufficient, say so explicitly. Cite claims with [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
