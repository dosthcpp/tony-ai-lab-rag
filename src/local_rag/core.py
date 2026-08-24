from __future__ import annotations

import json
import hashlib
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - reported clearly when a PDF is requested
    PdfReader = None


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


@dataclass(frozen=True)
class Chunk:
    source: str
    ordinal: int
    text: str
    embedding: list[float]
    document_id: str = ""
    title: str = ""
    page: int | None = None
    section: str | None = None
    chunk_id: str = ""


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


def _document_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]


def _chunk_id(document_id: str, page: int | None, section: str | None, ordinal: int, text: str) -> str:
    identity = f"{document_id}\0{page}\0{section}\0{ordinal}\0{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if "\n".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading, body = match.group(1), []
        else:
            body.append(line)
    if "\n".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections


def _read_document(path: Path) -> tuple[str, list[tuple[int | None, str | None, str]]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PDF ingestion requires the 'pypdf' package")
        reader = PdfReader(str(path))
        metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
        title = metadata_title.strip() if isinstance(metadata_title, str) and metadata_title.strip() else path.stem
        pages = [(number, None, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
        return title, pages
    if suffix not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"Unsupported document type: {path.suffix or '<none>'}")
    text = path.read_text(encoding="utf-8")
    if suffix in {".md", ".markdown"}:
        sections = [(None, heading, body) for heading, body in _markdown_sections(text)]
        title = next((heading for _, heading, _ in sections if heading), path.stem)
        return title, sections
    return path.stem, [(None, None, text)]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must be non-empty and have equal dimensions")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    if denominator == 0:
        return 0.0
    return sum(x * y for x, y in zip(left, right)) / denominator


def ingest_files(
    paths: Iterable[Path],
    client: OllamaClient,
    embedding_model: str,
    size: int,
    overlap: int,
    approved_roots: Iterable[Path] | None = None,
) -> list[Chunk]:
    roots = [root.expanduser().resolve(strict=True) for root in (approved_roots or [Path.cwd()])]
    pending: list[tuple[str, int, str, str, str, int | None, str | None, str]] = []
    for supplied_path in paths:
        path = supplied_path.expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Document is not a regular file: {supplied_path}")
        root = next((candidate for candidate in roots if path.is_relative_to(candidate)), None)
        if root is None:
            raise ValueError(f"Document is outside approved roots: {supplied_path}")
        source = path.relative_to(root).as_posix()
        document_id = _document_id(path, root)
        title, units = _read_document(path)
        ordinal = 0
        for page, section, unit_text in units:
            for text in chunk_text(unit_text, size, overlap):
                pending.append(
                    (source, ordinal, text, document_id, title, page, section, _chunk_id(document_id, page, section, ordinal, text))
                )
                ordinal += 1
    embeddings = client.embed([item[2] for item in pending], embedding_model)
    return [Chunk(source, ordinal, text, embedding, document_id, title, page, section, chunk_id)
            for (source, ordinal, text, document_id, title, page, section, chunk_id), embedding in zip(pending, embeddings)]


def save_index(path: Path, chunks: Sequence[Chunk], embedding_model: str) -> None:
    payload = {"version": 2, "embedding_model": embedding_model, "chunks": [asdict(chunk) for chunk in chunks]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(path: Path) -> tuple[str, list[Chunk]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in {1, 2} or not isinstance(payload.get("embedding_model"), str):
        raise ValueError("Unsupported or invalid index")
    return payload["embedding_model"], [Chunk(**item) for item in payload.get("chunks", [])]


def retrieve(chunks: Sequence[Chunk], query_embedding: Sequence[float], top_k: int) -> list[tuple[float, Chunk]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scored = [(cosine_similarity(query_embedding, chunk.embedding), chunk) for chunk in chunks]
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]


def answer_prompt(question: str, matches: Sequence[tuple[float, Chunk]]) -> str:
    context = "\n\n".join(
        f"[{number}] title={chunk.title or chunk.source} source={chunk.source} "
        f"page={chunk.page or 'n/a'} section={chunk.section or 'n/a'} chunk_id={chunk.chunk_id or chunk.ordinal}\n{chunk.text}"
        for number, (_, chunk) in enumerate(matches, start=1)
    )
    return (
        "Answer the question using only the supplied context. "
        "If the context is insufficient, respond exactly: "
        "\"The provided study materials do not directly support this.\" "
        "Cite claims supported by the context with [1], [2], etc.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
