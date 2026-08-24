# Tony AI Lab Local RAG Baseline

A dependency-light, local-only retrieval augmented generation baseline. Document text, embeddings, retrieval, and answer generation remain on the machine through Ollama.

## Runtime

- Python 3.11+
- Ollama at `http://127.0.0.1:11434`
- `nomic-embed-text` for embeddings
- `llama3.2` for grounded answer generation

No cloud API or API key is used.

## Setup and use

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
python3 -m venv .venv
.venv/bin/pip install -e . pytest
local-rag ingest --root ./course-materials course-materials/notes.md course-materials/handbook.pdf
local-rag ask "What does the handbook say about backups?"
```

The index defaults to `.rag-index.json`. Ingestion supports UTF-8 text, heading-aware Markdown, and text-based PDF files. Every chunk stores a deterministic document/chunk ID, title, page (PDF), section (Markdown), source-relative path, text, and embedding. Answers use retrieved context only and emit numbered citations with this metadata.

Every input must resolve inside an explicitly approved `--root` (repeat it for multiple roots); the current directory is the safe default. Resolved-path checks reject traversal and symlink escapes. Unsupported file types are rejected. Keep approved roots narrow and do not ingest private course material into any external service.

## Validate

```bash
.venv/bin/pytest
```

## Known baseline limits

- JSON index and exact cosine scan are intended for small local corpora.
- Scanned/image-only PDFs require a separately reviewed local OCR step; this baseline does not perform OCR.
- There is no reranker, multi-user authorization layer, or HTTP API yet.
- Source files are read at ingestion time; keep the generated index private if documents are sensitive.
