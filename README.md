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
local-rag ingest notes.md handbook.txt
local-rag ask "What does the handbook say about backups?"
```

The index defaults to `.rag-index.json`. Input is limited to UTF-8 text and Markdown in this baseline. Answers are instructed to use retrieved context only and emit numbered citations; the CLI prints the corresponding source paths and chunk scores.

## Validate

```bash
.venv/bin/pytest
```

## Known baseline limits

- JSON index and exact cosine scan are intended for small local corpora.
- There is no PDF parser, OCR, reranker, access-control layer, or HTTP API yet.
- Source files are read at ingestion time; keep the generated index private if documents are sensitive.
