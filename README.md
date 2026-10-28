# Tony AI Lab Local RAG Baseline

A dependency-light, local-only retrieval augmented generation baseline. Document text, embeddings, retrieval, and answer generation remain on the machine through Ollama.

## Requirements

- Python 3.11 or newer, available as `python3` (check with `python3 --version`). If your system Python is older, use a newer interpreter when creating the virtual environment.
- Git and [Ollama](https://ollama.com/download).
- A running local Ollama service at `http://127.0.0.1:11434`.
- `nomic-embed-text` for embeddings and `llama3.2` for answer generation. The commands below download these models; allow enough local disk space and memory to run them.

Installation and model downloads need network access. Document processing uses the local Ollama service; no cloud API or API key is used.

## Install

```bash
git clone https://github.com/dosthcpp/tony-ai-lab-rag.git
cd tony-ai-lab-rag
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

The commands below use `.venv/bin/` explicitly, so virtual environment activation is not required. These examples use a macOS/Linux shell.

If Ollama is not already running (for example, through its desktop app), start it in a **separate terminal** and leave it running:

```bash
ollama serve
```

In the repository terminal, download the two default models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Run the included example

From the repository directory, index the text file included in this repository and ask a question it can answer:

```bash
.venv/bin/local-rag ingest --root ./examples examples/baseline.txt
.venv/bin/local-rag ask "Which models does this baseline use?"
```

Ingestion prints `Indexed 1 chunks into .rag-index.json`. The answer should identify `nomic-embed-text` and `llama3.2`, followed by a `Sources:` section with `[1]` referencing `baseline.txt`. Answer wording can vary between runs.

The index defaults to `.rag-index.json` in the current directory. Each ingest command replaces that index, so pass all intended input files together. To use a different index, pass `--index PATH` to both `ingest` and `ask`. To use a different local Ollama address, put `--ollama-url URL` before the subcommand.

## Use your own documents

Ingestion supports UTF-8 text, heading-aware Markdown, and text-based PDF files. Supply your own existing files as positional arguments to `ingest`, and set `--root` to their containing directory. Course notes and PDFs are not included in this repository; `examples/baseline.txt` is sufficient for the first run above.

Every chunk stores a deterministic document/chunk ID, title, page (PDF), section (Markdown), source-relative path, text, and embedding. Answers are prompted to use retrieved context only; the CLI prints numbered source references with this metadata. Model output can still be incorrect, so check it against the sources.

Every input must resolve inside an explicitly approved `--root` (repeat it for multiple roots); the current directory is the safe default. Resolved-path checks reject traversal and symlink escapes. Unsupported file types are rejected. Keep approved roots narrow and do not ingest private course material into any external service.

## Validate

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest
```

The unit tests use synthetic inputs and a fake embedding client; they do not need a running Ollama service. The included example above exercises the real local service.

## Known baseline limits

- JSON index and exact cosine scan are intended for small local corpora.
- Scanned/image-only PDFs require a separately reviewed local OCR step; this baseline does not perform OCR.
- There is no reranker, multi-user authorization layer, or HTTP API yet.
- Source files are read at ingestion time; keep the generated index private if documents are sensitive.

## License

MIT; see [LICENSE](LICENSE). This license covers this repository's code and included example, not documents or models you obtain separately.
