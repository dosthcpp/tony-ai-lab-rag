from __future__ import annotations

import argparse
from pathlib import Path

from .core import OllamaClient, answer_prompt, ingest_files, load_index, retrieve, save_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-only Ollama RAG baseline")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index UTF-8 text or Markdown files")
    ingest.add_argument("files", nargs="+", type=Path)
    ingest.add_argument("--index", type=Path, default=Path(".rag-index.json"))
    ingest.add_argument("--embedding-model", default="nomic-embed-text")
    ingest.add_argument("--chunk-size", type=int, default=900)
    ingest.add_argument("--overlap", type=int, default=150)

    ask = subparsers.add_parser("ask", help="Retrieve evidence and answer locally")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=Path(".rag-index.json"))
    ask.add_argument("--model", default="llama3.2")
    ask.add_argument("--top-k", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = OllamaClient(args.ollama_url)
    if args.command == "ingest":
        chunks = ingest_files(args.files, client, args.embedding_model, args.chunk_size, args.overlap)
        save_index(args.index, chunks, args.embedding_model)
        print(f"Indexed {len(chunks)} chunks into {args.index}")
        return 0

    embedding_model, chunks = load_index(args.index)
    query_embedding = client.embed([args.question], embedding_model)[0]
    matches = retrieve(chunks, query_embedding, args.top_k)
    print(client.generate(answer_prompt(args.question, matches), args.model))
    print("\nSources:")
    for number, (score, chunk) in enumerate(matches, start=1):
        print(f"[{number}] {chunk.source}#chunk-{chunk.ordinal} score={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
