import json

import pytest

import local_rag.core as core
from local_rag.core import Chunk, answer_prompt, chunk_text, cosine_similarity, ingest_files, load_index, retrieve, save_index


class FakeClient:
    def embed(self, texts, model):
        return [[float(len(text)), 1.0] for text in texts]


def test_chunk_text_preserves_content_with_overlap():
    chunks = chunk_text("alpha beta gamma delta epsilon", size=16, overlap=5)
    assert chunks == ["alpha beta", "beta gamma", "gamma delta", "delta epsilon"]


def test_chunk_text_validates_configuration():
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)


def test_cosine_and_retrieval_order():
    chunks = [Chunk("a", 0, "A", [1.0, 0.0]), Chunk("b", 0, "B", [0.0, 1.0])]
    matches = retrieve(chunks, [0.9, 0.1], top_k=1)
    assert matches[0][1].source == "a"
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_index_round_trip(tmp_path):
    path = tmp_path / "index.json"
    chunks = [Chunk("notes.md", 2, "evidence", [0.1, 0.2])]
    save_index(path, chunks, "embed-model")
    model, loaded = load_index(path)
    assert model == "embed-model"
    assert loaded == chunks
    assert json.loads(path.read_text())["version"] == 2


def test_prompt_requires_grounding_and_citations():
    prompt = answer_prompt("What?", [(0.9, Chunk("doc.md", 0, "Fact", [1.0]))])
    assert "only the supplied context" in prompt
    assert "source=doc.md" in prompt
    assert "Cite claims" in prompt
    assert "The provided study materials do not directly support this." in prompt


def test_markdown_ingestion_preserves_metadata_and_headings(tmp_path):
    document = tmp_path / "lecture.md"
    document.write_text("# Lecture One\nIntro facts.\n\n## Equations\nx = y + z\n", encoding="utf-8")
    chunks = ingest_files([document], FakeClient(), "fake", 100, 10, [tmp_path])
    assert [chunk.section for chunk in chunks] == ["Lecture One", "Equations"]
    assert all(chunk.document_id and chunk.chunk_id for chunk in chunks)
    assert all(chunk.title == "Lecture One" and chunk.source == "lecture.md" for chunk in chunks)


def test_ingestion_rejects_paths_outside_approved_root(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(ValueError, match="outside approved roots"):
        ingest_files([outside], FakeClient(), "fake", 100, 10, [approved])


def test_pdf_ingestion_preserves_title_and_page(monkeypatch, tmp_path):
    document = tmp_path / "paper.pdf"
    document.write_bytes(b"fake fixture")

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        metadata = type("Metadata", (), {"title": "Course Paper"})()
        pages = [Page("first page evidence"), Page("second page evidence")]

    monkeypatch.setattr(core, "PdfReader", lambda _: Reader())
    chunks = ingest_files([document], FakeClient(), "fake", 100, 10, [tmp_path])
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert all(chunk.title == "Course Paper" for chunk in chunks)


def test_loads_legacy_v1_index(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "version": 1,
        "embedding_model": "legacy",
        "chunks": [{"source": "a.txt", "ordinal": 0, "text": "fact", "embedding": [1.0]}],
    }), encoding="utf-8")
    model, chunks = load_index(path)
    assert model == "legacy"
    assert chunks[0].document_id == ""
