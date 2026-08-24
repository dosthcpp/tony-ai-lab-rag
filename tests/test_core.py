import json

import pytest

from local_rag.core import Chunk, answer_prompt, chunk_text, cosine_similarity, load_index, retrieve, save_index


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
    assert json.loads(path.read_text())["version"] == 1


def test_prompt_requires_grounding_and_citations():
    prompt = answer_prompt("What?", [(0.9, Chunk("doc.md", 0, "Fact", [1.0]))])
    assert "only the supplied context" in prompt
    assert "source=doc.md" in prompt
    assert "Cite claims" in prompt
