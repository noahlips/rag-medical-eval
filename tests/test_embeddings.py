"""EmbeddingIndex tests.

A tiny fake model stands in for sentence-transformers, so these tests check the
indexing and ranking logic without downloading a model or installing torch.
"""

import numpy as np
import pytest

import src.embeddings as embeddings_module
from src.chunking import Chunk
from src.embeddings import EmbeddingIndex
from src.evaluate import evaluate_retrieval

CHUNKS = [
    Chunk("diabete", 0, "Le diabète de type 2 se caractérise par une glycémie élevée."),
    Chunk("asthme", 0, "L'asthme provoque une inflammation des bronches."),
    Chunk("grippe", 0, "La grippe est une infection virale avec fièvre."),
]


class FakeModel:
    """Encodes a text as the presence of a few keywords, then L2-normalises."""

    max_seq_length = 128
    vocabulary = ("diabète", "asthme", "grippe")

    def encode(self, texts, **kwargs):
        rows = [[float(word in text.lower()) for word in self.vocabulary] for text in texts]
        vectors = np.array(rows, dtype=np.float32) + 1e-6
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(embeddings_module, "_load_model", lambda name: FakeModel())


def test_search_ranks_semantically_closest_chunk_first():
    index = EmbeddingIndex(CHUNKS)
    results = index.search("quels sont les signes du diabète", k=3)
    assert results[0][0].doc_id == "diabete"
    assert results[0][1] > results[1][1]


def test_embeddings_are_normalised():
    index = EmbeddingIndex(CHUNKS)
    norms = np.linalg.norm(index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    EmbeddingIndex(CHUNKS).save(tmp_path)
    loaded = EmbeddingIndex.load(tmp_path)
    assert loaded.search("crise d'asthme", k=1)[0][0].doc_id == "asthme"


def test_evaluate_retrieval_accepts_embedding_index():
    index = EmbeddingIndex(CHUNKS)
    eval_set = [
        {"question": "le diabète", "expected_doc": "diabete", "type": "definition"},
        {"question": "la grippe", "expected_doc": "grippe", "type": "definition"},
    ]
    report = evaluate_retrieval(index, eval_set, k=2)
    assert report["hit@2"] == 1.0
    assert report["mrr"] == 1.0
