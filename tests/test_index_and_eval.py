import pytest

from src.chunking import Chunk
from src.evaluate import evaluate_retrieval
from src.index import BM25Index, tokenize
from src.rag import extract_citations, format_sources

CHUNKS = [
    Chunk("diabete", 0, "Le diabète de type 2 se caractérise par une glycémie élevée et une résistance à l'insuline."),
    Chunk("asthme", 0, "L'asthme provoque une inflammation des bronches avec des sifflements respiratoires."),
    Chunk("grippe", 0, "La grippe est une infection virale avec fièvre, courbatures et fatigue intense."),
]


@pytest.fixture
def index():
    return BM25Index(CHUNKS)


def test_tokenize_removes_stopwords_and_lowercases():
    tokens = tokenize("Le Diabète de type 2 est une maladie")
    assert "le" not in tokens
    assert "diabète" in tokens


def test_search_ranks_relevant_doc_first(index):
    results = index.search("glycémie insuline diabète", k=2)
    assert results[0][0].doc_id == "diabete"
    assert results[0][1] > results[1][1]


def test_index_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search("bronches sifflements", k=1)[0][0].doc_id == "asthme"


def test_evaluate_retrieval_perfect_case(index):
    eval_set = [
        {"question": "glycémie et insuline", "expected_doc": "diabete"},
        {"question": "inflammation des bronches", "expected_doc": "asthme"},
    ]
    report = evaluate_retrieval(index, eval_set, k=2)
    assert report["hit@2"] == 1.0
    assert report["mrr"] == 1.0


def test_evaluate_retrieval_miss_scores_zero(index):
    eval_set = [{"question": "fracture du fémur", "expected_doc": "inexistant"}]
    report = evaluate_retrieval(index, eval_set, k=2)
    assert report["hit@2"] == 0.0
    assert report["mrr"] == 0.0


def test_extract_citations_dedupes_and_orders():
    assert extract_citations("Fait A [2]. Fait B [1][2], fait C [3].") == [2, 1, 3]
    assert extract_citations("Aucune citation ici.") == []


def test_format_sources_numbers_chunks(index):
    results = index.search("grippe fièvre", k=2)
    formatted = format_sources(results)
    assert formatted.startswith("[1] (document : grippe)")
    assert "[2] (document :" in formatted
