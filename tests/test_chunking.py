from itertools import pairwise

from src.chunking import chunk_text, split_paragraphs


def test_split_paragraphs_ignores_blank_blocks():
    text = "Premier paragraphe.\n\n\n\nDeuxième.\n\n   \n\nTroisième."
    assert split_paragraphs(text) == ["Premier paragraphe.", "Deuxième.", "Troisième."]


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraphe {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, doc_id="doc", max_chars=1000)
    assert all(len(c.text) <= 1400 for c in chunks)  # max + one paragraph tolerance
    assert len(chunks) > 1


def test_chunks_overlap():
    text = "\n\n".join(f"Paragraphe numéro {i} " + "y" * 400 for i in range(6))
    chunks = chunk_text(text, doc_id="doc", max_chars=900, overlap_paragraphs=1)
    for previous, current in pairwise(chunks):
        last_paragraph = previous.text.split("\n\n")[-1]
        assert last_paragraph in current.text


def test_chunk_ids_are_sequential():
    text = "\n\n".join("z" * 500 for _ in range(5))
    chunks = chunk_text(text, doc_id="mydoc", max_chars=600)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "mydoc" for c in chunks)
