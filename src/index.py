"""BM25 index over the chunked corpus.

Usage:
    python -m src.index          # build and save the index
"""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.chunking import Chunk, chunk_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"

FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "et", "ou", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "que", "qui", "quoi",
    "dont", "est", "sont", "être", "avoir", "il", "elle", "on", "ce", "cette", "ces",
    "se", "sa", "son", "ses", "plus", "pas", "ne", "comme", "mais", "aussi",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zà-ÿ0-9]+", text.lower())
    return [t for t in tokens if t not in FRENCH_STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self, f)
        (index_dir / "stats.json").write_text(
            json.dumps({"n_chunks": len(self.chunks), "n_docs": len({c.doc_id for c in self.chunks})})
        )

    @staticmethod
    def load(index_dir: Path = INDEX_DIR) -> "BM25Index":
        with (index_dir / "bm25.pkl").open("rb") as f:
            return pickle.load(f)


def build_index() -> BM25Index:
    chunks = chunk_corpus(DATA_DIR / "corpus")
    if not chunks:
        raise FileNotFoundError("Empty corpus. Run `python -m src.corpus` first.")
    index = BM25Index(chunks)
    index.save()
    print(f"✓ Indexed {len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")
    return index


if __name__ == "__main__":
    build_index()
