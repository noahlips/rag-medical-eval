"""Dense retrieval over the chunked corpus, with a multilingual sentence embedding model.

Same interface as BM25Index (search, save, load), so the evaluation can swap one
retriever for the other without any other change.

Usage:
    python -m src.embeddings     # build and save the index
"""

import json
import pickle
from pathlib import Path

import numpy as np

from src.chunking import Chunk, chunk_corpus
from src.index import DATA_DIR, INDEX_DIR

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _load_model(name: str):
    # Imported here rather than at the top: sentence-transformers pulls in torch,
    # which the BM25 pipeline and the test suite do not need.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class EmbeddingIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = MODEL_NAME,
        embeddings: np.ndarray | None = None,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self._model = None
        self.embeddings = embeddings if embeddings is not None else self._encode([c.text for c in chunks])

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return vectors.astype(np.float32)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vector = self._encode([query])[0]
        # Vectors are L2-normalised, so the dot product is the cosine similarity.
        scores = self.embeddings @ query_vector
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with (index_dir / "embedding_chunks.pkl").open("wb") as f:
            pickle.dump(self.chunks, f)
        (index_dir / "embedding_meta.json").write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "n_chunks": len(self.chunks),
                    "dim": int(self.embeddings.shape[1]),
                    "max_seq_length": int(self.model.max_seq_length),
                }
            )
        )

    @staticmethod
    def load(index_dir: Path = INDEX_DIR) -> "EmbeddingIndex":
        meta = json.loads((index_dir / "embedding_meta.json").read_text())
        with (index_dir / "embedding_chunks.pkl").open("rb") as f:
            chunks = pickle.load(f)
        embeddings = np.load(index_dir / "embeddings.npy")
        return EmbeddingIndex(chunks, model_name=meta["model"], embeddings=embeddings)


def build_index() -> EmbeddingIndex:
    chunks = chunk_corpus(DATA_DIR / "corpus")
    if not chunks:
        raise FileNotFoundError("Empty corpus. Run `python -m src.corpus` first.")
    index = EmbeddingIndex(chunks)
    index.save()
    print(f"Embedded {len(chunks)} chunks with {index.model_name}")
    return index


if __name__ == "__main__":
    build_index()
