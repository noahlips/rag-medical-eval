"""Paragraph-aware chunking with overlap."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(text: str, doc_id: str, max_chars: int = 1200, overlap_paragraphs: int = 1) -> list[Chunk]:
    """Group consecutive paragraphs into chunks of at most max_chars.

    Consecutive chunks share `overlap_paragraphs` paragraphs so that
    answers spanning a paragraph boundary stay retrievable.
    """
    paragraphs = split_paragraphs(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > max_chars:
            chunks.append(Chunk(doc_id, len(chunks), "\n\n".join(current)))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) for p in current)
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        chunks.append(Chunk(doc_id, len(chunks), "\n\n".join(current)))
    return chunks


def chunk_corpus(corpus_dir: Path, max_chars: int = 1200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_text(text, doc_id=path.stem, max_chars=max_chars))
    return chunks
