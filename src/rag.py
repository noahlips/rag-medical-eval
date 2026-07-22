"""RAG answering with mandatory source citations.

Generation goes through a local Ollama server. The answer must cite its
sources as [1], [2]... referring to the numbered retrieved chunks; the
citation parser is used both at answer time and by the evaluation.

Usage:
    python -m src.rag "Quels sont les symptômes du diabète de type 2 ?"
"""

import re
import sys

import requests

from src.chunking import Chunk
from src.index import BM25Index

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

PROMPT_TEMPLATE = """Tu es un assistant documentaire médical. Réponds à la question en t'appuyant \
UNIQUEMENT sur les sources numérotées ci-dessous. Cite chaque affirmation avec le numéro de sa \
source entre crochets, par exemple [1] ou [2]. Si les sources ne permettent pas de répondre, \
dis-le explicitement et ne réponds pas de mémoire.

{sources}

Question : {question}

Réponse (avec citations [n]) :"""


def format_sources(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (chunk, _score) in enumerate(results, start=1):
        blocks.append(f"[{i}] (document : {chunk.doc_id})\n{chunk.text}")
    return "\n\n".join(blocks)


def extract_citations(answer: str) -> list[int]:
    """Return the distinct source numbers cited in an answer, in order."""
    seen: list[int] = []
    for match in re.findall(r"\[(\d{1,2})\]", answer):
        n = int(match)
        if n not in seen:
            seen.append(n)
    return seen


def generate(prompt: str, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def answer_question(
    question: str, index: BM25Index, k: int = 4, model: str = DEFAULT_MODEL
) -> dict:
    results = index.search(question, k=k)
    prompt = PROMPT_TEMPLATE.format(sources=format_sources(results), question=question)
    answer = generate(prompt, model=model)
    citations = extract_citations(answer)
    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": [chunk.doc_id for chunk, _ in results],
        "citations": citations,
        "valid_citations": all(1 <= c <= len(results) for c in citations),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = answer_question(" ".join(sys.argv[1:]), BM25Index.load())
    print(result["answer"])
    print("\nSources :", ", ".join(result["retrieved_docs"]))
