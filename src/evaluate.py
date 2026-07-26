"""Evaluation harness for the RAG pipeline.

Two levels:

1. Retrieval (always runs, deterministic, no LLM needed)
   - hit@k : the expected document appears in the top-k retrieved chunks
   - MRR   : mean reciprocal rank of the expected document

2. Generation (runs only if an Ollama server is reachable)
   - citation validity : every [n] cited actually exists among retrieved sources
   - citation coverage : the answer cites at least one source
   - groundedness (LLM-as-judge) : a second model checks the answer is
     supported by the cited chunks

Usage:
    python -m src.evaluate [--k 5] [--with-generation]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.index import BM25Index
from src.rag import OLLAMA_URL, answer_question, generate

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_FILE = BASE_DIR / "data" / "eval" / "questions.json"
RESULTS_DIR = BASE_DIR / "results"

JUDGE_PROMPT = """Voici des extraits de documents et une réponse générée. La réponse est-elle \
entièrement soutenue par les extraits (aucune information inventée) ? Réponds uniquement par \
OUI ou NON.

Extraits :
{sources}

Réponse à vérifier :
{answer}

Verdict (OUI/NON) :"""


def ollama_available() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def evaluate_retrieval(index: BM25Index, eval_set: list[dict], k: int = 5) -> dict:
    hits, reciprocal_ranks, details = 0, [], []
    for item in eval_set:
        results = index.search(item["question"], k=k)
        retrieved_docs = [chunk.doc_id for chunk, _ in results]
        try:
            rank = retrieved_docs.index(item["expected_doc"]) + 1
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            rank = None
            reciprocal_ranks.append(0.0)
        details.append({"question": item["question"], "expected": item["expected_doc"], "rank": rank})

    return {
        "k": k,
        "n_questions": len(eval_set),
        f"hit@{k}": round(hits / len(eval_set), 4),
        "mrr": round(sum(reciprocal_ranks) / len(eval_set), 4),
        "details": details,
    }


def evaluate_generation(index: BM25Index, eval_set: list[dict], k: int = 4) -> dict:
    n_valid, n_cited, n_grounded, answers = 0, 0, 0, []
    for item in eval_set:
        result = answer_question(item["question"], index, k=k)
        n_valid += result["valid_citations"]
        n_cited += bool(result["citations"])

        sources = "\n\n".join(
            chunk.text for chunk, _ in index.search(item["question"], k=k)
        )
        verdict = generate(JUDGE_PROMPT.format(sources=sources, answer=result["answer"]))
        grounded = verdict.strip().upper().startswith("OUI")
        n_grounded += grounded

        answers.append({**result, "grounded": grounded})

    n = len(eval_set)
    return {
        "citation_validity": round(n_valid / n, 4),
        "citation_coverage": round(n_cited / n, 4),
        "groundedness": round(n_grounded / n, 4),
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()

    index = BM25Index.load()
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval": evaluate_retrieval(index, eval_set, k=args.k),
    }

    print(f"Retrieval  hit@{args.k}={report['retrieval'][f'hit@{args.k}']}  MRR={report['retrieval']['mrr']}")

    if args.with_generation:
        if not ollama_available():
            print("Ollama unreachable, skipping generation evaluation.")
        else:
            report["generation"] = evaluate_generation(index, eval_set)
            g = report["generation"]
            print(
                f"Generation citation_validity={g['citation_validity']} "
                f"citation_coverage={g['citation_coverage']} groundedness={g['groundedness']}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
