"""Evaluation harness for the RAG pipeline.

The evaluation set mixes two kinds of questions:

- questions whose answer is in the corpus, tagged with the document that
  should answer them and with a ``type`` (definition, mecanisme,
  prise_en_charge, detail, langage_courant)
- questions whose answer is deliberately absent, with ``expected_doc``
  set to ``null``

The two are scored differently. Retrieval metrics only make sense for the
first group, so the second is excluded from hit@k and MRR and is scored on
the generation side instead: did the model say it did not know, or did it
answer anyway.

Two levels:

1. Retrieval (always runs, deterministic, no LLM needed)
   - hit@k : the expected document appears in the top-k retrieved chunks
   - MRR   : mean reciprocal rank of the expected document
   - the same two, broken down by question type

2. Generation (runs only if an Ollama server is reachable)
   - citation validity : every [n] cited actually exists among retrieved sources
   - citation coverage : the answer cites at least one source
   - groundedness (LLM-as-judge) : a second model checks the answer is
     supported by the cited chunks
   - abstention : on out-of-corpus questions, the model declined to answer

Usage:
    python -m src.evaluate [--k 5] [--with-generation]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.index import BM25Index
from src.rag import OLLAMA_URL, answer_question, generate, is_abstention

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


def split_eval_set(eval_set: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate questions answerable from the corpus from the others."""
    answerable = [item for item in eval_set if item.get("expected_doc")]
    unanswerable = [item for item in eval_set if not item.get("expected_doc")]
    return answerable, unanswerable


def _aggregate(ranks: list[int | None]) -> dict:
    """Hit rate and MRR from a list of ranks, None meaning the document was missed."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit": 0.0, "mrr": 0.0}
    hits = sum(1 for rank in ranks if rank is not None)
    reciprocal = sum(1.0 / rank for rank in ranks if rank is not None)
    return {"n": n, "hit": round(hits / n, 4), "mrr": round(reciprocal / n, 4)}


def evaluate_retrieval(index: BM25Index, eval_set: list[dict], k: int = 5) -> dict:
    answerable, unanswerable = split_eval_set(eval_set)

    details: list[dict] = []
    all_ranks: list[int | None] = []
    ranks_by_type: dict[str, list[int | None]] = defaultdict(list)

    for item in answerable:
        results = index.search(item["question"], k=k)
        retrieved_docs = [chunk.doc_id for chunk, _ in results]
        try:
            rank = retrieved_docs.index(item["expected_doc"]) + 1
        except ValueError:
            rank = None

        qtype = item.get("type", "sans_type")
        all_ranks.append(rank)
        ranks_by_type[qtype].append(rank)
        details.append(
            {
                "question": item["question"],
                "type": qtype,
                "expected": item["expected_doc"],
                "rank": rank,
            }
        )

    overall = _aggregate(all_ranks)
    by_type = {}
    for qtype in sorted(ranks_by_type):
        agg = _aggregate(ranks_by_type[qtype])
        by_type[qtype] = {"n": agg["n"], f"hit@{k}": agg["hit"], "mrr": agg["mrr"]}

    return {
        "k": k,
        "n_questions": overall["n"],
        "n_excluded_unanswerable": len(unanswerable),
        f"hit@{k}": overall["hit"],
        "mrr": overall["mrr"],
        "by_type": by_type,
        "details": details,
    }


def evaluate_generation(index: BM25Index, eval_set: list[dict], k: int = 4) -> dict:
    answerable, unanswerable = split_eval_set(eval_set)

    n_valid, n_cited, n_grounded, answers = 0, 0, 0, []
    for item in answerable:
        result = answer_question(item["question"], index, k=k)
        n_valid += result["valid_citations"]
        n_cited += bool(result["citations"])

        sources = "\n\n".join(chunk.text for chunk, _ in index.search(item["question"], k=k))
        verdict = generate(JUDGE_PROMPT.format(sources=sources, answer=result["answer"]))
        grounded = verdict.strip().upper().startswith("OUI")
        n_grounded += grounded

        answers.append({**result, "type": item.get("type"), "grounded": grounded})

    n = len(answerable)
    report = {
        "n_answerable": n,
        "citation_validity": round(n_valid / n, 4) if n else 0.0,
        "citation_coverage": round(n_cited / n, 4) if n else 0.0,
        "groundedness": round(n_grounded / n, 4) if n else 0.0,
        "answers": answers,
    }

    # Questions dont la reponse n'est pas dans le corpus : la bonne reponse est
    # de dire qu'on ne sait pas, repondre quand meme est une hallucination.
    if unanswerable:
        n_abstained, out_of_corpus = 0, []
        for item in unanswerable:
            result = answer_question(item["question"], index, k=k)
            abstained = is_abstention(result["answer"])
            n_abstained += abstained
            out_of_corpus.append({**result, "type": item.get("type"), "abstained": abstained})

        report["n_unanswerable"] = len(unanswerable)
        report["abstention_rate"] = round(n_abstained / len(unanswerable), 4)
        report["out_of_corpus_answers"] = out_of_corpus

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()

    index = BM25Index.load()
    eval_set = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    retrieval = evaluate_retrieval(index, eval_set, k=args.k)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "retrieval": retrieval}

    hit_key = f"hit@{args.k}"
    print(
        f"Retrieval  {hit_key}={retrieval[hit_key]}  MRR={retrieval['mrr']}  "
        f"({retrieval['n_questions']} questions, "
        f"{retrieval['n_excluded_unanswerable']} hors corpus exclues)"
    )
    for qtype, scores in retrieval["by_type"].items():
        print(f"  {qtype:<18} n={scores['n']:<4} {hit_key}={scores[hit_key]:<8} MRR={scores['mrr']}")

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
            if "abstention_rate" in g:
                print(
                    f"           abstention_rate={g['abstention_rate']} "
                    f"sur {g['n_unanswerable']} questions hors corpus"
                )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
