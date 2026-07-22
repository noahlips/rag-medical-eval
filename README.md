# RAG Medical — Retrieval-Augmented Generation with rigorous evaluation

A French-language medical document-retrieval and question-answering pipeline (RAG), built around what most RAG demos skip: **a measurable evaluation harness**.

The corpus is 20 French Wikipedia medical articles (fetched via the official MediaWiki API), chunked and indexed with BM25. Generation runs on a self-hosted LLM through [Ollama](https://ollama.com), with **mandatory source citations** enforced by the prompt and checked programmatically.

## Evaluation results

Retrieval is evaluated on 20 questions with known ground-truth documents (deterministic, no LLM needed):

| Metric | Value |
|--------|-------|
| hit@5 | **1.00** |
| MRR | **0.975** |
| Corpus | 796 chunks / 20 documents |

Generation is evaluated on three axes when an Ollama server is available (`--with-generation`):

- **citation validity** — every `[n]` cited by the model maps to a real retrieved source
- **citation coverage** — the answer actually cites its sources
- **groundedness** — an LLM-as-judge checks the answer against the cited chunks (hallucination detection)

## Pipeline

```
corpus.py (MediaWiki API) → chunking.py (paragraph-aware, overlap) → index.py (BM25)
                                                                          │
        evaluate.py (hit@k, MRR, citations, groundedness)  ←  rag.py (Ollama + citations [n])
```

## Usage

```bash
pip install -r requirements.txt
python -m src.corpus                       # download the corpus
python -m src.index                        # build the BM25 index
python -m src.evaluate --k 5               # retrieval evaluation (no LLM needed)

# With a local Ollama server (ollama pull llama3.2):
python -m src.rag "Quels sont les symptômes du diabète de type 2 ?"
python -m src.evaluate --k 5 --with-generation
```

## Design notes

- **BM25 over embeddings, deliberately** — on a small specialized corpus, lexical retrieval is a strong, fully reproducible baseline; the evaluation harness makes it easy to plug in embeddings and *prove* whether they help, rather than assume it.
- **Citations are a contract** — the prompt forces `[n]` citations, `extract_citations` parses them, and evaluation fails answers whose citations don't exist. Refusal is instructed when sources don't contain the answer.
- **Everything measurable is tested** — 11 pytest cases on chunking, tokenization, indexing, ranking, metrics and the citation parser; CI runs lint + tests on every push.

## Disclaimer

Educational project — not medical advice. Answers reflect the corpus, not clinical guidance.
