# RAG on French medical articles, with an evaluation harness

Most RAG demos show you a chatbot answering one question and stop there. The question I wanted to answer was different: how do you know the thing actually works?

So this repo is a small French medical question-answering system, plus the code that measures it.

## The corpus

20 medical articles from French Wikipedia (diabetes, asthma, Parkinson's, epilepsy and so on), pulled through the MediaWiki API. They get split into 796 chunks and indexed with BM25.

Chunking follows paragraph boundaries rather than a fixed character count, grouping whole paragraphs up to 1200 characters. Consecutive chunks share one paragraph, so an answer that straddles a boundary is still findable.

## Measuring retrieval

The evaluation set is 110 questions: 100 where I know which document should answer each one, and 10 whose answer is deliberately absent from the corpus. Each answerable question carries a type, five per document, so that a single average does not hide where the system actually fails.

The out-of-corpus questions are excluded from hit@k and MRR, since a retrieval metric has no meaning when no document is correct. They are scored on the generation side instead, as an abstention rate.

This part is deterministic and needs no LLM, so it runs in CI:

```
hit@5   0.92     MRR  0.880     (100 questions)

definition         n=20   hit@5 1.00   MRR 1.000
mecanisme          n=20   hit@5 1.00   MRR 1.000
detail             n=20   hit@5 1.00   MRR 1.000
prise_en_charge    n=20   hit@5 1.00   MRR 0.950
langage_courant    n=20   hit@5 0.60   MRR 0.452
```

The breakdown is the whole point. Four categories out of five are saturated at 1.00, and the aggregate of 0.92 is entirely carried by one category: questions phrased the way a patient would phrase them, with no vocabulary borrowed from the article.

An earlier version of this evaluation used 20 questions and reported hit@5 of 1.00. That number was an artefact. The questions reused the words of the documents, which is exactly what BM25 matches on, so the metric measured how I had written the questions rather than how the system retrieves.

The eight failures are all of that kind:

| question | expected document |
|---|---|
| "J'ai soif en permanence et je vais aux toilettes toute la nuit" | diabete-de-type-2 |
| "Mon père a la main qui tremble quand il ne fait rien et il marche à tout petits pas" | maladie-de-parkinson |
| "Mes chevilles gonflent le soir et je suis essoufflé dès que je monte un étage" | insuffisance-cardiaque |
| "J'ai froid tout le temps, je suis fatiguée et j'ai grossi sans rien changer" | hypothyroidie |

None of these share a rare term with their document. "Soif" and "toilettes" do not appear where "polyurie" and "polydipsie" do. This is the known failure mode of lexical search, and it is now measured rather than asserted.

## Measuring generation

Generation runs on a local model through Ollama. The prompt requires the model to cite its sources as `[1]`, `[2]`, and to say so explicitly when the sources do not contain the answer.

Those citations are then checked in code, which is the part I find useful:

- **citation validity**: does every `[n]` in the answer point at a source that was actually retrieved
- **citation coverage**: did the model cite anything at all
- **groundedness**: a second model reads the cited chunks and the answer and judges whether one supports the other
- **abstention rate**: on the 10 questions whose answer is not in the corpus, did the model say so, or did it answer anyway

The third one is the weakest. An LLM judge has its own biases and I have not validated it against human annotations. But it gives a signal you can track.

The fourth is the one most demos never measure. Retrieval always returns its top-k chunks, including for a question about Crohn's disease in a corpus that contains nothing about it. What matters then is whether the model uses those irrelevant chunks anyway. Answering under those conditions is a hallucination, and abstaining is the correct behaviour.

## Why BM25 and not embeddings

This is the choice people will question, and it is deliberate. On a corpus this small and this specialised, lexical search is a strong baseline, fully reproducible, with no model dependency.

More importantly, having the evaluation harness first means embeddings can be plugged in and *measured* rather than assumed to be better. That is the opposite of the usual order, where you reach for embeddings and never check.

The breakdown above now says precisely what an embedding model would have to improve: `langage_courant` at 0.60, without regressing the four categories currently at 1.00. That is a testable claim rather than a preference, and it is the next step on this repo.

## Running it

```bash
pip install -r requirements.txt
python -m src.corpus          # download the articles
python -m src.index           # build the BM25 index
python -m src.evaluate --k 5  # retrieval metrics, no LLM needed
```

With Ollama running (`ollama pull llama3.2`):

```bash
python -m src.rag "Quels sont les symptômes du diabète de type 2 ?"
python -m src.evaluate --k 5 --with-generation
```

## Limits

100 answerable questions is still a small evaluation set. At 0.92, the 95% confidence interval is roughly 0.85 to 0.96, and each per-type figure rests on 20 questions only, so a category at 1.00 means "no failure observed in 20 tries", not "no failure possible".

I wrote the questions myself, which biases them. The `langage_courant` category is an attempt to counter that bias, but it remains my idea of how a patient writes, not an observation of real queries.

And Wikipedia is not a medical reference: a serious system would use HAS guidelines or something equivalent.

Educational project, not medical advice.
