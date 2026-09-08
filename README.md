# RAG on French medical articles, with an evaluation harness

Most RAG demos show you a chatbot answering one question and stop there. The question I wanted to answer was different: how do you know the thing actually works?

So this repo is a small French medical question-answering system, plus the code that measures it.

## The corpus

20 medical articles from French Wikipedia (diabetes, asthma, Parkinson's, epilepsy and so on), pulled through the MediaWiki API. They get split into 796 chunks and indexed with BM25.

Chunking follows paragraph boundaries rather than a fixed character count, grouping whole paragraphs up to 1200 characters. Consecutive chunks share one paragraph, so an answer that straddles a boundary is still findable.

## Measuring retrieval

The evaluation set is 20 questions where I know which document should answer each one. This part is deterministic and needs no LLM, so it runs in CI:

```
hit@5   1.00
MRR     0.975
```

I should be upfront about this: hit@5 of 1.00 is not impressive. Twenty questions across twenty very distinct documents, phrased with vocabulary taken from the articles, is an easy case. The number says the pipeline is wired correctly, not that it is robust. MRR is more informative, and 0.975 means one question does not have its document in first place.

## Measuring generation

Generation runs on a local model through Ollama. The prompt requires the model to cite its sources as `[1]`, `[2]`, and to say so explicitly when the sources do not contain the answer.

Those citations are then checked in code, which is the part I find useful:

- **citation validity**: does every `[n]` in the answer point at a source that was actually retrieved
- **citation coverage**: did the model cite anything at all
- **groundedness**: a second model reads the cited chunks and the answer and judges whether one supports the other

The third one is the weakest. An LLM judge has its own biases and I have not validated it against human annotations. But it gives a signal you can track.

## Why BM25 and not embeddings

This is the choice people will question, and it is deliberate. On a corpus this small and this specialised, lexical search is a strong baseline, fully reproducible, with no model dependency.

More importantly, having the evaluation harness first means embeddings can be plugged in and *measured* rather than assumed to be better. That is the opposite of the usual order, where you reach for embeddings and never check.

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

20 questions is a tiny evaluation set, so the confidence intervals are wide. I wrote the questions myself from the documents, which makes them easier than what a real user would ask. And Wikipedia is not a medical reference: a serious system would use HAS guidelines or something equivalent.

Educational project, not medical advice.
