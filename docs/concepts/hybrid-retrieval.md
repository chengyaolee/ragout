# Hybrid retrieval

[Docs](../README.md) · [Pipeline](pipeline.md) · [Generation](generation.md)

Keyword search and vector search fail in different places. BM25 hits exact terms and misses paraphrases. Cosine search hits paraphrases and can miss a rare identifier, a SKU, or an error code. Ragout runs both and merges the rankings.

## Dense search

The embedder turns the question into one vector. `VectorStore.SearchDense` returns the `topK` chunks with the highest cosine similarity. The in-memory store, Chroma, and Qdrant all implement that method. The in-memory store computes cosine in process and applies the metadata filter while it scans. Chunks whose embedding length does not match the query vector produce `ErrDimensionMismatch`.

You can call `store.CosineSimilarity` yourself. A zero vector scores 0. Vectors that are already unit length take the dot product.

## Sparse search

`IndexStore.SearchSparse` scores the raw question string. The built-in index is Okapi BM25.

`store.NewBM25Index` uses `k1=1.5` and `b=0.75`. `WithK1` and `WithB` change them. The default tokenizer lowercases and splits on anything that is not a letter or a digit. Pass `WithTokenizer` for a different split. `TermsMatched` reports how many query terms hit the index; the sparse-search span records that number.

BM25 applies the same metadata filter as the vector store, after it has scored candidate chunks.

## Why the scores are not averaged

A cosine similarity sits roughly in `[-1, 1]`. A BM25 score is unbounded. Averaging them lets one side dominate. Reciprocal rank fusion ignores the raw scores and uses rank:

```
score(chunk) = Σ 1 / (k + rank)
```

`rank` starts at 1 for the best hit in that list. `k` defaults to 60 (`WithRRFK`). A chunk that is first in both lists scores `2/(k+1)`. A chunk that appears in only one list still gets that list's term. The engine then keeps the top `topKRecall` fused chunks.

Fusion runs only when both searches returned at least one hit. If the vector search misses and BM25 hits, the BM25 order is the candidate list, and `Score` is still the BM25 score. The symmetric case uses cosine. That is the whole degradation path: configure one store, or configure both and tolerate one side returning nothing.

## Helpers the engine does not call

`reranker.ConvexScoreCombination` blends sigmoid-normalized dense and sparse scores with an alpha weight. `MinMaxNormalization` and `LogisticSigmoidNormalization` are the scaling helpers next to it. The engine does not call them. Wiring blend-by-score in as an engine option is a [roadmap](../roadmap.md) item. Today the query path calls `ReciprocalRankFusion` only.

## After the ranking

The fused list is still a recall set (default 50). The reranker, if you set one, reorders it and cuts to `topN`. With no reranker the engine keeps the first `topN` (default 5) and passes those to the generator. See [Rerank results](../how-to/rerank.md).

## Next

[Generation and citations](generation.md) is what happens to that short list.
