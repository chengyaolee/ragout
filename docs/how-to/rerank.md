# Rerank results

[Docs](../README.md) · [Hybrid retrieval](../concepts/hybrid-retrieval.md) · [Rerankers](../components/rerankers.md)

Retrieval returns up to `topKRecall` chunks (default 50). A reranker reorders that list and the engine keeps `topN` (default 5) for the generator. Set it with `WithReranker`.

With no reranker, the engine keeps the first `topN` of the fused ranking and does not call out.

## Cohere

`reranker.NewCohereReranker` posts the question and the chunk texts to `https://api.cohere.ai/v1/rerank`. The default model is `rerank-v3.5`. `WithCohereModel` changes it.

```go
engine, err := ragout.NewEngine(
    // reader, chunker, embedder, stores, generator...
    ragout.WithReranker(reranker.NewCohereReranker(os.Getenv("COHERE_API_KEY"))),
    ragout.WithTopNRerank(8),
)
```

The relevance score from Cohere replaces `ScoredChunk.Score` on the returned chunks. A reranker error fails the query.

## Maximal marginal relevance

MMR picks chunks that are close to the question and far from chunks already picked. Use it when the fused list is full of near-duplicates and you want the prompt to cover more of the document.

```go
ragout.WithReranker(reranker.NewMMRReranker(emb, 0.7))
```

`lambda` weights relevance against diversity:

```
score = lambda * similarity(query, chunk) - (1 - lambda) * max similarity(chunk, already picked)
```

`lambda` closer to 1 favors the question. `lambda` closer to 0 favors chunks that differ from what is already selected. A `lambda` outside `(0, 1]` is replaced with `0.7`.

The reranker embeds the question with the embedder you pass, then compares chunk embeddings with cosine similarity. Chunks that have no embedding fall back to their current `Score`. Pass the same embedder the engine uses so the vectors match.

`reranker.MaximalMarginalRelevance` is the same algorithm if you already have a query vector and want to call it outside the engine.

## A local cross-encoder

A generic HTTP reranker for BGE or Hugging Face text-embeddings-inference is on the [roadmap](../roadmap.md). It is not in this release. Cohere and MMR are the two `Reranker` implementations you can pass to `WithReranker` today. Anything that implements `Rerank(ctx, query, candidates, topN)` can be passed as well.

## Next

[Traces and metrics](observability.md) records the rerank stage's latency.
