# Rerankers

[Docs](../README.md) · [Rerank results](../how-to/rerank.md) · [Generators](generators.md)

A reranker implements `ragout.Reranker`:

```go
Rerank(ctx context.Context, query string, candidates []ScoredChunk, topN int) ([]ScoredChunk, error)
```

The engine calls it after fusion, with the recall list and `topN`. The returned slice is what the generator sees. An error fails the query. An empty candidate list or a non-positive `topN` returns an empty slice.

The same package holds fusion helpers. `ReciprocalRankFusion` is what the engine calls when both searches hit. `ConvexScoreCombination`, `MinMaxNormalization`, and `LogisticSigmoidNormalization` are available for callers who want to blend raw scores themselves. The engine does not.

## Cohere

```go
rr := reranker.NewCohereReranker(apiKey,
    reranker.WithCohereModel("rerank-v3.5"),
    reranker.WithCohereHTTPClient(client),
)
```

Default model `rerank-v3.5`, timeout 10 seconds, URL `https://api.cohere.ai/v1/rerank`. The request sends the query, the chunk texts, and `topN`. Results come back as indexes into that list plus a `relevance_score`, which is written to `Score`. The original chunk, including its dense and sparse scores, is preserved.

## MMR

```go
rr := reranker.NewMMRReranker(emb, 0.7)
```

`lambda` in `(0, 1]` is kept. Any other value, including 0, becomes 0.7.

`Rerank` embeds the query, then calls `MaximalMarginalRelevance`. Each pick maximizes

```
lambda * similarity(query, chunk) - (1 - lambda) * max similarity(chunk, selected)
```

Similarity is cosine over chunk embeddings. If the embedder is nil, the query embedding is missing, or a chunk has no embedding, the candidate's current `Score` is used as its similarity to the query. You can call `MaximalMarginalRelevance` with your own query vector and `MMROptions`. A lambda outside `[0, 1]` on that direct call returns an error. A lambda of 0 on that call is treated as "use 0.7".

## Write your own

Return at most `topN` chunks, best first. The generator assumes that order when it spends the token budget: it tries earlier chunks first. Set `Score` to whatever you want traces and logs to show. Leave `Chunk` intact so citations still have text and metadata.

A cross-encoder behind a generic HTTP `/rerank` endpoint is a [roadmap](../roadmap.md) item.

## Next

[Generators](generators.md) turn the reranked list into a cited answer.
