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

Run the model yourself and point `reranker.NewHTTP` at it. No API key, no per-call cost.

```bash
brew install llama.cpp   # or build it; any OS works
llama-server -hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --reranking --port 8080
```

```go
ragout.WithReranker(reranker.NewHTTP("")) // bge-reranker-v2-m3 on localhost:8080
```

To try the larger Qwen3-Reranker-4B, serve it and switch one option:

```go
ragout.WithReranker(reranker.NewHTTP("http://localhost:8081/v1/rerank",
    reranker.WithModel(reranker.Qwen3Reranker4B)))
```

Rerank a pool smaller than the recall depth if latency matters: `WithTopKRecall(30)` sends 30 candidates instead of 50. Measured effects on two question sets are in [eval/results](../../eval/results/). Server setup, request shapes, and the vLLM variant are in [Rerankers](../components/rerankers.md).

## Next

[Traces and metrics](observability.md) records the rerank stage's latency.
