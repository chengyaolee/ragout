# Rerankers

[Docs](../README.md) · [Rerank results](../how-to/rerank.md) · [Generators](generators.md)

A reranker implements `ragout.Reranker`:

```go
Rerank(ctx context.Context, query string, candidates []ScoredChunk, topN int) ([]ScoredChunk, error)
```

The engine calls it after fusion, with the recall list and `topN`. The returned slice is what the generator sees. An error fails the query. An empty candidate list or a non-positive `topN` returns an empty slice.

The same package holds fusion helpers. `ReciprocalRankFusion` is the two-list form of the engine's default fusion. `ConvexScoreCombination`, `MinMaxNormalization`, and `LogisticSigmoidNormalization` are available for callers who want to blend raw scores themselves; pass a `ragout.FusionStrategy` to `WithFusion` to use them in the engine.

## Local cross-encoder (HTTP)

```go
rr := reranker.NewHTTP("http://localhost:8080/v1/rerank",
    reranker.WithModel(reranker.BGERerankerV2M3), // the default
    reranker.WithAPI(reranker.CohereAPI),         // the default
    reranker.WithAPIKey(key),                     // optional
    reranker.WithHTTPClient(client),
)
```

`NewHTTP` calls a rerank endpoint you run yourself. It sends the question and each candidate's indexed text (its title and section breadcrumb plus its content), and writes the model's relevance score to `Score`. An empty endpoint means `reranker.DefaultEndpoint`, llama-server's default port. The HTTP timeout is 2 minutes, since a local server may load the model on the first call. A non-200 status returns an error carrying the server's message.

| Model | Preset | License | Size | Notes |
| --- | --- | --- | --- | --- |
| BAAI/bge-reranker-v2-m3 | `reranker.BGERerankerV2M3` (default) | Apache 2.0 | 0.6B | Multilingual. Returns raw logits. |
| Qwen/Qwen3-Reranker-4B | `reranker.Qwen3Reranker4B` | Apache 2.0 | 4B | Stronger and slower. Returns 0 to 1. |

The servers they were tested with:

```bash
llama-server -hf gpustack/bge-reranker-v2-m3-GGUF:Q8_0 --reranking --port 8080
llama-server -hf Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp:Q4_K_M --reranking --pooling rank --embedding --port 8081
```

Request shapes:

| `API` | Request | Response | Servers |
| --- | --- | --- | --- |
| `CohereAPI` | `{model, query, documents, top_n}` | `{results: [{index, relevance_score}]}` | llama.cpp, vLLM, Infinity, Jina, Cohere |
| `TEIAPI` | `{query, texts}` | `[{index, score}]` | Hugging Face text-embeddings-inference |

Either response shape is read whichever API is set.

Qwen3-Reranker scores a yes/no answer from a chat template. llama.cpp applies that template itself, so `Qwen3Reranker4B` sends raw text. vLLM leaves it to the client: use `reranker.Qwen3Reranker(name, instruction)`, which wraps the query and documents here and takes a task-specific instruction (the model card says a written-for-your-task English instruction typically adds 1 to 5%). Do not combine the two: wrapping twice lowers the scores. TEI does not serve Qwen3-Reranker.

jina-reranker-v3 is not offered as a preset because its license (CC BY-NC 4.0) does not allow commercial use without a separate agreement. Any model your server hosts works with `WithModel(reranker.Model{Name: "..."})`.

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

## Next

[Generators](generators.md) turn the reranked list into a cited answer.
