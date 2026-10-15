# Measure retrieval

[Docs](../README.md) · [Give chunks context](chunk-context.md) · [Rewrite questions](query-rewrite.md)

Every option that changes retrieval (a contextualizer, a query transformer, a reranker, an expander, chunk size) helps on some data and not on other data. The `eval` package measures it on yours.

## Retrieval without generation

`Engine.Retrieve` runs everything `Query` does before generation: rewrite, search, fusion, rerank, expansion. It returns an `Answer` with `Retrieved`, `Question`, and `Searches` filled, and no `Text`. It needs no generator, so it is also the call for retrieval-only services.

```go
answer, err := engine.Retrieve(ctx, "How are leaders elected?")
```

## Write cases

A case is a question and the text a relevant chunk contains:

```go
cases := []eval.Case{
    {Question: "How are leaders elected?", Expected: []string{"elects a leader"}},
    {Question: "Compare Raft and Paxos", Expected: []string{"Raft elects", "proposers and acceptors"}},
    {Question: "and how long is a term?", History: history, Expected: []string{"term lasts"}},
}
```

A retrieved chunk counts as relevant if it contains any `Expected` entry, ignoring case and whitespace. Use several entries when the answer is spread over several chunks. Matching on text rather than chunk IDs keeps the cases valid when you re-chunk or re-ingest.

Twenty to fifty questions that real users ask are enough to see a difference.

## Run and compare

```go
base, _ := eval.Run(ctx, plainEngine, cases, 5)
withRewrite, _ := eval.Run(ctx, rewriteEngine, cases, 5)
fmt.Println(base)        // cases=30 k=5 hit_rate=0.733 recall=0.650 mrr=0.581 misses=8
fmt.Println(withRewrite)
```

| Field | Meaning |
| --- | --- |
| `HitRate` | Share of questions with at least one relevant chunk in the top `k`. |
| `Recall` | Mean share of each question's `Expected` entries found in the top `k`. |
| `MRR` | Mean of 1 / rank of the first relevant chunk, 0 when there is none. |
| `Misses` | The questions with no relevant chunk in the top `k`. Read these first. |

A question that retrieves nothing counts as a miss. Any other error stops the run.

Ingest the same files into both engines. For `k`, use the `topN` you will run in production.

## Integration tests

The `integration` package runs the library against real services. Each test is skipped unless its environment variable is set, so `go test ./...` stays offline.

| Variable | Service |
| --- | --- |
| `RAGOUT_IT_QDRANT_URL` | Qdrant, e.g. `http://localhost:6333` |
| `RAGOUT_IT_CHROMA_URL` | Chroma, e.g. `http://localhost:8000` |
| `RAGOUT_IT_OLLAMA_URL`, `RAGOUT_IT_OLLAMA_MODEL` | Ollama and a chat model |
| `RAGOUT_IT_OLLAMA_EMBED_MODEL` | Optional Ollama embedding model |
| `RAGOUT_IT_OPENAI_API_KEY` | OpenAI. Spends a few cents. |

```bash
docker run -d -p 6333:6333 qdrant/qdrant
RAGOUT_IT_QDRANT_URL=http://localhost:6333 go test ./integration -v
```

Each store run uses a new collection and deletes it afterwards.

## Next

[Traces and metrics](observability.md) show the same stages in production.
