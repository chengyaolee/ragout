# Rewrite questions

[Docs](../README.md) · [Pipeline](../concepts/pipeline.md) · [Hybrid retrieval](../concepts/hybrid-retrieval.md)

The question a user types is not always a good search query. A follow-up such as "and in 2023?" means nothing without the conversation. A short question misses chunks that use other words. A comparison needs one lookup per side.

A `QueryTransformer` rewrites the question before retrieval. With none set, the question is searched as written and no model is called.

## Built-in transformers

The `transform` package has four, each one model call, and `Chain` to combine them.

| Constructor | What it adds |
| --- | --- |
| `transform.Condense(llm)` | Rewrites a follow-up into a standalone question from the history. Without history it makes no call. |
| `transform.MultiQuery(llm, n)` | Up to `n` rephrasings (default 3), searched on every index. |
| `transform.Decompose(llm, n)` | Up to `n` sub-questions (default 4) for a compound question. A simple question adds nothing. |
| `transform.HyDE(llm)` | A hypothetical answer passage, searched on the vector store only. |
| `transform.Chain(t...)` | Runs transformers in order. Each sees the latest standalone question. |

```go
llm := generator.NewOpenAIGenerator(os.Getenv("OPENAI_API_KEY"))
engine, err := ragout.NewEngine(
    // reader, chunker, embedder, stores, generator...
    ragout.WithQueryTransformer(transform.Chain(
        transform.Condense(llm),
        transform.MultiQuery(llm, 3),
    )),
    ragout.WithTransformTimeout(3*time.Second),
)
```

HyDE helps most with weaker embedders. A model that does not know your private data can write a misleading passage, so measure it before relying on it.

## Conversations

The engine keeps no state between queries. Pass the earlier turns with each call, oldest first:

```go
answer, err := engine.Query(ctx, "and in 2023?", ragout.WithHistory([]ragout.Turn{
    {Role: "user", Content: "What was Acme's revenue in 2024?"},
    {Role: "assistant", Content: "$45M [1]"},
}))
```

`Condense` reads the last six turns. By default the history only shapes retrieval: the generator answers the standalone question. To also put earlier turns in the answer prompt, build the generator with `generator.WithOpenAIHistoryTurns(n)` or `generator.WithOllamaHistoryTurns(n)`. History then takes at most a quarter of the token budget, newest turns first, and the prompt marks it as not citable.

## What the engine does with a rewrite

1. The standalone question is always searched, on every configured index. Rewrites are added to it, never substituted for it.
2. Duplicate searches are dropped.
3. Every dense text is embedded in one `EmbedBatch` call. Every search runs concurrently. All result lists are fused once, with reciprocal rank fusion by default (`WithFusion` replaces it).
4. The reranker and the generator see the standalone question.

`Answer.Question` is that standalone question. `Answer.Searches` lists every search requested, the question first.

## When the model fails

Rewriting is on the query's critical path, so failures fall back instead of failing the query. If the transformer returns an error, or `WithTransformTimeout` runs out, the query continues with whatever the transformer returned, which is at least the original question. Cancelling the query's own context still aborts.

The failure is reported three ways: `Answer.TransformErr`, the `ragout_query_transform_failures_total` counter, and an error status on the `query.transform` span.

## Write your own

```go
type QueryTransformer interface {
    Transform(ctx context.Context, question string, history []Turn) (Rewrite, error)
}

type Rewrite struct {
    Question string   // standalone question; empty keeps the original
    Searches []Search // added searches
}

type Search struct {
    Text   string
    Target SearchTarget // SearchAll, SearchDense, or SearchSparse
}
```

The built-ins read model output as plain lines rather than JSON, and skip `<think>` blocks, numbering, bullets, quotes, and preamble lines, so any chat model works.

## Next

[Measure retrieval](evaluate.md) shows whether a transformer helps on your data.
