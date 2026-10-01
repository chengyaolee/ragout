# Streaming

[Docs](../README.md) · [Generation](generation.md) · [Pipeline](pipeline.md)

Retrieval finishes before the first token. All three query methods run that work up front, fill `Answer.Retrieved` and `Answer.Sources`, and then differ only in how you read the model output.

## Query

`Query` drains the token stream and returns the finished `Answer`. Use it when you want the whole string.

```go
answer, err := engine.Query(ctx, "How does ingest roll back a failed write?")
if err != nil {
    return err
}
fmt.Println(answer.Text)
```

A failure during retrieval or generation returns that error and whatever `Answer` was filled in before the failure. An empty question is `ErrEmptyQuery`. Nothing retrieved is `ErrNoResults`.

## QueryStream

`QueryStream` calls your callback once per token, then returns the same `Answer` with `Text` filled in.

```go
answer, err := engine.QueryStream(ctx, question, func(token string) error {
    fmt.Print(token)
    return nil
})
```

A nil callback is `ErrNilCallback`. If the callback returns an error, streaming stops and that error is returned. `answer.Text` includes the tokens already received.

## QueryIter

`QueryIter` returns the `Answer` pointer and a Go 1.23 `iter.Seq2[string, error]`. Sources are set before you range. `Text` grows as you pull tokens, and the iterator's `defer` writes the final string onto `Answer.Text` when the loop ends, including when you `break`.

```go
answer, tokens := engine.QueryIter(ctx, question)
for token, err := range tokens {
    if err != nil {
        return err
    }
    fmt.Print(token)
}
fmt.Println(answer.Text)
```

On a failure that happens before generation, the iterator yields a single `("", err)` pair. `Query` and `QueryStream` are thin wrappers over this iterator.

Cancelling `ctx` stops the embedder, the stores, and the generator's HTTP stream. The span for `query.generation` records `first_token_latency_ms` when the first token arrives. `prompt_tokens` is a separate estimate, about four characters per token, over the question and each source chunk's text. It is not the budgeter's tokenizer count.

## Next

[Traces and metrics](../how-to/observability.md) lists every span and histogram.
