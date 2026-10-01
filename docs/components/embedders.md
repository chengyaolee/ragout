# Embedders

[Docs](../README.md) · [Chunkers](chunkers.md) · [Stores](stores.md)

An embedder implements `ragout.Embedder`:

```go
EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
Dimension() int
```

The returned slice is aligned with `texts`. A count mismatch is `ragout.ErrCountMismatch`. An empty embedding list on a query is `ragout.ErrEmptyEmbeddings`.

During ingest the engine groups chunks into batches of `WithEmbedBatchSize` (default 32) and runs up to `WithEmbedConcurrency` workers (default 8). One batch failure cancels the rest. During query the engine embeds the question as a single text.

`Dimension` is what you pass to `store.NewQdrantStore`. The in-memory store checks dimensions at search time and returns `ErrDimensionMismatch` when a stored vector differs.

## OpenAI

```go
emb := embedder.NewOpenAIEmbedder(apiKey,
    embedder.WithOpenAIModel("text-embedding-3-small"),
    embedder.WithOpenAIEndpoint("https://api.openai.com/v1/embeddings"),
    embedder.WithOpenAIDimension(1536),
    embedder.WithOpenAIHTTPClient(client),
)
```

Defaults: model `text-embedding-3-small`, endpoint `https://api.openai.com/v1/embeddings`, dimension 1536, 30 second HTTP timeout.

`WithOpenAIModel` also sets the dimension for known models: `text-embedding-3-small` and `text-embedding-ada-002` are 1536, `text-embedding-3-large` is 3072. `WithOpenAIDimension` overrides that, which is what you want for a shortened `text-embedding-3-*` vector. The same dimension must be used for every ingest and every query against a given index.

The client sends `{"model", "input"}` and reads `data[].embedding` back in `index` order. A non-200 status is a plain error string. It is not marked temporary, so the rate limiter below will not retry it unless you wrap the client.

## Ollama

```go
emb := embedder.NewOllamaEmbedder("nomic-embed-text",
    embedder.WithOllamaEndpoint("http://localhost:11434/api/embed"),
    embedder.WithOllamaDimension(768),
    embedder.WithOllamaHTTPClient(client),
)
```

The model name is required. The default endpoint is `http://localhost:11434/api/embed`. The default dimension is 768, which matches `nomic-embed-text`. Set `WithOllamaDimension` for any other model. The HTTP timeout is 60 seconds. The request body is `{"model", "input"}` and the response field is `embeddings`.

## Rate limiter

`embedder.NewRateLimiter` wraps another embedder. Before each batch it waits on a `golang.org/x/time/rate.Limiter`. On failure it retries with exponential backoff and full jitter, up to the attempt count you pass.

```go
limited, err := embedder.NewRateLimiter(
    emb,
    rate.NewLimiter(rate.Limit(5), 1),
    4,                    // attempts
    200*time.Millisecond, // base backoff
    2*time.Second,        // max backoff
)
```

Retries happen only when the error implements `Temporary() bool` and returns true. Context cancellation, deadline exceeded, `ErrCountMismatch`, `ErrDimensionMismatch`, and `ErrEmptyEmbeddings` are not retried. The OpenAI and Ollama clients return ordinary `fmt.Errorf` values for HTTP failures, so those are not retried either. Wrap a client that turns a 429 into a `Temporary` error if you want that behavior. `NewRateLimiter` rejects a nil embedder, a nil limiter, fewer than 1 attempt, and a backoff whose max is less than its base.

`Dimension` is forwarded to the inner embedder.

## Next

[Stores](stores.md) persist the vectors those batches return.
