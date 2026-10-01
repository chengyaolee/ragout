# Traces and metrics

[Docs](../README.md) · [Pipeline](../concepts/pipeline.md) · [Streaming](../concepts/streaming.md)

The engine uses the global OpenTelemetry tracer and the default Prometheus registerer. You do not pass a tracer into `NewEngine`.

## Traces

```go
import (
    "go.opentelemetry.io/otel"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

tp := sdktrace.NewTracerProvider(/* exporter, sampler */)
defer func() { _ = tp.Shutdown(ctx) }()
otel.SetTracerProvider(tp)
```

The tracer name is `github.com/chengyaolee/ragout`.

| Span | When | Attributes |
| --- | --- | --- |
| `ragout.Engine.Query` | Every query | `ragout.query`, `ragout.top_k_recall`, `ragout.top_n_rerank` |
| `query.retrieval` | Around the fan-out | |
| `query.embed` | Question embedding | `text.len`, `model` (the embedder's Go type) |
| `query.dense_search` | Vector search | `top_k`, `candidates_found` |
| `query.sparse_search` | BM25 search | `terms_matched`, `candidates_found` |
| `query.rrf_fusion` | Both sides returned hits | `rrf_k`, `fused_count` |
| `query.reranker` | A reranker is configured | `reranker_type`, `top_n` |
| `query.generation` | Token stream | `prompt_tokens`, `first_token_latency_ms` |

`terms_matched` is set when the index store implements `TermsMatched(string) int`. `BM25Index` does. A failed span records the error and sets status `Error`. `ragout.query` is the raw question text. Treat trace exports as sensitive if questions are.

`prompt_tokens` estimates size as `len(text)/4` over the question and the content of each source chunk. `first_token_latency_ms` is the time from the start of generation until the first token.

Ingest does not open these spans. It does increment the ingest counter below.

## Metrics

`init` registers two collectors on `prometheus.DefaultRegisterer`:

| Name | Type | Labels |
| --- | --- | --- |
| `ragout_query_duration_seconds` | Histogram | `stage`: `embed`, `dense`, `sparse`, `fusion`, `rerank`, `generation` |
| `ragout_ingested_chunks_total` | Counter | Chunks written by a successful `Ingest` |

Buckets are 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s.

Expose them with the Prometheus HTTP handler on the default gatherer. `fusion` and `rerank` observations appear only when those stages run.

If your process uses a custom registerer, call `ragout.RegisterMetrics(reg)`. An already-registered collector is ignored. Any other registration error is returned.

## Next

[Engine](../components/engine.md) lists the options that change `top_k` and `top_n` on the root span.
