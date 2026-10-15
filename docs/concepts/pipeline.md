# Pipeline

[Docs](../README.md) · [Documents](documents.md) · [Hybrid retrieval](hybrid-retrieval.md) · [Generation](generation.md) · [Streaming](streaming.md)

An `Engine` is one ingest path and one query path. Stages you do not configure are skipped. A missing reranker does not fail the query. A missing vector store does not embed the question.

## Ingest

`Ingest(ctx, source, r, metadata)`:

1. Copies `metadata`, sets `source` to the source name, and asks the reader to turn the stream into documents. The source name is also how `reader.ByExtension` picks `.md`, `.pdf`, and the other formats. Markdown, HTML, and DOCX come back as one document per heading section.
2. Assigns a document ID from the source name, the section, and the text when the reader left `ID` empty. Re-ingesting the same file produces the same IDs, so the stores overwrite instead of duplicating.
3. Chunks each document.
4. Contextualizes the chunks: by default each gets its document's title and section as `Chunk.Context`, which is indexed but never shown to the generator. See [Give chunks context](../how-to/chunk-context.md).
5. If a vector store is configured, embeds the chunks on a worker pool. The default batch size is 32 and the default worker count is 8. Change them with `WithEmbedBatchSize` and `WithEmbedConcurrency`.
6. Writes the vector store and the BM25 index at the same time. If either write fails, both stores delete those chunk IDs, including when the failure is a cancelled context. A retry then starts clean.
7. With `WithReplaceSource`, deletes the source's chunks that this ingest did not rewrite. See [Update and delete sources](../how-to/update-sources.md).

`Ingest` returns `ErrNilReader`, `ErrNilChunker`, `ErrEmptyDocument`, or `ErrEmptyChunk` when that stage has nothing to do. A reader that hits its byte cap returns `ErrDocumentTooLarge`.

## Query

`Query`, `QueryStream`, `QueryIter`, and `Retrieve` share one retrieval path. `Retrieve` stops before step 7.

1. Rejects an empty question (`ErrEmptyQuery`) and, for the query methods, a missing generator (`ErrNilGenerator`).
2. Rewrites, when a query transformer is set: the question becomes a standalone question, plus extra searches. On error or timeout it falls back to the original question. See [Rewrite questions](../how-to/query-rewrite.md).
3. Fans out. Every search runs on the indexes it targets; the question always runs on all of them. All dense texts are embedded in one call. Every search takes the same metadata filter and the same `topKRecall` (default 50).
4. Fuses. When more than one list has hits, reciprocal rank fusion (default `k=60`) builds one ranking and keeps `topKRecall` chunks. When only one list has hits, it is used as-is. Fusion is how the engine avoids comparing cosine scores with BM25 scores. See [Hybrid retrieval](hybrid-retrieval.md).
5. Reranks against the standalone question when a reranker is set, down to `topN` (default 5). With no reranker, the engine keeps the first `topN` of the fused list. An empty list is `ErrNoResults`.
6. Expands, when an expander is set: each hit is widened with neighboring chunks. See [Widen hits](../how-to/expand-hits.md).
7. Hands that list to the generator with the standalone question, plus the history when the query passed it and the generator supports it. The generator drops chunks that do not fit the token budget, numbers the rest, and streams tokens.

Per-query options override the engine defaults for that call only: `WithQueryFilter`, `WithQueryTopKRecall`, `WithQueryTopNRerank`, `WithHistory`.

## Defaults

| Knob | Default | Option |
| --- | --- | --- |
| Recall depth | 50 | `WithTopKRecall` |
| Chunks after rerank, or the cut when there is no reranker | 5 | `WithTopNRerank` |
| RRF constant `k` | 60 | `WithRRFK` |
| Embed batch size | 32 | `WithEmbedBatchSize` |
| Embed workers | 8 | `WithEmbedConcurrency` |
| Chunk context | Title and section | `WithContextualizer` |
| Query rewrite | None | `WithQueryTransformer` |
| Fusion | `RRF(k)` | `WithFusion` |
| Expansion | None | `WithExpander` |

Values of 0 or less are ignored, and the default stays.

## What a trace looks like

A configured OpenTelemetry tracer records `ragout.Engine.Query` (or `ragout.Engine.Retrieve`), then `query.transform` when a transformer is set, `query.retrieval`, `query.embed`, one `query.dense_search` and `query.sparse_search` per search, `query.rrf_fusion` when more than one list hit, `query.reranker` when a reranker is set, `query.expand` when an expander is set, and `query.generation`. Prometheus histograms use the same stage names. See [Traces and metrics](../how-to/observability.md).

## Next

[Documents and chunks](documents.md) explains the types that move through this path.
