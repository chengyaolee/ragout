# Changelog

All notable changes to this module. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Before v1, a minor version may contain breaking changes; they are listed under **Breaking** with what to do.

## [Unreleased]

### Breaking

- **Chunks are indexed with a title and section by default.** The engine now uses `ragout.MetadataContext()`. It writes each chunk's breadcrumb (for example `Annual Report > Financials`) into the new `Chunk.Context`, which the embedder and BM25 see and the generator does not. Re-ingest existing indexes so every chunk was indexed the same way. Pass `ragout.WithContextualizer(nil)` to keep the old behavior.
- **The token chunker numbers chunks `0, 1, 2, ...`** instead of by token offset, matching the character chunker, so expansion can find neighbors. Chunk IDs for token-chunked documents change: re-ingest them with `WithReplaceSource`. It also returns `ragout.ErrEmptyDocument` for empty input, like the character chunker.
- **Markdown, HTML, and DOCX readers return one document per heading section**, with the heading path in `metadata["section"]`. The HTML reader also sets `metadata["title"]` from `<title>` and drops `<head>` text from the content. Chunk IDs for these files change. Re-ingest them with `WithReplaceSource` so the old whole-file chunks are removed.

### Added

- **Chunk context:** `Chunk.Context`, `Chunk.IndexText()`, `ragout.Contextualizer`, `ragout.MetadataContext()`, `ragout.Breadcrumb()`, and `contextual.New` for LLM-written context (Anthropic's contextual retrieval). The LLM version has `WithConcurrency` and `WithWindow`, which bounds how much of a document each call sees.
- **Model calls:** `ragout.Completer`, a one-shot model call, implemented by the OpenAI and Ollama generators.
- **Query rewriting:** `ragout.QueryTransformer`, `WithQueryTransformer`, `WithTransformTimeout`, and the per-query `WithHistory`. The `transform` package has `Condense`, `MultiQuery`, `Decompose`, `HyDE`, and `Chain`. Searches can target the vector store, BM25, or both. Failures and timeouts fall back to the original question and are reported in `Answer.TransformErr` and the `ragout_query_transform_failures_total` counter.
- **Pluggable fusion:** `ragout.FusionStrategy`, `ragout.RRF(k)`, and `WithFusion`. Fusion works over any number of result lists, and all dense searches are embedded in one call.
- **Expansion:** `ragout.Expander`, `WithExpander`, and the `expand` package (`Neighbors`, `Section`), backed by `ragout.ChunkGetter`, which all four stores implement.
- **History in the answer prompt (opt-in):** `ragout.HistoryGenerator`, `generator.WithOpenAIHistoryTurns`, `generator.WithOllamaHistoryTurns`, and `ContextBudgeter.BuildPromptWithHistory`.
- **Source management:** `Engine.DeleteSource`, `WithReplaceSource`, and `ragout.SourceDeleter`, which all four stores implement.
- **Evaluation:** `Engine.Retrieve`, which runs everything before generation and needs no generator, and the `eval` package, which reports hit rate, recall, and MRR.
- **Metadata keys:** `ragout.MetadataTitle` and `ragout.MetadataSection`.
- **Answer fields:** `Answer.Question`, `Answer.Searches`, and `Answer.TransformErr`.
- **Ollama:** `generator.WithOllamaNumCtx` sets Ollama's context window, since Ollama silently truncates longer prompts.
- **Integration tests:** the `integration` package runs against real Chroma, Qdrant, Ollama, and OpenAI when `RAGOUT_IT_*` variables are set.

### Fixed

- **BM25 snapshots:** a reloaded `BM25Index` snapshot keeps each chunk's `Context`, so the index matches the same terms after a restart.
- **Cohere:** the reranker sends `top_n` under the field name Cohere expects, so Cohere returns only the requested results.
- **Qdrant:** `Delete` skips IDs that are not UUIDs instead of failing the whole request, since Qdrant cannot hold them. `GetChunks` and `DeleteSource` do the same.

## First public release

First public release: hybrid BM25 and vector retrieval with reciprocal rank fusion, Cohere and MMR rerankers, cited streaming generation with a token budget, readers for text, Markdown, HTML, PDF, and DOCX, in-memory stores with snapshots, Chroma and Qdrant over HTTP, OpenAI and Ollama clients, and OpenTelemetry and Prometheus instrumentation.
