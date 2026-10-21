# Changelog

All notable changes to this module, by date, oldest first. Each entry names the feature, what it does, and the package it lives in. Versions follow [Semantic Versioning](https://semver.org/). Before v1, a minor version may contain breaking changes; they are marked **Breaking** with what to do.

## September 8, 2026

- **Project skeleton:** the core types and interfaces (`ragout`), package documentation, and the README.

## September 10, 2026

- **Ingestion chunkers:** a plain-text reader (`reader`), a character chunker that splits recursively on separators, and a token-window chunker (`chunker`).

## September 26, 2026

- **Concurrent embedding:** `EmbedConcurrent` embeds chunks with a bounded worker pool, and one failed batch cancels the rest. A wrong vector count or dimension is reported as `ErrCountMismatch` or `ErrDimensionMismatch`. A mock embedder supports tests (`embedder`).
- **Rate-limited embedder:** `embedder.NewRateLimiter` waits on a rate limiter before each batch and retries temporary failures with exponential backoff and full jitter.

## September 28, 2026

- **Hybrid retrieval stores:** `BM25Index` for sparse search and an in-memory `VectorStore` for dense search by cosine similarity, both with metadata filters (`store`). `Embedder` now takes whole batches through `EmbedBatch`.

## September 29, 2026

- **Reranking and fusion:** a Cohere reranker and maximal marginal relevance (`reranker`), and fusion by reciprocal rank fusion or convex score combination. Scores are `float64` for precision and vectors `float32` for memory.

## September 30, 2026

- **Engine:** `ragout.Engine` ties the pipeline together with `Ingest` and `Query`. Ingest embeds chunks concurrently and writes to both stores. OpenTelemetry spans and Prometheus stage histograms cover every stage, and registering the metrics twice is safe.
- **Cited generation:** the `generator` package with OpenAI and Ollama clients, a token budget that reserves room for the prompt, and lost-in-the-middle ordering of chunks. HTTP clients can be injected, and the header wait is bounded so streaming calls can be tested.
- **Hosted and local embedders:** OpenAI and Ollama embedders (`embedder`).
- **More readers:** HTML and Markdown (`reader`).
- **Per-query control:** each query sets its own metadata filter and recall limits. Filters support comparison and set operators.
- **Retrieval quality and speed:** dense search scores true cosine similarity when vectors are not unit length. Top-K selection uses a bounded heap, with pooled candidate buffers. A fused chunk keeps its embedding so reranking still has a vector, and maximal marginal relevance falls back to retrieval scores when embeddings are missing. The character chunker keeps trailing text once it runs out of separators.

## October 1, 2026

- **Remote vector stores:** Chroma and Qdrant over HTTP, with upsert and delete (`store`).
- **More readers:** DOCX and PDF, with a read limit so an oversized document cannot exhaust memory (`reader`).
- **Source tracking:** `Ingest` takes the source name, which picks the reader by extension and is recorded on every chunk.
- **First public release:** hybrid BM25 and vector retrieval with reciprocal rank fusion, Cohere and MMR rerankers, cited streaming generation with a token budget, readers for text, Markdown, HTML, PDF, and DOCX, in-memory stores with snapshots, Chroma and Qdrant over HTTP, OpenAI and Ollama clients, and OpenTelemetry and Prometheus instrumentation.

## October 15, 2026

- **Chunk context:** `Chunk.Context`, `Chunk.IndexText()`, `ragout.Contextualizer`, `ragout.MetadataContext()`, `ragout.Breadcrumb()`, and `contextual.New` for LLM-written context (Anthropic's contextual retrieval). The embedder and BM25 index see the context and the generator does not. The LLM version has `WithConcurrency` and `WithWindow`, which bounds how much of a document each call sees. Snapshots of the in-memory stores keep each chunk's context, so a reloaded index matches the same terms.
  - **Breaking:** chunks are indexed with a title and section by default. The engine uses `ragout.MetadataContext()` and writes each chunk's breadcrumb (for example `Annual Report > Financials`) into `Chunk.Context`. Re-ingest existing indexes so every chunk was indexed the same way. Pass `ragout.WithContextualizer(nil)` to keep the old behavior.
- **Heading sections:** `metadata["section"]` and `metadata["title"]`, with `ragout.MetadataTitle` and `ragout.MetadataSection` (`reader`).
  - **Breaking:** the Markdown, HTML, and DOCX readers return one document per heading section, with the heading path in `metadata["section"]`. The HTML reader also sets `metadata["title"]` from `<title>` and drops `<head>` text from the content. Chunk IDs for these files change. Re-ingest them with `WithReplaceSource` so the old whole-file chunks are removed.
- **Model calls:** `ragout.Completer`, a one-shot model call, implemented by the OpenAI and Ollama generators.
- **Query rewriting:** `ragout.QueryTransformer`, `WithQueryTransformer`, `WithTransformTimeout`, and the per-query `WithHistory`. The `transform` package has `Condense`, `MultiQuery`, `Decompose`, `HyDE`, and `Chain`. Searches can target the vector store, BM25, or both. Failures and timeouts fall back to the original question and are reported in `Answer.TransformErr` and the `ragout_query_transform_failures_total` counter. `Answer` also gains `Question` and `Searches`.
- **Pluggable fusion:** `ragout.FusionStrategy` over labeled `ragout.RankedList` values, `ragout.RRF(k)`, and `WithFusion`. Fusion works over any number of result lists, and all dense searches are embedded in one call.
- **Expansion:** `ragout.Expander`, `WithExpander`, and the `expand` package (`Neighbors`, `Section`), backed by `ragout.ChunkGetter`, which all four stores implement.
- **History in the answer prompt (opt-in):** `ragout.HistoryGenerator`, `generator.WithOpenAIHistoryTurns`, `generator.WithOllamaHistoryTurns`, and `ContextBudgeter.BuildPromptWithHistory`.
- **Source management:** `Engine.DeleteSource`, `WithReplaceSource`, and `ragout.SourceDeleter`, which all four stores implement.
- **Retrieval evaluation:** `Engine.Retrieve`, which runs everything before generation and needs no generator, and the `eval` package, which reports hit rate, recall, and MRR.
- **Ollama context window:** `generator.WithOllamaNumCtx` sets Ollama's context window, since Ollama silently truncates longer prompts.
- **Integration tests:** the `integration` package runs against real Chroma, Qdrant, Ollama, and OpenAI when `RAGOUT_IT_*` variables are set.
- **Fixed:** the Cohere reranker sends `top_n` under the field name Cohere expects, so Cohere returns only the requested results. Qdrant `Delete` skips IDs that are not UUIDs instead of failing the whole request, since Qdrant cannot hold them. `GetChunks` and `DeleteSource` do the same.

## October 18, 2026

- **Chunk numbering:**
  - **Breaking:** the token chunker numbers chunks `0, 1, 2, ...` instead of by token offset, matching the character chunker, so expansion can find neighbors. Chunk IDs for token-chunked documents change: re-ingest them with `WithReplaceSource`. It also returns `ragout.ErrEmptyDocument` for empty input, like the character chunker.
- **Fixed (Ollama):**
  - A non-200 status or an `error` in the stream, such as a model that is not pulled, is returned as an error instead of an empty answer.
  - The generator's response-header timeout is 5 minutes, up from 30 seconds. A model that took longer than 30 seconds to load was cancelled on every request and could never answer.
  - `Complete` streams under the hood on both generators, so the header timeout bounds the time to the first token rather than the whole reply.

## October 20, 2026

- **Ollama reasoning models:** `generator.WithOllamaThink` controls whether reasoning models think first. They do not by default, since a rewrite or a grounded answer rarely needs it and it delays the first token.

## October 21, 2026

- **Local cross-encoder reranking:** `reranker.NewHTTP` calls any rerank server (llama.cpp, vLLM, Infinity, TEI, or Jina- and Cohere-compatible APIs). bge-reranker-v2-m3 is the default model, `reranker.Qwen3Reranker4B` the larger option, and `reranker.Qwen3Reranker` applies Qwen3's template for servers that do not.
- **Retrieval workflows:** the `flow` package runs retrieval as a graph of named steps that can be replaced, inserted, removed, or rewired, with `flow.CRAG` as a corrective-RAG preset (retrieve, grade, re-search, generate or abstain). Its grader judges each passage's relevance rather than overall sufficiency, which a small model does far more reliably, and it keeps the passages in retrieval order.
- **Benchmarking:** `eval.Case.Category` and `ExpectedSources`, per-category reports, nDCG, `eval.RunWith`, and `eval/cmd/bench`, which benchmarks every retrieval method on this repository's docs and on BEIR SciFact. Results are in `eval/results/`.
- **Weighted fusion:** `ragout.WeightedRRF(k, w)`. The default fusion counts a query transformer's added searches at half weight, which kept rephrasings from outvoting the question on the docs benchmark. Without a transformer it is plain RRF.
- **Query resilience:** `WithMinDenseScore` drops dense hits below a similarity floor, so a question nothing matches returns `ErrNoResults` instead of the nearest unrelated chunks. `WithDenseFailOpen` runs a query on BM25 when embedding the question or the vector search fails, and reports the error in `Answer.DenseErr`.
- **Query cache (opt-in):** the `cache` package wraps an engine, or anything with a `Retrieve` method, from outside; the engine itself never caches. `cache.Wrap` takes a `cache.Store` and a `cache.Policy`. Matching is `Exact` (the default; ignores case and spacing), `Semantic`, or `ExactThenSemantic`. Entries do not expire unless `Policy.TTL` is set. `cache.Memory` is the in-process store; other backends implement `cache.Store`. Semantic matching takes any `ragout.Embedder` and a required `MinScore`, and keeps vectors in any `ragout.VectorStore` (the in-memory store by default). `Invalidate` drops older entries after an ingest. Failed or degraded results are never cached.
