# Widen hits before generation

[Docs](../README.md) · [Pipeline](../concepts/pipeline.md) · [Stores](../components/stores.md)

Small chunks retrieve precisely but can be too short to answer from. An expander runs after reranking and before generation. It replaces each hit's text with a wider span, so search stays precise and the generator reads more.

With no expander set, this stage does not run.

## Neighbors

```go
vectors := store.NewVectorStore()
ragout.WithExpander(expand.Neighbors(vectors, 1))
```

`expand.Neighbors(store, n)` adds up to `n` chunks on each side of every hit, from the same document. The surrounding chunks are fetched with one `GetChunks` call per query.

## Whole sections (parent-child)

```go
ragout.WithExpander(expand.Section(vectors, 20))
```

The Markdown, HTML, and DOCX readers emit one document per heading, and a window never crosses documents. `expand.Section(store, max)` uses a window wide enough to return the hit's whole section, up to `max` chunks on each side.

## What stays the same

- Each hit keeps its own ID, score, and citation number. Only `Content` changes.
- A hit that already sits inside a higher-ranked hit's window is dropped, not repeated.
- `Answer.Retrieved` holds the expanded chunks.

Wider hits cost prompt tokens on every query. The token budget still applies, so a hit that no longer fits is left out of the prompt.

## Requirements and limits

- **Store:** pass a store that implements `ragout.ChunkGetter`. The in-memory vector store, BM25, Chroma, and Qdrant all do.
- **Chunker:** neighbors are found by chunk index, so they need consecutive indexes. The character chunker numbers chunks `0, 1, 2, ...`. The token chunker uses token offsets, so expansion finds no neighbors with it.
- **Overlap:** if the chunker overlaps chunks, the overlapping text appears twice where neighbors join.

## Write your own

```go
type Expander interface {
    Expand(ctx context.Context, candidates []ScoredChunk) ([]ScoredChunk, error)
}
```

Keep the candidates in order.

## Next

[Measure retrieval](evaluate.md) compares an engine with and without an expander.
