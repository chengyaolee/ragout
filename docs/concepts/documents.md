# Documents and chunks

[Docs](../README.md) · [Pipeline](pipeline.md) · [Hybrid retrieval](hybrid-retrieval.md)

Ingest turns a byte stream into documents, then into chunks. Search and generation only see chunks.

## Document

```go
type Document struct {
    ID       string
    Content  string
    Metadata map[string]any
}
```

A reader returns one or more documents. The text and PDF readers return one document per file. The Markdown, HTML, and DOCX readers return one document per heading section, with the heading path in `metadata["section"]`. See [Readers](../components/readers.md).

`Ingest` sets `metadata["source"]` to the source name before the reader runs. That key is `ragout.MetadataSource`. If you pass your own `source` key, the source argument replaces it.

If `Document.ID` is empty, ingest sets it to `DocumentID(source, content)`, a UUID derived from the source name and the text (and the section path, when there is one). The same file and the same bytes always get the same ID.

## Chunk

```go
type Chunk struct {
    ID         string
    DocumentID string
    Index      int
    Content    string
    Context    string
    Embedding  []float32
    Metadata   map[string]any
}
```

`Content` is the chunk's text. It is what the generator reads and what citations quote. `Context` is extra text written at ingest by a contextualizer, by default the document's title and section. The embedder and the BM25 index see `Chunk.IndexText()`, which is `Context` followed by `Content`. See [Give chunks context](../how-to/chunk-context.md).

The chunker copies the document's metadata onto every chunk. `CloneMetadata` copies the map. Nested maps and slices are shared with the original, so treat metadata values as immutable after ingest.

`ChunkID(docID, index)` is a UUID of the document ID and the index. Both built-in chunkers number chunks `0, 1, 2, ...` in document order, and a custom chunker should too: expanders find a chunk's neighbors at `index ± 1`. Re-chunking the same document with the same chunker settings writes the same IDs.

The vector store and the BM25 index both keep the chunk, including its metadata. Embeddings live on the chunk in the vector store. BM25 does not need them.

## ScoredChunk

Search returns `ScoredChunk`:

| Field | Meaning |
| --- | --- |
| `Chunk` | The stored chunk. |
| `DenseScore` | Cosine similarity when this hit came from the vector store. |
| `SparseScore` | BM25 score when this hit came from the index. |
| `Score` | The score the current stage is ranking on. After fusion this is the reciprocal-rank sum, not the cosine or BM25 value. |

A chunk that appears in only one kind of list keeps the other score field at zero. When a question is searched several ways (see [Rewrite questions](../how-to/query-rewrite.md)), a fused chunk keeps the best `DenseScore` and `SparseScore` it got.

## Filters see this metadata

`WithQueryFilter` matches against `Chunk.Metadata`. Equality, a slice of allowed values, and the operators `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, and `$lte` are documented in [Filter by metadata](../how-to/metadata-filters.md).

## Next

[Hybrid retrieval](hybrid-retrieval.md) is what happens to these chunks at query time.
