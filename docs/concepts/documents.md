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

A reader returns one or more documents. The built-in readers each return a single document: the text of the file, with scripts and tags removed for HTML, and with a simple frontmatter block lifted into metadata for Markdown.

`Ingest` sets `metadata["source"]` to the source name before the reader runs. That key is `ragout.MetadataSource`. If you pass your own `source` key, the source argument replaces it.

If `Document.ID` is empty, ingest sets it to `DocumentID(source, content)`, a UUID derived from the source name and the text. The same file and the same bytes always get the same ID.

## Chunk

```go
type Chunk struct {
    ID         string
    DocumentID string
    Index      int
    Content    string
    Embedding  []float32
    Metadata   map[string]any
}
```

The chunker copies the document's metadata onto every chunk. `CloneMetadata` copies the map. Nested maps and slices are shared with the original, so treat metadata values as immutable after ingest.

`ChunkID(docID, index)` is a UUID of the document ID and the index. The character chunker uses `0, 1, 2, ...`. The token chunker uses the starting token offset as `Index`, so those indexes are not a dense `0..n` sequence. Either way, re-chunking the same document with the same chunker writes the same IDs.

The vector store and the BM25 index both keep the chunk, including its metadata. Embeddings live on the chunk in the vector store. BM25 does not need them.

## ScoredChunk

Search returns `ScoredChunk`:

| Field | Meaning |
| --- | --- |
| `Chunk` | The stored chunk. |
| `DenseScore` | Cosine similarity when this hit came from the vector store. |
| `SparseScore` | BM25 score when this hit came from the index. |
| `Score` | The score the current stage is ranking on. After fusion this is the reciprocal-rank sum, not the cosine or BM25 value. |

A chunk that appears in only one list keeps that side's score field at zero.

## Filters see this metadata

`WithQueryFilter` matches against `Chunk.Metadata`. Equality, a slice of allowed values, and the operators `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, and `$lte` are documented in [Filter by metadata](../how-to/metadata-filters.md).

## Next

[Hybrid retrieval](hybrid-retrieval.md) is what happens to these chunks at query time.
