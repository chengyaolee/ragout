# Update and delete sources

[Docs](../README.md) · [Ingest files](ingest-files.md) · [Stores](../components/stores.md)

Chunk IDs come from the document's source name, its text, and the chunk's position. Ingesting the same file twice overwrites the same chunks. Ingesting an edited file writes new IDs for the parts that changed, and by default the old chunks stay in the stores and keep showing up in results.

## Replace a source on every ingest

```go
engine, err := ragout.NewEngine(
    // reader, chunker, embedder, stores...
    ragout.WithReplaceSource(),
)
err = engine.Ingest(ctx, "handbook.md", f, nil)
```

With `WithReplaceSource`, `Ingest` writes the new chunks first. It then deletes the chunks of that source that the new ingest did not rewrite. A failed write never removes the previous version.

It is off by default, because some callers ingest several different streams under one source name. If the new chunks are written but the cleanup fails, `Ingest` returns an error that says so. The new content is already searchable.

## Delete a source

```go
err := engine.DeleteSource(ctx, "handbook.md")
```

This removes every chunk whose `metadata["source"]` is that name, from every configured store. An empty name returns `ErrNoSource`.

## Store support

Both features need every configured store to implement `ragout.SourceDeleter`:

```go
type SourceDeleter interface {
    DeleteSource(ctx context.Context, source string, keep []string) error
}
```

The in-memory vector store, BM25, Chroma, and Qdrant all do. A store that does not makes `DeleteSource` return `ErrDeleteUnsupported`, and `NewEngine` rejects `WithReplaceSource`. All stores are checked before any is touched.

## Upgrading an existing index

Since this release, Markdown, HTML, and DOCX files are split by heading, and chunks are indexed with a title and section breadcrumb. Re-ingest each source with `WithReplaceSource` set, so the chunks from the older ingest are removed.

## Next

[Measure retrieval](evaluate.md) to check the index after a change.
