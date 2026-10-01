# Ingest files

[Docs](../README.md) · [Readers](../components/readers.md) · [Metadata filters](metadata-filters.md)

`reader.NewByExtension` picks a reader from the file extension on `metadata["source"]`. `Ingest` sets that key from the source name you pass, so the name has to include the extension even if you opened the file from another path.

```go
readers := reader.NewByExtension(10 << 20) // 10 MiB cap, shared by every format

engine, err := ragout.NewEngine(
    ragout.WithReader(readers),
    ragout.WithChunker(chunks),
    // embedder, stores, generator...
)

f, err := os.Open(path)
if err != nil {
    return err
}
defer f.Close()

err = engine.Ingest(ctx, path, f, map[string]any{
    "department": "engineering",
})
```

| Extension | Reader |
| --- | --- |
| `.txt` | Plain text, trimmed. |
| `.md`, `.markdown` | Text, plus a leading `---` frontmatter block copied into metadata as strings. |
| `.html`, `.htm` | Tags, `script`, and `style` removed. Entities unescaped. |
| `.pdf` | Extracted text, one document. |
| `.docx` | Body text from `word/document.xml`. Legacy `.doc` is not supported. |

Anything else returns `ErrUnsupportedFormat`. A file larger than the cap returns `ErrDocumentTooLarge`. The reader does not truncate. Passing `maxBytes <= 0` to a single reader constructor (`NewTextReader` and the others) uses a 10 MiB default. `NewByExtension` forwards the number you give it.

Frontmatter parsing is line-oriented `key: value`. It is not a full YAML parser. Quotes around a value are stripped. Lines starting with `#` are skipped.

## One format

If every input is PDF, skip the dispatcher:

```go
ragout.WithReader(reader.NewPDFReader(10 << 20))
```

The source name is still stored as `source`. The extension is only required when the reader is a `ByExtension` map.

## Add a format

`ByExtension` is a map. Assign a `ragout.Reader` for your extension:

```go
readers := reader.NewByExtension(10 << 20)
readers[".csv"] = myCSVReader
```

`Read` must honor `ctx`, respect a size limit, and return `ErrEmptyDocument` when there is no text. Copy metadata with `ragout.CloneMetadata` so later stages cannot mutate the caller's map.

## Several files

Call `Ingest` once per file. Chunk IDs depend on the source name and the content, so ingesting `docs/a.md` twice replaces the previous chunks. Ingesting the same bytes under a different source name creates a second document.

## Next

Attach the fields you want to filter on, then see [Filter by metadata](metadata-filters.md).
