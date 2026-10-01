# Readers

[Docs](../README.md) · [Ingest files](../how-to/ingest-files.md) · [Chunkers](chunkers.md)

A reader implements `ragout.Reader`:

```go
Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error)
```

It must not read past its byte cap. `read` uses `io.LimitReader` and returns `ragout.ErrDocumentTooLarge` when the stream is longer than the cap, instead of silently cutting the document. Cancelling `ctx` aborts the read. An empty result is `ragout.ErrEmptyDocument`.

Every constructor treats `maxBytes <= 0` as 10 MiB.

| Constructor | Output |
| --- | --- |
| `NewTextReader(maxBytes)` | Trimmed text. One document. |
| `NewMarkdownReader(maxBytes)` | Text, with a leading `---` block parsed into metadata. |
| `NewHTMLReader(maxBytes)` | Visible text. `script` and `style` removed, tags stripped, entities unescaped. |
| `NewPDFReader(maxBytes)` | Extracted page text concatenated into one document. |
| `NewDocxReader(maxBytes)` | Paragraph text from `word/document.xml`. |
| `NewByExtension(maxBytes)` | Dispatches on `filepath.Ext` of `metadata["source"]`. |

`Ingest` sets `source` before calling `Read`, so the dispatcher sees the name you passed to `Ingest`.

## By extension

```go
readers := reader.NewByExtension(10 << 20)
// .txt .md .markdown .html .htm .pdf .docx
```

The map is the extension table. Add or replace an entry:

```go
readers[".csv"] = csvReader
```

A missing extension returns `ErrUnsupportedFormat` wrapped with the extension string.

## Markdown frontmatter

A file that starts with `---` is split on the next `---`. Lines of the form `key: value` become metadata strings. Surrounding quotes are removed. Blank lines and `#` comments are skipped. The body after the closing marker is the document content. Nested YAML, lists, and multiline values are not parsed.

## HTML and office formats

The HTML reader is a tag stripper, not a layout engine. The PDF reader uses a pure-Go PDF parser and recovers from panics on malformed files by returning an error. The DOCX reader uses `archive/zip` and `encoding/xml` from the standard library. Binary `.doc` files are rejected by the zip reader.

All of them copy metadata with `ragout.CloneMetadata` onto the document they return.

## Next

[Chunkers](chunkers.md) split those documents.
