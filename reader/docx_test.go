package reader_test

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/reader"
)

const docxBody = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>World</w:t></w:r></w:p>
<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
</w:body>
</w:document>`

// buildMinimalDocx zips a single word/document.xml entry; ragout's DocxReader
// doesn't need the other OOXML parts (content types, rels, …) to extract text.
func buildMinimalDocx(t *testing.T, xmlBody string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create("word/document.xml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte(xmlBody)); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestDocxReader_ExtractsText(t *testing.T) {
	dr := reader.NewDocxReader(0)
	docxBytes := buildMinimalDocx(t, docxBody)

	docs, err := dr.Read(context.Background(), bytes.NewReader(docxBytes), map[string]any{"source": "test.docx"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}

	content := docs[0].Content
	if !strings.Contains(content, "Hello\tWorld") {
		t.Errorf("content = %q, want a tab between %q and %q", content, "Hello", "World")
	}
	if !strings.Contains(content, "Second paragraph") {
		t.Errorf("content = %q, want it to contain the second paragraph", content)
	}
	if !strings.Contains(content, "World\n") && !strings.HasSuffix(strings.TrimSpace(content), "Second paragraph") {
		t.Errorf("content = %q, want paragraphs separated by a newline", content)
	}
}

func TestDocxReader_MaxBytesLimit(t *testing.T) {
	dr := reader.NewDocxReader(5)
	docxBytes := buildMinimalDocx(t, docxBody)

	_, err := dr.Read(context.Background(), bytes.NewReader(docxBytes), nil)
	if !errors.Is(err, ragout.ErrDocumentTooLarge) {
		t.Fatalf("expected ErrDocumentTooLarge, got %v", err)
	}
}

func TestDocxReader_MissingDocumentXML(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}

	dr := reader.NewDocxReader(0)
	_, err := dr.Read(context.Background(), bytes.NewReader(buf.Bytes()), nil)
	if err == nil {
		t.Fatal("expected an error for a docx with no word/document.xml")
	}
}
