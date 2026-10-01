package reader_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/reader"
)

// buildMinimalPDF writes a tiny single-page, single-text-run PDF by hand (no
// compression), byte-accurate xref table included, so pdf.NewReader can parse it
// without a PDF authoring tool.
func buildMinimalPDF(text string) []byte {
	var buf bytes.Buffer
	offsets := make([]int, 6)
	buf.WriteString("%PDF-1.4\n")

	write := func(i int, s string) {
		offsets[i] = buf.Len()
		buf.WriteString(s)
	}

	write(1, "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
	write(2, "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
	write(3, "3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 200 200] /Contents 5 0 R >>\nendobj\n")
	write(4, "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

	content := fmt.Sprintf("BT /F1 24 Tf 10 100 Td (%s) Tj ET", text)
	write(5, fmt.Sprintf("5 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n", len(content), content))

	xrefOffset := buf.Len()
	buf.WriteString("xref\n0 6\n0000000000 65535 f \n")
	for i := 1; i <= 5; i++ {
		fmt.Fprintf(&buf, "%010d 00000 n \n", offsets[i])
	}
	buf.WriteString("trailer\n<< /Size 6 /Root 1 0 R >>\n")
	fmt.Fprintf(&buf, "startxref\n%d\n%%%%EOF", xrefOffset)

	return buf.Bytes()
}

func TestPDFReader_ExtractsText(t *testing.T) {
	pr := reader.NewPDFReader(0)
	pdfBytes := buildMinimalPDF("Hello World")

	docs, err := pr.Read(context.Background(), bytes.NewReader(pdfBytes), map[string]any{"source": "test.pdf"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if !strings.Contains(docs[0].Content, "Hello World") {
		t.Errorf("content = %q, want it to contain %q", docs[0].Content, "Hello World")
	}
}

func TestPDFReader_MaxBytesLimit(t *testing.T) {
	pr := reader.NewPDFReader(5)
	pdfBytes := buildMinimalPDF("Hello World")

	_, err := pr.Read(context.Background(), bytes.NewReader(pdfBytes), nil)
	if !errors.Is(err, ragout.ErrDocumentTooLarge) {
		t.Fatalf("expected ErrDocumentTooLarge, got %v", err)
	}
}

func TestPDFReader_Malformed(t *testing.T) {
	pr := reader.NewPDFReader(0)
	_, err := pr.Read(context.Background(), strings.NewReader("not a pdf"), nil)
	if err == nil {
		t.Fatal("expected an error for malformed input")
	}
}
