package reader

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/chengyaolee/ragout"
	"github.com/ledongthuc/pdf"
)

// PDFReader extracts plain text from PDF streams.
type PDFReader struct {
	maxBytes int64
}

// NewPDFReader creates an initialized PDFReader with a maximum byte limit.
func NewPDFReader(maxBytes int64) *PDFReader {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024 // Default 10MB
	}
	return &PDFReader{maxBytes: maxBytes}
}

// Read parses a PDF stream and emits its extracted text as a single document.
func (pr *PDFReader) Read(ctx context.Context, r io.Reader, metadata map[string]any) (docs []ragout.Document, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b, err := readLimited(r, pr.maxBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// ledongthuc/pdf panics on some malformed PDFs rather than returning an error.
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("reader: pdf parse panic: %v", rec)
		}
	}()

	reader, err := pdf.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		return nil, fmt.Errorf("reader: pdf: %w", err)
	}

	var buf strings.Builder
	content, err := reader.GetPlainText()
	if err != nil {
		return nil, fmt.Errorf("reader: pdf: %w", err)
	}
	if _, err := io.Copy(&buf, content); err != nil {
		return nil, fmt.Errorf("reader: pdf: %w", err)
	}

	text := strings.TrimSpace(buf.String())
	if text == "" {
		// Scanned PDFs with no text layer need OCR, which is out of scope here.
		return nil, ragout.ErrEmptyDocument
	}

	return []ragout.Document{{
		Content:  text,
		Metadata: ragout.CloneMetadata(metadata),
	}}, nil
}
