package reader

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"strings"

	"github.com/chengyaolee/ragout"
)

// DocxReader extracts plain text from Word (.docx) documents using the stdlib
// zip and xml packages; no external dependency needed for OOXML.
type DocxReader struct {
	maxBytes int64
}

// NewDocxReader creates an initialized DocxReader with a maximum byte limit.
func NewDocxReader(maxBytes int64) *DocxReader {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024 // Default 10MB
	}
	return &DocxReader{maxBytes: maxBytes}
}

// Read unzips a .docx stream and emits its extracted body text as a single document.
// Legacy binary .doc files are not supported.
func (dr *DocxReader) Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b, err := readLimited(r, dr.maxBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	zr, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		return nil, fmt.Errorf("reader: docx: %w", err)
	}

	f, err := zr.Open("word/document.xml")
	if err != nil {
		return nil, fmt.Errorf("reader: docx: missing word/document.xml: %w", err)
	}
	defer f.Close()

	// Cap the decompressed XML too, as a guard against zip-bomb inputs.
	xmlBytes, err := readLimited(f, dr.maxBytes)
	if err != nil {
		return nil, err
	}

	text, err := extractDocxText(xmlBytes)
	if err != nil {
		return nil, fmt.Errorf("reader: docx: %w", err)
	}

	text = strings.TrimSpace(text)
	if text == "" {
		return nil, ragout.ErrEmptyDocument
	}

	return []ragout.Document{{
		Content:  text,
		Metadata: ragout.CloneMetadata(metadata),
	}}, nil
}

// extractDocxText walks word/document.xml's token stream, keeping text runs (w:t)
// and turning paragraph (w:p) and line-break (w:br) boundaries into newlines.
func extractDocxText(xmlBytes []byte) (string, error) {
	dec := xml.NewDecoder(bytes.NewReader(xmlBytes))
	var b strings.Builder
	inText := false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}

		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t":
				inText = true
			case "br":
				b.WriteByte('\n')
			case "tab":
				b.WriteByte('\t')
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				b.WriteByte('\n')
			}
		case xml.CharData:
			if inText {
				b.Write(t)
			}
		}
	}

	return b.String(), nil
}
