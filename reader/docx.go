package reader

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"strconv"
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

	s := newSections(ragout.CloneMetadata(metadata))
	if err := extractDocxText(xmlBytes, s); err != nil {
		return nil, fmt.Errorf("reader: docx: %w", err)
	}
	return s.done()
}

// extractDocxText walks word/document.xml's token stream, keeping text runs (w:t),
// turning line breaks (w:br) into newlines, and sending each paragraph (w:p) to s as
// a heading or body text.
// Headings are found by HeadingN style IDs or an inline outline level; a
// localized heading style defined only in styles.xml reads as body text.
func extractDocxText(xmlBytes []byte, s *sections) error {
	dec := xml.NewDecoder(bytes.NewReader(xmlBytes))
	var b strings.Builder
	inText := false
	level := 0

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
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
			case "pStyle":
				level = max(level, headingLevel(attr(t, "val")))
			case "outlineLvl":
				if n, err := strconv.Atoi(attr(t, "val")); err == nil && n < 9 {
					level = max(level, n+1)
				}
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				inText = false
			case "p":
				if text := strings.TrimSpace(b.String()); level > 0 && text != "" {
					s.heading(level, text, text)
				} else {
					s.text(b.String() + "\n")
				}
				b.Reset()
				level = 0
			}
		case xml.CharData:
			if inText {
				b.Write(t)
			}
		}
	}

	s.text(b.String())
	return nil
}

// headingLevel maps style IDs like "Heading2" or "heading 2" to 2, and anything else to 0.
func headingLevel(style string) int {
	style = strings.ToLower(strings.ReplaceAll(style, " ", ""))
	n, err := strconv.Atoi(strings.TrimPrefix(style, "heading"))
	if !strings.HasPrefix(style, "heading") || err != nil || n < 1 || n > 9 {
		return 0
	}
	return n
}

func attr(e xml.StartElement, local string) string {
	for _, a := range e.Attr {
		if a.Name.Local == local {
			return a.Value
		}
	}
	return ""
}
