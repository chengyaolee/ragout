package reader

import (
	"context"
	"io"
	"strings"

	"github.com/chengyaolee/ragout"
)

type TextReader struct {
	maxBytes int64
}

func NewTextReader(maxBytes int64) *TextReader {
	if maxBytes <= 0 {
		maxBytes = 10 * 1024 * 1024 // Default 10MB
	}
	return &TextReader{maxBytes: maxBytes}
}

func (tr *TextReader) Read(ctx context.Context, r io.Reader, metadata map[string]any) ([]ragout.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	b, err := readLimited(r, tr.maxBytes)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	text := strings.TrimSpace(string(b))
	if text == "" {
		return nil, ragout.ErrEmptyDocument
	}

	return []ragout.Document{{
		Content:  text,
		Metadata: ragout.CloneMetadata(metadata),
	}}, nil
}
