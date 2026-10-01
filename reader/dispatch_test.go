package reader_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/reader"
)

func TestByExtension_Dispatch(t *testing.T) {
	byExt := reader.NewByExtension(0)

	docs, err := byExt.Read(context.Background(), strings.NewReader("hello world"), map[string]any{"source": "notes.txt"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 || docs[0].Content != "hello world" {
		t.Fatalf("docs = %+v, want one doc with content %q", docs, "hello world")
	}

	docs, err = byExt.Read(context.Background(), strings.NewReader("# Title\nbody"), map[string]any{"source": "notes.MD"})
	if err != nil {
		t.Fatalf("unexpected error for .MD: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc for .MD, got %d", len(docs))
	}
}

func TestByExtension_Unsupported(t *testing.T) {
	byExt := reader.NewByExtension(0)
	_, err := byExt.Read(context.Background(), strings.NewReader("..."), map[string]any{"source": "data.csv"})
	if !errors.Is(err, ragout.ErrUnsupportedFormat) {
		t.Fatalf("expected ErrUnsupportedFormat, got %v", err)
	}
}
