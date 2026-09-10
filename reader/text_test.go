package reader_test

import (
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/reader"
)

func TestTextReader_Success(t *testing.T) {
	tr := reader.NewTextReader(0)
	metadata := map[string]any{"source": "test"}

	docs, err := tr.Read(context.Background(), strings.NewReader("  hello world  "), metadata)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}
	if docs[0].ID == "" {
		t.Error("expected document ID")
	}
	if docs[0].Content != "hello world" {
		t.Errorf("content = %q, want %q", docs[0].Content, "hello world")
	}
	if docs[0].Metadata["source"] != "test" {
		t.Errorf("metadata source = %v, want test", docs[0].Metadata["source"])
	}
	docs[0].Metadata["source"] = "mutated"
	if metadata["source"] != "test" {
		t.Error("caller metadata was mutated")
	}
}

func TestTextReader_Empty(t *testing.T) {
	tr := reader.NewTextReader(0)
	docs, err := tr.Read(context.Background(), strings.NewReader("   "), nil)
	if err != ragout.ErrEmptyDocument {
		t.Errorf("expected ErrEmptyDocument, got %v", err)
	}
	if docs != nil {
		t.Errorf("expected nil documents, got %v", docs)
	}
}

func TestTextReader_MaxBytesLimit(t *testing.T) {
	tr := reader.NewTextReader(5)
	docs, err := tr.Read(context.Background(), strings.NewReader("hello world"), nil)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if docs[0].Content != "hello" {
		t.Errorf("content = %q, want truncated %q", docs[0].Content, "hello")
	}
}

func TestTextReader_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tr := reader.NewTextReader(0)
	_, err := tr.Read(ctx, strings.NewReader("hello"), nil)
	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}
