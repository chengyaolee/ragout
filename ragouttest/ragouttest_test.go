package ragouttest_test

import (
	"context"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/ragouttest"
)

var (
	_ ragout.Embedder  = (*ragouttest.Embedder)(nil)
	_ ragout.Reranker  = (*ragouttest.Reranker)(nil)
	_ ragout.Generator = (*ragouttest.Generator)(nil)
)

func TestEmbedder(t *testing.T) {
	m, err := ragouttest.NewEmbedder(4, 50*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if m.Dimension() != 4 {
		t.Fatalf("dim = %d, want 4", m.Dimension())
	}
	start := time.Now()
	first, err := m.EmbedBatch(context.Background(), []string{"cat", "cat", "dog"})
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(start) < 40*time.Millisecond {
		t.Fatal("EmbedBatch returned before the configured latency")
	}
	second, err := m.EmbedBatch(context.Background(), []string{"cat", "cat", "dog"})
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 3 || len(first[0]) != 4 {
		t.Fatalf("vectors = %d x %d, want 3 x 4", len(first), len(first[0]))
	}
	for d := range first[0] {
		if first[0][d] != first[1][d] || first[0][d] != second[0][d] {
			t.Fatalf("same text produced different vectors")
		}
		if first[0][d] == first[2][d] {
			t.Fatalf("different texts produced the same vector component %d", d)
		}
	}
}

func TestReranker(t *testing.T) {
	candidates := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "dogs", Content: "dogs run"}},
		{Chunk: ragout.Chunk{ID: "both", Content: "the cats sleep here"}},
		{Chunk: ragout.Chunk{ID: "cats", Content: "cats"}},
	}

	got, err := ragouttest.NewReranker().Rerank(context.Background(), "cats sleep", candidates, 2)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(got) != 2 || got[0].Chunk.ID != "both" || got[0].Score != 2 || got[1].Chunk.ID != "cats" || got[1].Score != 1 {
		t.Fatalf("got %#v, want both (2) then cats (1)", got)
	}
}
