package reranker

import (
	"context"
	"testing"

	"github.com/chengyaolee/ragout"
)

func TestMMR_Deduplication(t *testing.T) {
	query := []float32{1, 0}
	candidates := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "cats", Embedding: []float32{0.8, 0.6}}},
		{Chunk: ragout.Chunk{ID: "cats-dup", Embedding: []float32{0.8, 0.6}}},
		{Chunk: ragout.Chunk{ID: "dogs", Embedding: []float32{0.8, -0.6}}},
	}

	got, err := MaximalMarginalRelevance(query, candidates, 2, &MMROptions{Lambda: 0.7})
	if err != nil {
		t.Fatalf("mmr: %v", err)
	}
	if len(got) != 2 || got[0].Chunk.ID != "cats" || got[1].Chunk.ID != "dogs" {
		t.Fatalf("got %s then %s, want cats then dogs", idAt(got, 0), idAt(got, 1))
	}
}

func TestMockReranker(t *testing.T) {
	var _ ragout.Reranker = (*MockReranker)(nil)

	candidates := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "dogs", Content: "dogs run"}},
		{Chunk: ragout.Chunk{ID: "both", Content: "the cats sleep here"}},
		{Chunk: ragout.Chunk{ID: "cats", Content: "cats"}},
	}

	got, err := NewMockReranker().Rerank(context.Background(), "cats sleep", candidates, 2)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(got) != 2 || got[0].Chunk.ID != "both" || got[0].Score != 2 || got[1].Chunk.ID != "cats" || got[1].Score != 1 {
		t.Fatalf("got %#v, want both (2) then cats (1)", got)
	}
}

func TestMMRReranker_Interface(t *testing.T) {
	var _ ragout.Reranker = (*MMRReranker)(nil)

	candidates := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "cats", Embedding: []float32{0.8, 0.6}}, Score: 0.9},
		{Chunk: ragout.Chunk{ID: "cats-dup", Embedding: []float32{0.8, 0.6}}, Score: 0.9},
		{Chunk: ragout.Chunk{ID: "dogs", Embedding: []float32{0.8, -0.6}}, Score: 0.8},
	}

	reranker := NewMMRReranker(nil, 0.7)
	got, err := reranker.Rerank(context.Background(), "cats", candidates, 2)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2, got %d", len(got))
	}
}

func idAt(chunks []ragout.ScoredChunk, i int) string {
	if i >= len(chunks) {
		return ""
	}
	return chunks[i].Chunk.ID
}
