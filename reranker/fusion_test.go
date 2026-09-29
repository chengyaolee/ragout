package reranker

import (
	"testing"

	"github.com/chengyaolee/ragout"
)

func TestConvexScoreCombination(t *testing.T) {
	dense := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "a"}, DenseScore: 1, Score: 1},
		{Chunk: ragout.Chunk{ID: "b"}, DenseScore: 0, Score: 0},
	}
	sparse := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "b"}, SparseScore: 10, Score: 10},
		{Chunk: ragout.Chunk{ID: "c"}, SparseScore: 0, Score: 0},
	}

	got := ConvexScoreCombination(dense, sparse, 10, 0.6)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if got[0].Chunk.ID != "b" || got[1].Chunk.ID != "a" || got[2].Chunk.ID != "c" {
		t.Fatalf("order = %s %s %s, want b a c", got[0].Chunk.ID, got[1].Chunk.ID, got[2].Chunk.ID)
	}
	if got[0].DenseScore != 0 || got[0].SparseScore != 10 {
		t.Fatalf("b raw scores = dense %v sparse %v", got[0].DenseScore, got[0].SparseScore)
	}
}

func TestRRF_ConsensusWins(t *testing.T) {
	only := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "only"}, DenseScore: 0.99}
	both := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "both"}, DenseScore: 0.4, SparseScore: 0.4}
	dense := []ragout.ScoredChunk{only, both}
	sparse := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "sparse-only"}, SparseScore: 0.99},
		both,
	}

	got := ReciprocalRankFusion(dense, sparse, 10, 60)
	if len(got) == 0 || got[0].Chunk.ID != "both" {
		t.Fatalf("top = %v, want both", got)
	}
}

func TestRRF_DisjointSets(t *testing.T) {
	dense := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "a"}, DenseScore: 0.9},
		{Chunk: ragout.Chunk{ID: "b"}, DenseScore: 0.2},
	}
	sparse := []ragout.ScoredChunk{
		{Chunk: ragout.Chunk{ID: "c"}, SparseScore: 3},
		{Chunk: ragout.Chunk{ID: "d"}, SparseScore: 1},
	}

	got := ReciprocalRankFusion(dense, sparse, 10, 60)
	if len(got) != 4 {
		t.Fatalf("len = %d, want 4", len(got))
	}
	seen := map[string]bool{}
	for _, hit := range got {
		seen[hit.Chunk.ID] = true
	}
	for _, id := range []string{"a", "b", "c", "d"} {
		if !seen[id] {
			t.Fatalf("missing %s in %#v", id, got)
		}
	}
}

func TestRRF_EmptyAndEdgeCases(t *testing.T) {
	if got := ReciprocalRankFusion(nil, nil, 5, 60); len(got) != 0 {
		t.Fatalf("empty inputs len = %d, want 0", len(got))
	}
	dense := []ragout.ScoredChunk{{Chunk: ragout.Chunk{ID: "a"}, DenseScore: 1}}
	if got := ReciprocalRankFusion(dense, nil, 0, 60); len(got) != 0 {
		t.Fatalf("topK 0 len = %d, want 0", len(got))
	}
	if got := ReciprocalRankFusion(dense, nil, -1, 60); len(got) != 0 {
		t.Fatalf("topK -1 len = %d, want 0", len(got))
	}

	want := ReciprocalRankFusion(dense, nil, 5, 60)
	for _, k := range []int{0, -3} {
		got := ReciprocalRankFusion(dense, nil, 5, k)
		if len(got) != 1 || got[0].Score != want[0].Score {
			t.Fatalf("k=%d score = %v, want %v", k, got, want)
		}
	}
}

func TestReciprocalRankFusion(t *testing.T) {
	only := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "only"}, DenseScore: 0.99}
	both := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "both"}, DenseScore: 0.5, SparseScore: 0.4}
	other := ragout.ScoredChunk{Chunk: ragout.Chunk{ID: "other"}, SparseScore: 0.1}

	dense := []ragout.ScoredChunk{only, both}
	sparse := []ragout.ScoredChunk{both, other}

	got := ReciprocalRankFusion(dense, sparse, 10, 60)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if got[0].Chunk.ID != "both" || got[1].Chunk.ID != "only" || got[2].Chunk.ID != "other" {
		t.Fatalf("order = %s %s %s, want both only other", got[0].Chunk.ID, got[1].Chunk.ID, got[2].Chunk.ID)
	}
	if got[0].DenseScore != 0.5 || got[0].SparseScore != 0.4 {
		t.Fatalf("both raw scores = dense %v sparse %v", got[0].DenseScore, got[0].SparseScore)
	}

	top := ReciprocalRankFusion(dense, sparse, 1, 60)
	if len(top) != 1 || top[0].Chunk.ID != "both" {
		t.Fatalf("top1 = %v, want both", top)
	}
}
