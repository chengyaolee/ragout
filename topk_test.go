package ragout_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/chengyaolee/ragout"
)

func TestSelectTopK(t *testing.T) {
	t.Run("empty or invalid k", func(t *testing.T) {
		if len(ragout.SelectTopK(nil, 5)) != 0 {
			t.Fatal("expected empty result")
		}
		chunks := []ragout.ScoredChunk{{Score: 1.0}}
		if len(ragout.SelectTopK(chunks, 0)) != 0 {
			t.Fatal("expected empty result for k=0")
		}
	})

	t.Run("fewer than k candidates", func(t *testing.T) {
		chunks := []ragout.ScoredChunk{
			{Score: 2.0},
			{Score: 5.0},
			{Score: 1.0},
		}
		top := ragout.SelectTopK(chunks, 5)
		if len(top) != 3 {
			t.Fatalf("expected 3, got %d", len(top))
		}
		if top[0].Score != 5.0 || top[1].Score != 2.0 || top[2].Score != 1.0 {
			t.Fatalf("unexpected order: %v", top)
		}
	})

	t.Run("more than k candidates matches sort.Slice", func(t *testing.T) {
		n := 1000
		k := 20
		chunks := make([]ragout.ScoredChunk, n)
		for i := 0; i < n; i++ {
			chunks[i] = ragout.ScoredChunk{
				Score: rand.Float64() * 100,
			}
		}

		expected := make([]ragout.ScoredChunk, n)
		copy(expected, chunks)
		slices.SortFunc(expected, func(a, b ragout.ScoredChunk) int {
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
			return 0
		})
		expected = expected[:k]

		got := ragout.SelectTopK(chunks, k)
		if len(got) != k {
			t.Fatalf("expected len %d, got %d", k, len(got))
		}
		for i := 0; i < k; i++ {
			if got[i].Score != expected[i].Score {
				t.Fatalf("mismatch at index %d: got %f, want %f", i, got[i].Score, expected[i].Score)
			}
		}
	})
}
