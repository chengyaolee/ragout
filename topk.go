package ragout

import (
	"container/heap"
	"slices"
)

type scoredChunkMinHeap []ScoredChunk

func (h scoredChunkMinHeap) Len() int           { return len(h) }
func (h scoredChunkMinHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h scoredChunkMinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *scoredChunkMinHeap) Push(x any) {
	*h = append(*h, x.(ScoredChunk))
}

func (h *scoredChunkMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// SelectTopK selects the top-k highest scoring chunks from candidates.
// If len(candidates) <= k, it sorts candidates in descending order.
// If len(candidates) > k, it maintains a min-heap of capacity k in O(M log K) time.
// The returned slice is sorted in descending order of score.
func SelectTopK(candidates []ScoredChunk, k int) []ScoredChunk {
	if k <= 0 || len(candidates) == 0 {
		return []ScoredChunk{}
	}
	if k >= len(candidates) {
		slices.SortFunc(candidates, func(a, b ScoredChunk) int {
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
			return 0
		})
		return candidates
	}

	h := make(scoredChunkMinHeap, k)
	copy(h, candidates[:k])
	heap.Init(&h)

	for _, cand := range candidates[k:] {
		if cand.Score > h[0].Score {
			h[0] = cand
			heap.Fix(&h, 0)
		}
	}

	res := make([]ScoredChunk, k)
	for i := k - 1; i >= 0; i-- {
		res[i] = heap.Pop(&h).(ScoredChunk)
	}
	return res
}
