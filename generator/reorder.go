package generator

import "github.com/chengyaolee/ragout"

// ReorderLostInTheMiddle places high-relevance chunks at the beginning and end of the context.
func ReorderLostInTheMiddle(candidates []ragout.ScoredChunk) []ragout.ScoredChunk {
	n := len(candidates)
	if n <= 2 {
		return candidates
	}

	reordered := make([]ragout.ScoredChunk, n)
	left := 0
	right := n - 1

	for i, cand := range candidates {
		if i%2 == 0 {
			reordered[left] = cand
			left++
		} else {
			reordered[right] = cand
			right--
		}
	}

	return reordered
}
