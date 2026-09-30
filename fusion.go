package ragout

import "sort"

// ReciprocalRankFusion combines two ranked lists of ScoredChunk (dense and sparse) using reciprocal rank fusion.
// dense and sparse are ranked best-first. topK is the number of results to return.
// k is the smoothing constant (values <= 0 use 60).
func ReciprocalRankFusion(dense []ScoredChunk, sparse []ScoredChunk, topK int, k int) []ScoredChunk {
	if topK <= 0 {
		return []ScoredChunk{}
	}
	if k <= 0 {
		k = 60
	}

	// Assumption: Input is sorted in descending order of score
	rrfScores := make(map[string]ScoredChunk)

	for i, chunk := range dense {
		updated := rrfScores[chunk.Chunk.ID]
		updated.Chunk = chunk.Chunk
		updated.DenseScore = chunk.DenseScore
		updated.Score += 1.0 / float64(k+i+1)
		rrfScores[chunk.Chunk.ID] = updated
	}
	for i, chunk := range sparse {
		updated := rrfScores[chunk.Chunk.ID]
		updated.Chunk = chunk.Chunk
		updated.SparseScore = chunk.SparseScore
		updated.Score += 1.0 / float64(k+i+1)
		rrfScores[chunk.Chunk.ID] = updated
	}

	ptr := acquireCandidateSlice()
	results := (*ptr)[:0]
	for _, chunk := range rrfScores {
		results = append(results, chunk)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topK > len(results) {
		topK = len(results)
	}
	*ptr = results[:topK]
	return *ptr
}
