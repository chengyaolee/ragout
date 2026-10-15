package ragout

// RRF returns reciprocal rank fusion with smoothing constant k (values <= 0 use 60).
// It is the engine's default FusionStrategy.
func RRF(k int) FusionStrategy {
	if k <= 0 {
		k = 60
	}
	return rrf{k: k}
}

type rrf struct{ k int }

func (r rrf) Fuse(lists [][]ScoredChunk, topK int) []ScoredChunk {
	return fuseRRF(lists, topK, r.k)
}

// ReciprocalRankFusion combines two ranked lists of ScoredChunk (dense and sparse) using reciprocal rank fusion.
// dense and sparse are ranked best-first. topK is the number of results to return.
// k is the smoothing constant (values <= 0 use 60).
func ReciprocalRankFusion(dense []ScoredChunk, sparse []ScoredChunk, topK int, k int) []ScoredChunk {
	return fuseRRF([][]ScoredChunk{dense, sparse}, topK, k)
}

// fuseRRF scores each chunk by the sum of 1/(k+rank) over every list it appears in.
// A fused chunk keeps the best DenseScore and SparseScore seen, and prefers a copy
// that carries its embedding.
func fuseRRF(lists [][]ScoredChunk, topK int, k int) []ScoredChunk {
	if topK <= 0 {
		return []ScoredChunk{}
	}
	if k <= 0 {
		k = 60
	}

	// Assumption: each list is sorted in descending order of score
	fused := make(map[string]ScoredChunk)
	for _, list := range lists {
		for i, sc := range list {
			f, seen := fused[sc.Chunk.ID]
			if !seen || (len(f.Chunk.Embedding) == 0 && len(sc.Chunk.Embedding) > 0) {
				f.Chunk = sc.Chunk
			}
			f.DenseScore = bestScore(f.DenseScore, sc.DenseScore)
			f.SparseScore = bestScore(f.SparseScore, sc.SparseScore)
			f.Score += 1.0 / float64(k+i+1)
			fused[sc.Chunk.ID] = f
		}
	}

	ptr := acquireCandidateSlice()
	results := (*ptr)[:0]
	for _, chunk := range fused {
		results = append(results, chunk)
	}

	top := SelectTopK(results, topK)
	*ptr = top
	return *ptr
}

// bestScore keeps the higher of two scores, treating zero as "not scored by this
// retriever" so a negative cosine is not replaced by an unset value.
func bestScore(cur, next float64) float64 {
	if next != 0 && (cur == 0 || next > cur) {
		return next
	}
	return cur
}
