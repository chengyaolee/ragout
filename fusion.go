package ragout

// RRF returns reciprocal rank fusion with smoothing constant k (values <= 0 use 60).
// Every list counts the same.
func RRF(k int) FusionStrategy { return WeightedRRF(k, 1) }

// DefaultRewriteWeight is how much a query transformer's added searches count in the
// engine's default fusion, relative to the question's own searches. On this module's
// docs benchmark, equal weight let MultiQuery and Decompose rephrasings outvote the
// question and lowered hit rate; at 0.5 MultiQuery matched the no-rewrite baseline and
// Decompose recovered most of its loss (see eval/results).
const DefaultRewriteWeight = 0.5

// WeightedRRF is reciprocal rank fusion in which lists from a query transformer's added
// searches count rewriteWeight times as much as the question's own lists (values <= 0
// use 1). With no transformer it is identical to RRF.
//
// Rephrasings and sub-questions are noisier than the question they came from. At equal
// weight, several of them can outvote it and push its best hits down the ranking.
func WeightedRRF(k int, rewriteWeight float64) FusionStrategy {
	if k <= 0 {
		k = 60
	}
	if rewriteWeight <= 0 {
		rewriteWeight = 1
	}
	return rrf{k: k, rewrite: rewriteWeight}
}

type rrf struct {
	k       int
	rewrite float64
}

func (r rrf) Fuse(lists []RankedList, topK int) []ScoredChunk {
	return fuseRRF(lists, topK, r.k, r.rewrite)
}

// ReciprocalRankFusion combines two ranked lists of ScoredChunk (dense and sparse) using reciprocal rank fusion.
// dense and sparse are ranked best-first. topK is the number of results to return.
// k is the smoothing constant (values <= 0 use 60).
func ReciprocalRankFusion(dense []ScoredChunk, sparse []ScoredChunk, topK int, k int) []ScoredChunk {
	return fuseRRF([]RankedList{{Dense: true, Hits: dense}, {Hits: sparse}}, topK, k, 1)
}

// fuseRRF scores each chunk by the sum of weight/(k+rank) over every list it appears
// in. A fused chunk keeps the best DenseScore and SparseScore seen, and prefers a copy
// that carries its embedding.
func fuseRRF(lists []RankedList, topK int, k int, rewriteWeight float64) []ScoredChunk {
	if topK <= 0 {
		return []ScoredChunk{}
	}
	if k <= 0 {
		k = 60
	}

	// Assumption: each list is sorted in descending order of score
	fused := make(map[string]ScoredChunk)
	for _, list := range lists {
		weight := 1.0
		if list.Rewrite {
			weight = rewriteWeight
		}
		for i, sc := range list.Hits {
			f, seen := fused[sc.Chunk.ID]
			if !seen || (len(f.Chunk.Embedding) == 0 && len(sc.Chunk.Embedding) > 0) {
				f.Chunk = sc.Chunk
			}
			f.DenseScore = bestScore(f.DenseScore, sc.DenseScore)
			f.SparseScore = bestScore(f.SparseScore, sc.SparseScore)
			f.Score += weight / float64(k+i+1)
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
