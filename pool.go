package ragout

import "sync"

var scoredChunkPool = sync.Pool{
	New: func() any {
		// Pre-allocate capacity for topKRecall
		slice := make([]ScoredChunk, 0, 64)
		return &slice
	},
}

func acquireCandidateSlice() *[]ScoredChunk {
	ptr := scoredChunkPool.Get().(*[]ScoredChunk)
	*ptr = (*ptr)[:0] // Reset length to 0 while preserving allocated backing array
	return ptr
}

func releaseCandidateSlice(ptr *[]ScoredChunk) {
	if cap(*ptr) <= 128 { // Avoid retaining abnormally bloated slices
		clear((*ptr)[:cap(*ptr)])
		*ptr = (*ptr)[:0]
		scoredChunkPool.Put(ptr)
	}
}

// AcquireCandidateSlice borrows a candidate buffer. Release it with ReleaseCandidateSlice.
func AcquireCandidateSlice() *[]ScoredChunk { return acquireCandidateSlice() }

// ReleaseCandidateSlice returns a buffer from AcquireCandidateSlice.
func ReleaseCandidateSlice(ptr *[]ScoredChunk) { releaseCandidateSlice(ptr) }

// releaseCandidates puts a returned candidate slice back in the pool.
// ponytail: &buf is a fresh header each call; the backing array is what is reused. Cap above 128 is dropped. Upgrade path: hand the *[]ScoredChunk through VectorStore if the header alloc shows up in profiles.
func releaseCandidates(s []ScoredChunk) {
	if cap(s) == 0 || cap(s) > 128 {
		return
	}
	buf := s[:0]
	releaseCandidateSlice(&buf)
}

func sameBacking(a, b []ScoredChunk) bool {
	if cap(a) == 0 || cap(b) == 0 {
		return false
	}
	return &a[:cap(a)][0] == &b[:cap(b)][0]
}
