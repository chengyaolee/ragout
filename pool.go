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
	if ptr == nil {
		return
	}
	if cap(*ptr) <= 1024 { // Retain buffers up to 1024 chunks for production workloads
		clear((*ptr)[:cap(*ptr)])
		*ptr = (*ptr)[:0]
		scoredChunkPool.Put(ptr)
	}
}

// AcquireCandidateSlice borrows a candidate buffer. Release it with ReleaseCandidateSlice.
func AcquireCandidateSlice() *[]ScoredChunk { return acquireCandidateSlice() }

// ReleaseCandidateSlice returns a buffer from AcquireCandidateSlice.
func ReleaseCandidateSlice(ptr *[]ScoredChunk) { releaseCandidateSlice(ptr) }

// releaseCandidates puts a returned candidate slice back in the pool without stack-escaping allocations.
func releaseCandidates(s []ScoredChunk) {
	if cap(s) == 0 || cap(s) > 1024 {
		return
	}
	ptr := scoredChunkPool.Get().(*[]ScoredChunk)
	clear(s[:cap(s)])
	*ptr = s[:0]
	scoredChunkPool.Put(ptr)
}

func sameBacking(a, b []ScoredChunk) bool {
	if cap(a) == 0 || cap(b) == 0 {
		return false
	}
	return &a[:cap(a)][0] == &b[:cap(b)][0]
}
