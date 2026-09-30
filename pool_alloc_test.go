//go:build !race

package ragout

import "testing"

func TestCandidateSlicePoolZeroAlloc(t *testing.T) {
	p := acquireCandidateSlice()
	releaseCandidateSlice(p)

	allocs := testing.AllocsPerRun(100, func() {
		p := acquireCandidateSlice()
		s := *p
		s = append(s, ScoredChunk{})
		*p = s
		releaseCandidateSlice(p)
	})
	if allocs != 0 {
		t.Fatalf("allocs/op = %v, want 0", allocs)
	}
}
