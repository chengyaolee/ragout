//go:build !race

package ragout_test

import (
	"testing"

	"github.com/chengyaolee/ragout"
)

func TestCandidateSlicePoolZeroAlloc(t *testing.T) {
	p := ragout.AcquireCandidateSlice()
	ragout.ReleaseCandidateSlice(p)

	allocs := testing.AllocsPerRun(100, func() {
		p := ragout.AcquireCandidateSlice()
		s := *p
		s = append(s, ragout.ScoredChunk{})
		*p = s
		ragout.ReleaseCandidateSlice(p)
	})
	if allocs != 0 {
		t.Fatalf("allocs/op = %v, want 0", allocs)
	}
}
