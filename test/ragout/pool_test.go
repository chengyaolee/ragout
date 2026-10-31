package ragout_test

import (
	"testing"

	"github.com/chengyaolee/ragout"
)

func BenchmarkCandidateSlicePool(b *testing.B) {
	p := ragout.AcquireCandidateSlice()
	ragout.ReleaseCandidateSlice(p)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := ragout.AcquireCandidateSlice()
		s := *p
		s = append(s, ragout.ScoredChunk{})
		*p = s
		ragout.ReleaseCandidateSlice(p)
	}
}
