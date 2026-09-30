package ragout

import "testing"

func BenchmarkCandidateSlicePool(b *testing.B) {
	p := acquireCandidateSlice()
	releaseCandidateSlice(p)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := acquireCandidateSlice()
		s := *p
		s = append(s, ScoredChunk{})
		*p = s
		releaseCandidateSlice(p)
	}
}
