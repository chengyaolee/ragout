package generator

import (
	"fmt"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

// byteCounter counts bytes so the ceiling test can size each citation exactly.
type byteCounter struct{}

func (byteCounter) CountTokens(text string) int { return len(text) }

func TestBudgeter_StrictCeiling(t *testing.T) {
	const (
		chunkTokens = 100
		budget      = 350
		n           = 50
	)
	counter := byteCounter{}
	candidates := make([]ragout.ScoredChunk, n)
	for i := range candidates {
		id := fmt.Sprintf("id-%02d", i)
		prefix := fmt.Sprintf("[%d] (ID: %s)\n", i+1, id)
		contentLen := chunkTokens - len(prefix) - len("\n\n")
		if contentLen <= 0 {
			t.Fatalf("citation prefix longer than %d tokens", chunkTokens)
		}
		candidates[i] = ragout.ScoredChunk{
			Chunk: ragout.Chunk{ID: id, Content: strings.Repeat("x", contentLen)},
		}
		citation := fmt.Sprintf("[%d] (ID: %s)\n%s\n\n", i+1, id, candidates[i].Chunk.Content)
		if counter.CountTokens(citation) != chunkTokens {
			t.Fatalf("chunk %d costs %d tokens, want %d", i, counter.CountTokens(citation), chunkTokens)
		}
	}

	assembled, selected := NewContextBudgeter(counter, budget).AssembleContext(candidates)
	got := counter.CountTokens(assembled)
	if got > budget {
		t.Fatalf("assembled context is %d tokens, budget is %d", got, budget)
	}
	if len(selected) != 3 || got != 300 {
		t.Fatalf("selected %d chunks (%d tokens), want 3 chunks (300 tokens)", len(selected), got)
	}
}

func TestBudgeter_LostInMiddleReordering(t *testing.T) {
	ids := []string{"C1", "C2", "C3", "C4", "C5"}
	in := make([]ragout.ScoredChunk, len(ids))
	for i, id := range ids {
		in[i] = ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id}}
	}

	out := ReorderLostInTheMiddle(in)
	want := []string{"C1", "C3", "C5", "C4", "C2"}
	if len(out) != len(want) {
		t.Fatalf("len = %d, want %d", len(out), len(want))
	}
	for i, id := range want {
		if out[i].Chunk.ID != id {
			t.Fatalf("index %d = %s, want %s", i, out[i].Chunk.ID, id)
		}
	}
}
