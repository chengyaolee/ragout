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

// TestBudgeter_KeepsTopRankedWhenOverBudget guards against a prior bug where
// BuildPrompt reordered candidates (moving rank #2 to the tail) before budgeting, and
// AssembleContextWithBudget stopped at the first chunk that didn't fit. Under a tight
// budget that combination dropped rank #2 in favor of rank #3. Selection must run in
// relevance order (skipping, not stopping, at an oversized chunk); reordering happens
// only after the budget has picked the chunks to keep.
func TestBudgeter_KeepsTopRankedWhenOverBudget(t *testing.T) {
	counter := byteCounter{}
	ids := []string{"rank1", "rank2", "rank3", "rank4", "rank5"}
	candidates := make([]ragout.ScoredChunk, len(ids))
	for i, id := range ids {
		candidates[i] = ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id, Content: strings.Repeat("x", 20)}}
	}

	citationCost := func(id string) int {
		return counter.CountTokens(fmt.Sprintf("[0] (ID: %s)\n%s\n\n", id, strings.Repeat("x", 20)))
	}
	budget := citationCost("rank1") + citationCost("rank2")

	_, selected := NewContextBudgeter(counter, budget).AssembleContext(candidates)

	got := make(map[string]bool, len(selected))
	for _, sc := range selected {
		got[sc.Chunk.ID] = true
	}
	if len(selected) != 2 || !got["rank1"] || !got["rank2"] {
		t.Fatalf("selected = %v, want exactly {rank1, rank2}", got)
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

func TestBuildPromptWithHistory(t *testing.T) {
	cb := NewContextBudgeter(NewHeuristicTokenCounter(), 400) // history gets at most 100 tokens
	history := []ragout.Turn{
		{Role: "user", Content: strings.Repeat("old ", 200)}, // too big: dropped with everything before it
		{Role: "assistant", Content: "Raft uses terms."},
		{Role: "user", Content: "What about leaders?"},
	}
	chunks := []ragout.ScoredChunk{{Chunk: ragout.Chunk{ID: "c1", Content: "Leaders send heartbeats."}}}

	prompt, sources := cb.BuildPromptWithHistory("", "How are leaders elected?", history, chunks)
	if len(sources) != 1 {
		t.Fatalf("sources = %d, want the chunk to still fit", len(sources))
	}
	if strings.Contains(prompt, "old old") || !strings.Contains(prompt, "assistant: Raft uses terms.\nuser: What about leaders?") {
		t.Errorf("want only the newest turns that fit, in order:\n%s", prompt)
	}
	if strings.Index(prompt, "Conversation so far") > strings.Index(prompt, "Context Information") {
		t.Errorf("history should precede the context:\n%s", prompt)
	}

	plain, _ := cb.BuildPrompt("", "How are leaders elected?", chunks)
	if strings.Contains(plain, "Conversation so far") {
		t.Errorf("BuildPrompt must not add a history block")
	}
}
