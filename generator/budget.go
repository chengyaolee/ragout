package generator

import (
	"fmt"
	"strings"

	"github.com/chengyaolee/ragout"
)

type ContextBudgeter struct {
	counter   TokenCounter
	maxTokens int
}

func NewContextBudgeter(counter TokenCounter, maxTokens int) *ContextBudgeter {
	if counter == nil {
		if tc, err := NewTiktokenCounter("cl100k_base"); err == nil {
			counter = tc
		} else {
			// ponytail: falls back to the char-count heuristic if the BPE vocab
			// file can't load (e.g. offline); upgrade by vendoring the encoding.
			counter = NewHeuristicTokenCounter()
		}
	}
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	return &ContextBudgeter{
		counter:   counter,
		maxTokens: maxTokens,
	}
}

// AssembleContext greedily selects candidates that fit within the token budget.
func (cb *ContextBudgeter) AssembleContext(candidates []ragout.ScoredChunk) (string, []ragout.ScoredChunk) {
	return cb.AssembleContextWithBudget(candidates, cb.maxTokens)
}

// AssembleContextWithBudget greedily selects candidates fitting within a specific budget.
// Candidates are tried in relevance order; one that doesn't fit is skipped (not a stop),
// so a single oversized chunk doesn't starve the rest of the budget. Selected chunks are
// then reordered to mitigate "lost in the middle" before citation numbers are assigned.
func (cb *ContextBudgeter) AssembleContextWithBudget(candidates []ragout.ScoredChunk, budget int) (string, []ragout.ScoredChunk) {
	var selected []ragout.ScoredChunk
	accumulatedTokens := 0

	for _, sc := range candidates {
		// Citation index doesn't affect token count meaningfully; count against a
		// placeholder since the real index is only known after reordering below.
		citation := fmt.Sprintf("[0] (ID: %s)\n%s\n\n", sc.Chunk.ID, sc.Chunk.Content)
		cost := cb.counter.CountTokens(citation)
		if accumulatedTokens+cost > budget {
			continue
		}
		selected = append(selected, sc)
		accumulatedTokens += cost
	}

	selected = ReorderLostInTheMiddle(selected)

	var b strings.Builder
	for i, sc := range selected {
		fmt.Fprintf(&b, "[%d] (ID: %s)\n%s\n\n", i+1, sc.Chunk.ID, sc.Chunk.Content)
	}

	return b.String(), selected
}

// BuildPrompt combines system prompt, assembled context, and user query within cb.maxTokens.
// It returns the chunks placed in the prompt, in citation order (sources[i] is cited as [i+1]).
func (cb *ContextBudgeter) BuildPrompt(systemPrompt string, query string, candidates []ragout.ScoredChunk) (string, []ragout.ScoredChunk) {
	const templateOverhead = "Context Information:\n\nUser Question:\n\n\nAnswer based strictly on the context above. Cite sources using [1], [2], etc."
	baseCost := cb.counter.CountTokens(templateOverhead)
	if systemPrompt != "" {
		baseCost += cb.counter.CountTokens(systemPrompt + "\n\n")
	}
	baseCost += cb.counter.CountTokens(query)

	remainingBudget := cb.maxTokens - baseCost
	if remainingBudget < 0 {
		remainingBudget = 0
	}

	contextStr, selected := cb.AssembleContextWithBudget(candidates, remainingBudget)

	var prompt strings.Builder
	if systemPrompt != "" {
		prompt.WriteString(systemPrompt)
		prompt.WriteString("\n\n")
	}

	prompt.WriteString("Context Information:\n")
	prompt.WriteString(contextStr)
	prompt.WriteString("User Question:\n")
	prompt.WriteString(query)
	prompt.WriteString("\n\nAnswer based strictly on the context above. Cite sources using [1], [2], etc.")

	return prompt.String(), selected
}
