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
		counter = NewHeuristicTokenCounter()
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
func (cb *ContextBudgeter) AssembleContext(candidates []ragout.ScoredChunk) (string, []ragout.Chunk) {
	return cb.AssembleContextWithBudget(candidates, cb.maxTokens)
}

// AssembleContextWithBudget greedily selects candidates fitting within a specific budget.
func (cb *ContextBudgeter) AssembleContextWithBudget(candidates []ragout.ScoredChunk, budget int) (string, []ragout.Chunk) {
	var b strings.Builder
	var selected []ragout.Chunk
	accumulatedTokens := 0

	for i, sc := range candidates {
		citation := fmt.Sprintf("[%d] (ID: %s)\n%s\n\n", i+1, sc.Chunk.ID, sc.Chunk.Content)
		cost := cb.counter.CountTokens(citation)

		if accumulatedTokens+cost > budget {
			break // Ceiling reached
		}

		b.WriteString(citation)
		selected = append(selected, sc.Chunk)
		accumulatedTokens += cost
	}

	return b.String(), selected
}

// BuildPrompt combines system prompt, assembled context, and user query within cb.maxTokens.
func (cb *ContextBudgeter) BuildPrompt(systemPrompt string, query string, candidates []ragout.ScoredChunk) (string, []ragout.Chunk) {
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
