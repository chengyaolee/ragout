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
			// Falls back to the char-count heuristic if the BPE vocab
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
	return cb.BuildPromptWithHistory(systemPrompt, query, nil, candidates)
}

const historyHeader = "Conversation so far (for reference only; do not cite it):\n"

// BuildPromptWithHistory is BuildPrompt with earlier turns of the conversation placed
// before the context. History gets at most a quarter of the budget; the newest turns
// are kept and older ones dropped once the next would not fit.
func (cb *ContextBudgeter) BuildPromptWithHistory(systemPrompt string, query string, history []ragout.Turn, candidates []ragout.ScoredChunk) (string, []ragout.ScoredChunk) {
	const templateOverhead = "Context Information:\n\nUser Question:\n\n\nAnswer based strictly on the context above. Cite sources using [1], [2], etc."
	baseCost := cb.counter.CountTokens(templateOverhead)
	if systemPrompt != "" {
		baseCost += cb.counter.CountTokens(systemPrompt + "\n\n")
	}
	baseCost += cb.counter.CountTokens(query)

	historyStr := cb.assembleHistory(history, cb.maxTokens/4)
	baseCost += cb.counter.CountTokens(historyStr)

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

	prompt.WriteString(historyStr)
	prompt.WriteString("Context Information:\n")
	prompt.WriteString(contextStr)
	prompt.WriteString("User Question:\n")
	prompt.WriteString(query)
	prompt.WriteString("\n\nAnswer based strictly on the context above. Cite sources using [1], [2], etc.")

	return prompt.String(), selected
}

// assembleHistory renders the newest turns that fit in budget tokens, oldest first.
func (cb *ContextBudgeter) assembleHistory(history []ragout.Turn, budget int) string {
	if len(history) == 0 {
		return ""
	}
	used := cb.counter.CountTokens(historyHeader + "\n")
	start := len(history)
	for i := len(history) - 1; i >= 0; i-- {
		cost := cb.counter.CountTokens(history[i].Role + ": " + history[i].Content + "\n")
		if used+cost > budget {
			break
		}
		used += cost
		start = i
	}
	if start == len(history) {
		return ""
	}
	var b strings.Builder
	b.WriteString(historyHeader)
	for _, t := range history[start:] {
		fmt.Fprintf(&b, "%s: %s\n", t.Role, t.Content)
	}
	b.WriteString("\n")
	return b.String()
}
