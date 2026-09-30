package generator

// TokenCounter measures text in model tokens.
type TokenCounter interface {
	CountTokens(text string) int
}

// HeuristicTokenCounter estimates tokens from character length.
// It needs no vocabulary files or network access.
type HeuristicTokenCounter struct {
	CharsPerToken float64 // Standard default: 4.0
}

func NewHeuristicTokenCounter() *HeuristicTokenCounter {
	return &HeuristicTokenCounter{CharsPerToken: 4.0}
}

func (h *HeuristicTokenCounter) CountTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	ratio := h.CharsPerToken
	if ratio <= 0 {
		ratio = 4.0
	}
	return max(1, int(float64(len(text))/ratio))
}
