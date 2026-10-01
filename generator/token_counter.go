package generator

import "github.com/pkoukk/tiktoken-go"

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

// TiktokenCounter counts exact BPE tokens using tiktoken-go, matching how
// OpenAI-family models (and TokenChunker) actually tokenize text.
type TiktokenCounter struct {
	encoder *tiktoken.Tiktoken
}

// NewTiktokenCounter builds a counter for the given encoding (e.g. "cl100k_base"
// for gpt-3.5/gpt-4, "o200k_base" for gpt-4o).
func NewTiktokenCounter(encodingName string) (*TiktokenCounter, error) {
	encoder, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, err
	}
	return &TiktokenCounter{encoder: encoder}, nil
}

func (t *TiktokenCounter) CountTokens(text string) int {
	if text == "" {
		return 0
	}
	return len(t.encoder.Encode(text, nil, nil))
}
