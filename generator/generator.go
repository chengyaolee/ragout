package generator

import (
	"context"
	"iter"
	"strings"

	"github.com/chengyaolee/ragout"
)

// MockGenerator streams scripted tokens, then the retrieved chunk IDs.
// Its sources are the candidates it was given, in order.
type MockGenerator struct {
	tokensToStream []string
}

func NewMockGenerator(tokens []string) *MockGenerator {
	return &MockGenerator{tokensToStream: tokens}
}

func (mg *MockGenerator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	sources := append([]ragout.ScoredChunk(nil), candidates...)
	return sources, func(yield func(string, error) bool) {
		for _, token := range mg.tokensToStream {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			if !yield(token, nil) {
				return
			}
		}
		for _, c := range candidates {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			if !yield(c.Chunk.ID, nil) {
				return
			}
		}
	}
}

func joinPrompt(prefix, prompt string) string {
	if prefix == "" {
		return prompt
	}
	return prefix + "\n\n" + prompt
}

// collect drains a token stream into one string.
func collect(tokens iter.Seq2[string, error]) (string, error) {
	var b strings.Builder
	for token, err := range tokens {
		if err != nil {
			return b.String(), err
		}
		b.WriteString(token)
	}
	return b.String(), nil
}

// lastTurns returns the final n turns of history.
func lastTurns(history []ragout.Turn, n int) []ragout.Turn {
	return history[max(0, len(history)-n):]
}
