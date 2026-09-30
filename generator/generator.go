package generator

import (
	"context"
	"iter"
	"strings"

	"github.com/chengyaolee/ragout"
)

// StreamFromIter adapts an iter.Seq2 stream into a push-based StreamCallback.
func StreamFromIter(iter iter.Seq2[string, error], cb ragout.StreamCallback) error {
	if cb == nil {
		return ragout.ErrNilCallback
	}
	for token, err := range iter {
		if err != nil {
			return err
		}
		if err := cb(token); err != nil {
			return err
		}
	}
	return nil
}

// CollectFromIter accumulates all streamed tokens into a single complete string.
func CollectFromIter(iter iter.Seq2[string, error]) (string, error) {
	var b strings.Builder
	for token, err := range iter {
		if err != nil {
			return "", err
		}
		b.WriteString(token)
	}
	return b.String(), nil
}

// MockGenerator streams scripted tokens, then the retrieved chunk IDs.
type MockGenerator struct {
	tokensToStream []string
}

func NewMockGenerator(tokens []string) *MockGenerator {
	return &MockGenerator{tokensToStream: tokens}
}

func (mg *MockGenerator) GenerateIter(ctx context.Context, query string, candidates []ragout.ScoredChunk) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
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

func (mg *MockGenerator) GenerateStream(ctx context.Context, query string, candidates []ragout.ScoredChunk, cb ragout.StreamCallback) error {
	return StreamFromIter(mg.GenerateIter(ctx, query, candidates), cb)
}

func (mg *MockGenerator) Generate(ctx context.Context, query string, candidates []ragout.ScoredChunk) (string, error) {
	return CollectFromIter(mg.GenerateIter(ctx, query, candidates))
}
