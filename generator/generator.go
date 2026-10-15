package generator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"

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

// doJSON sends req and decodes a 200 response body into out.
func doJSON(c *http.Client, req *http.Request, name string, out any) error {
	resp, err := c.Do(req)
	if err != nil {
		return fmt.Errorf("%s: http request failed: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<10))
		return fmt.Errorf("%s: bad status code %d: %s", name, resp.StatusCode, b)
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("%s: decode response: %w", name, err)
	}
	return nil
}
