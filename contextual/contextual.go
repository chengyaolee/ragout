// Package contextual implements ragout.Contextualizer with an LLM.
package contextual

import (
	"context"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/chengyaolee/ragout"
)

// LLM asks a Completer to situate each chunk within its document.
type LLM struct {
	completer   ragout.Completer
	concurrency int
}

// New returns a contextualizer making up to concurrency parallel calls (<=0 means 4).
func New(c ragout.Completer, concurrency int) *LLM {
	if concurrency <= 0 {
		concurrency = 4
	}
	return &LLM{completer: c, concurrency: concurrency}
}

const chunkPrompt = `Here is the chunk we want to situate within the whole document:
<chunk>
%s
</chunk>
Please give a short, succinct context (1-2 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval. Answer ONLY with the succinct context.`

// Contextualize returns one context per chunk. The whole document is the shared
// prefix of every call, so the provider can cache it after the first request.
// The full document is sent as-is, so a document larger than the model's
// context window fails; window it around each chunk if that bites.
func (l *LLM) Contextualize(ctx context.Context, doc ragout.Document, chunks []ragout.Chunk) ([]string, error) {
	out := make([]string, len(chunks))
	if len(chunks) == 0 {
		return out, nil
	}
	prefix := "<document>\n" + doc.Content + "\n</document>"
	one := func(ctx context.Context, i int) error {
		s, err := l.completer.Complete(ctx, prefix, strings.Replace(chunkPrompt, "%s", chunks[i].Content, 1))
		out[i] = strings.TrimSpace(s)
		return err
	}

	// First call alone: concurrent first calls would all miss the cache they are about to create.
	if err := one(ctx, 0); err != nil {
		return nil, err
	}
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(l.concurrency)
	for i := 1; i < len(chunks); i++ {
		g.Go(func() error { return one(gctx, i) })
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}
