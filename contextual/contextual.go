// Package contextual implements LLM-written ragout.Contextualizers. The engine's
// default, ragout.MetadataContext, needs no model; use this when headings and titles
// are not enough to tell chunks apart.
package contextual

import (
	"context"
	"strings"
	"unicode/utf8"

	"golang.org/x/sync/errgroup"

	"github.com/chengyaolee/ragout"
)

// LLM asks a Completer for a sentence or two situating each chunk in its document
// (Anthropic's contextual retrieval). The result is the document's breadcrumb plus
// that sentence, so switching from MetadataContext keeps the title and section.
type LLM struct {
	completer   ragout.Completer
	concurrency int
	window      int
}

// Option configures LLM.
type Option func(*LLM)

// WithConcurrency sets how many calls run in parallel. Default 4.
func WithConcurrency(n int) Option {
	return func(l *LLM) {
		if n > 0 {
			l.concurrency = n
		}
	}
}

// WithWindow caps how many bytes of the document each call sees. A document that fits
// is sent whole as a prefix shared by every call, which providers can cache. A larger
// one is cut to a window around each chunk, which keeps the cost per chunk bounded
// and the prompt inside small context windows (e.g. Ollama's default). Default 8000.
func WithWindow(bytes int) Option {
	return func(l *LLM) {
		if bytes > 0 {
			l.window = bytes
		}
	}
}

func New(c ragout.Completer, opts ...Option) *LLM {
	l := &LLM{completer: c, concurrency: 4, window: 8000}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

const chunkPrompt = `Here is the chunk we want to situate within the whole document:
<chunk>
%s
</chunk>
Please give a short, succinct context (1-2 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval. Answer ONLY with the succinct context.`

func (l *LLM) Contextualize(ctx context.Context, doc ragout.Document, chunks []ragout.Chunk) ([]string, error) {
	out := make([]string, len(chunks))
	if len(chunks) == 0 {
		return out, nil
	}
	crumb := ragout.Breadcrumb(doc.Metadata)
	whole := len(doc.Content) <= l.window

	one := func(ctx context.Context, i int) error {
		text := doc.Content
		if !whole {
			text = window(doc.Content, chunks[i], i, len(chunks), l.window)
		}
		prefix := "<document>\n" + text + "\n</document>"
		if crumb != "" {
			prefix = "Document: " + crumb + "\n" + prefix
		}
		s, err := l.completer.Complete(ctx, prefix, strings.Replace(chunkPrompt, "%s", chunks[i].Content, 1))
		out[i] = strings.TrimSpace(crumb + "\n" + strings.TrimSpace(s))
		return err
	}

	start := 0
	if whole {
		// Shared prefix: send one call alone so the cache exists before the rest fan out.
		if err := one(ctx, 0); err != nil {
			return nil, err
		}
		start = 1
	}
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(l.concurrency)
	for i := start; i < len(chunks); i++ {
		g.Go(func() error { return one(gctx, i) })
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

// window returns about size bytes of content centered on chunk, cut on rune boundaries.
// Chunks whose text cannot be found are placed proportionally to their index.
func window(content string, chunk ragout.Chunk, i, n, size int) string {
	center := strings.Index(content, chunk.Content)
	if center < 0 {
		center = len(content) * i / n
	} else {
		center += len(chunk.Content) / 2
	}
	start := min(max(0, center-size/2), len(content)-size)
	end := start + size
	for start > 0 && !utf8.RuneStart(content[start]) {
		start++
	}
	for end < len(content) && !utf8.RuneStart(content[end]) {
		end--
	}
	return content[start:end]
}
