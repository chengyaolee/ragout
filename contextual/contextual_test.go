package contextual

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/chengyaolee/ragout"
)

type completerFunc func(ctx context.Context, prefix, prompt string) (string, error)

func (f completerFunc) Complete(ctx context.Context, prefix, prompt string) (string, error) {
	return f(ctx, prefix, prompt)
}

func TestLLM_Contextualize(t *testing.T) {
	doc := ragout.Document{Content: "Acme Corp annual report. Revenue grew 24%.", Metadata: map[string]any{ragout.MetadataTitle: "Acme 2024"}}
	chunks := []ragout.Chunk{{Content: "one"}, {Content: "two"}, {Content: "three"}}

	var mu sync.Mutex
	var first string
	c := completerFunc(func(_ context.Context, prefix, prompt string) (string, error) {
		mu.Lock()
		defer mu.Unlock()
		if first == "" {
			first = prefix
		}
		if prefix != first || !strings.Contains(prefix, doc.Content) {
			t.Errorf("prefix must be identical and contain the document, got %q", prefix)
		}
		for _, ch := range chunks {
			if strings.Contains(prompt, "<chunk>\n"+ch.Content+"\n</chunk>") {
				return " ctx-" + ch.Content + " ", nil
			}
		}
		t.Errorf("prompt has no known chunk: %q", prompt)
		return "", nil
	})

	got, err := New(c, WithConcurrency(2)).Contextualize(context.Background(), doc, chunks)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"Acme 2024\nctx-one", "Acme 2024\nctx-two", "Acme 2024\nctx-three"} {
		if got[i] != want {
			t.Errorf("context[%d] = %q, want %q", i, got[i], want)
		}
	}
}

func TestLLM_ContextualizeError(t *testing.T) {
	boom := errors.New("boom")
	c := completerFunc(func(context.Context, string, string) (string, error) { return "", boom })
	if _, err := New(c).Contextualize(context.Background(), ragout.Document{}, []ragout.Chunk{{Content: "x"}, {Content: "y"}}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
}

func TestLLM_WindowBoundsLargeDocuments(t *testing.T) {
	content := strings.Repeat("filler ", 2000) + "NEEDLE chunk text" + strings.Repeat(" tail", 2000)
	doc := ragout.Document{Content: content}
	chunk := ragout.Chunk{Content: "NEEDLE chunk text"}

	var prefix string
	c := completerFunc(func(_ context.Context, p, _ string) (string, error) { prefix = p; return "ctx", nil })
	if _, err := New(c, WithWindow(1000)).Contextualize(context.Background(), doc, []ragout.Chunk{chunk}); err != nil {
		t.Fatal(err)
	}
	if len(prefix) > 1100 || !strings.Contains(prefix, "NEEDLE chunk text") {
		t.Errorf("prefix should be a ~1000 byte window around the chunk, got %d bytes", len(prefix))
	}
}
