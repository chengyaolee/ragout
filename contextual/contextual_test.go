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
	doc := ragout.Document{Content: "Acme Corp annual report. Revenue grew 24%."}
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

	got, err := New(c, 2).Contextualize(context.Background(), doc, chunks)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"ctx-one", "ctx-two", "ctx-three"} {
		if got[i] != want {
			t.Errorf("context[%d] = %q, want %q", i, got[i], want)
		}
	}
}

func TestLLM_ContextualizeError(t *testing.T) {
	boom := errors.New("boom")
	c := completerFunc(func(context.Context, string, string) (string, error) { return "", boom })
	if _, err := New(c, 2).Contextualize(context.Background(), ragout.Document{}, []ragout.Chunk{{Content: "x"}, {Content: "y"}}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
}
