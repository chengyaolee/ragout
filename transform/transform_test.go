package transform

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

type completerFunc func(ctx context.Context, prefix, prompt string) (string, error)

func (f completerFunc) Complete(ctx context.Context, prefix, prompt string) (string, error) {
	return f(ctx, prefix, prompt)
}

func reply(s string, prompts *[]string) completerFunc {
	return func(_ context.Context, _, prompt string) (string, error) {
		if prompts != nil {
			*prompts = append(*prompts, prompt)
		}
		return s, nil
	}
}

func TestLines(t *testing.T) {
	got := lines("<think>\nmaybe 1. this\n</think>\nHere are 3 queries:\n1. foo bar\n2) Baz\n- \"qux\"\n* Foo  Bar\n**the question**\nextra", 3, "The question")
	want := []string{"foo bar", "Baz", "qux"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("lines = %q, want %q", got, want)
	}
}

func TestCondense(t *testing.T) {
	var prompts []string
	c := Condense(reply("What was Acme's revenue in 2023?\n", &prompts))

	rw, err := c.Transform(context.Background(), "and 2023?", nil)
	if err != nil || rw.Question != "" || len(prompts) != 0 {
		t.Fatalf("no history must not call the model: %+v %v %d", rw, err, len(prompts))
	}

	var history []ragout.Turn
	for i := range 8 {
		history = append(history, ragout.Turn{Role: "user", Content: strings.Repeat("x", 2000) + string(rune('a'+i))})
	}
	history[7].Content = "Acme revenue 2024?"
	rw, err = c.Transform(context.Background(), "and 2023?", history)
	if err != nil || rw.Question != "What was Acme's revenue in 2023?" {
		t.Fatalf("Transform = %+v, %v", rw, err)
	}
	p := prompts[0]
	if strings.Count(p, "user: ") != historyTurns || !strings.Contains(p, "Acme revenue 2024?") || len(p) > historyTurns*(historyTurnSize+20)+800 {
		t.Errorf("prompt should hold the last %d turns, each truncated:\n%d bytes", historyTurns, len(p))
	}
}

func TestMultiQueryDecomposeHyDE(t *testing.T) {
	ctx := context.Background()
	q := "how do nodes agree"

	rw, _ := MultiQuery(reply("1. raft consensus\n2. paxos\n3. how do nodes agree\n4. leader election\n5. extra", nil), 3).Transform(ctx, q, nil)
	want := []ragout.Search{{Text: "raft consensus"}, {Text: "paxos"}, {Text: "leader election"}}
	if !reflect.DeepEqual(rw.Searches, want) {
		t.Errorf("MultiQuery = %+v", rw.Searches)
	}

	rw, _ = Decompose(reply("How do nodes agree", nil), 4).Transform(ctx, q, nil)
	if len(rw.Searches) != 0 {
		t.Errorf("a simple question must add no sub-questions, got %+v", rw.Searches)
	}

	rw, _ = HyDE(reply("<think>hmm</think>\nNodes agree via Raft.", nil)).Transform(ctx, q, nil)
	if len(rw.Searches) != 1 || rw.Searches[0] != (ragout.Search{Text: "Nodes agree via Raft.", Target: ragout.SearchDense}) {
		t.Errorf("HyDE = %+v", rw.Searches)
	}
}

func TestChain(t *testing.T) {
	var prompts []string
	boom := errors.New("boom")
	condensed := Condense(reply("standalone q", nil))
	expand := MultiQuery(reply("alt one", &prompts), 2)
	failing := HyDE(completerFunc(func(context.Context, string, string) (string, error) { return "", boom }))

	rw, err := Chain(condensed, expand, failing).Transform(context.Background(), "follow-up", []ragout.Turn{{Role: "user", Content: "hi"}})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	if rw.Question != "standalone q" || len(rw.Searches) != 1 || rw.Searches[0].Text != "alt one" {
		t.Errorf("partial result lost: %+v", rw)
	}
	if !strings.Contains(prompts[0], "Question: standalone q") {
		t.Errorf("MultiQuery should see the condensed question: %q", prompts[0])
	}
}
