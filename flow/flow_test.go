package flow

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
)

// --- fakes ---------------------------------------------------------------

// fakeRetriever returns, per query, the chunks whose ID appears in index[query].
type fakeRetriever struct {
	index   map[string][]string
	queries []string
	history [][]ragout.Turn
}

func (f *fakeRetriever) Retrieve(_ context.Context, q string, opts ...ragout.QueryOption) (ragout.Answer, error) {
	var p ragout.QueryParams
	for _, o := range opts {
		o(&p)
	}
	f.queries = append(f.queries, q)
	f.history = append(f.history, p.History)
	ids, ok := f.index[q]
	if !ok {
		return ragout.Answer{Question: q}, ragout.ErrNoResults
	}
	ans := ragout.Answer{Question: q, Searches: []ragout.Search{{Text: q}}}
	for _, id := range ids {
		ans.Retrieved = append(ans.Retrieved, ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id, Content: "text of " + id}})
	}
	return ans, nil
}

// scriptedLLM answers each Complete call with the next reply.
type scriptedLLM struct {
	replies []string
	err     error
	prompts []string
}

func (s *scriptedLLM) Complete(_ context.Context, _, prompt string) (string, error) {
	s.prompts = append(s.prompts, prompt)
	if s.err != nil {
		return "", s.err
	}
	r := s.replies[0]
	s.replies = s.replies[1:]
	return r, nil
}

type echoGenerator struct{ question string }

func (g *echoGenerator) GenerateIter(_ context.Context, q string, c []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	g.question = q
	return c, func(yield func(string, error) bool) {
		for _, sc := range c {
			if !yield(sc.Chunk.ID+" ", nil) {
				return
			}
		}
	}
}

func steps(s *State) []string {
	var out []string
	for _, e := range s.Trace {
		out = append(out, e.Step)
	}
	return out
}

// --- CRAG ----------------------------------------------------------------

func TestCRAG_SufficientFirstHop(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"who leads raft": {"a", "b"}}}
	gen := &echoGenerator{}
	var streamed strings.Builder
	s, err := CRAG(r, &scriptedLLM{replies: []string{"<think>looks fine</think>\n1, 2"}}, gen).
		Run(context.Background(), "who leads raft", WithTokens(func(t string) error { streamed.WriteString(t); return nil }))
	if err != nil {
		t.Fatal(err)
	}
	if got := steps(s); !slices.Equal(got, []string{"retrieve", "grade", "generate"}) {
		t.Errorf("steps = %v", got)
	}
	if s.Answer.Text != "a b " || streamed.String() != "a b " || len(s.Answer.Sources) != 2 || s.Abstained {
		t.Errorf("answer = %+v, streamed %q", s.Answer, streamed.String())
	}
}

func TestCRAG_CorrectsThenAnswers(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{
		"who owns billing and where does it run": {"a"},
		"billing service hosting region":         {"b", "a"},
	}}
	llm := &scriptedLLM{replies: []string{
		"NONE\nSearch query: billing service hosting region",
		"1",
	}}
	gen := &echoGenerator{}
	s, err := CRAG(r, llm, gen).Run(context.Background(), "who owns billing and where does it run")
	if err != nil {
		t.Fatal(err)
	}
	if got := steps(s); !slices.Equal(got, []string{"retrieve", "grade", "refine", "retrieve", "grade", "generate"}) {
		t.Errorf("steps = %v", got)
	}
	if !slices.Equal(r.queries, []string{"who owns billing and where does it run", "billing service hosting region"}) {
		t.Errorf("queries = %q", r.queries)
	}
	if len(s.Evidence) != 2 || s.Hops != 2 || gen.question != "who owns billing and where does it run" {
		t.Errorf("evidence must be pooled and de-duplicated: %d chunks, %d hops, generated for %q", len(s.Evidence), s.Hops, gen.question)
	}
}

func TestCRAG_AbstainsWhenHopsRunOut(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"q": {"a"}, "more": {"b"}}}
	llm := &scriptedLLM{replies: []string{"NONE\nmore", "NONE\nstill more"}}
	s, err := CRAG(r, llm, &echoGenerator{}).Run(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if !s.Abstained || s.Answer.Text != DefaultAbstainMessage || s.Hops != 2 {
		t.Errorf("want abstain after 2 hops: %+v", s)
	}

	// With AnswerWhenExhausted it answers from what it found instead.
	llm = &scriptedLLM{replies: []string{"NONE\nmore", "NONE\nstill more"}}
	s, _ = CRAG(r, llm, &echoGenerator{}, AnswerWhenExhausted()).Run(context.Background(), "q")
	if s.Abstained || s.Answer.Text != "a b " {
		t.Errorf("want an answer from pooled evidence: %+v", s.Answer)
	}
}

func TestCRAG_NoEvidenceAbstainsWithoutCallingTheModel(t *testing.T) {
	llm := &scriptedLLM{}
	s, err := CRAG(&fakeRetriever{}, llm, &echoGenerator{}).Run(context.Background(), "unknown thing")
	if err != nil || !s.Abstained || len(llm.prompts) != 0 {
		t.Fatalf("abstained=%v prompts=%d err=%v", s.Abstained, len(llm.prompts), err)
	}
}

func TestCRAG_GraderFailureFailsOpen(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"q": {"a"}}}
	s, err := CRAG(r, &scriptedLLM{err: errors.New("model down")}, &echoGenerator{}).Run(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if s.Verdict != Ungraded || len(s.Errors) != 1 || s.Answer.Text != "a " {
		t.Errorf("a failed grader must still answer and record the error: %+v", s)
	}
}

func TestCRAG_RepeatSearchIsNotRetried(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"q": {"a"}}}
	s, _ := CRAG(r, &scriptedLLM{replies: []string{"NONE\nQ"}}, &echoGenerator{}, MaxHops(5)).Run(context.Background(), "q")
	if s.Hops != 1 || !s.Abstained {
		t.Errorf("suggesting the same search again must not loop: hops=%d", s.Hops)
	}
}

func TestRetrievePassesHistoryOnFirstHopOnly(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"q": {"a"}, "more": {"b"}}}
	llm := &scriptedLLM{replies: []string{"NONE\nmore", "1"}}
	hist := []ragout.Turn{{Role: "user", Content: "earlier"}}
	if _, err := CRAG(r, llm, &echoGenerator{}).Run(context.Background(), "q", WithHistory(hist)); err != nil {
		t.Fatal(err)
	}
	if len(r.history[0]) != 1 || r.history[1] != nil {
		t.Errorf("history per hop = %v", r.history)
	}
}

func TestParseGrade(t *testing.T) {
	for _, tc := range []struct {
		reply    string
		relevant []int
		missing  string
		ok       bool
	}{
		{"2, 5", []int{1, 4}, "", true},
		{"Passages 3 and 1.", []int{2, 0}, "", true},
		{"**1**", []int{0}, "", true},
		{"9, 2", []int{1}, "", true}, // 9 is out of range
		{"NONE\nquery: raft election timeout", nil, "raft election timeout", true},
		{"NONE\n\nA different search query that might find the answer could be: \"Ragout enterprise license cost\"", nil, "Ragout enterprise license cost", true},
		{"<think>hmm</think>\nNONE\n1. \"paxos phases\"", nil, "paxos phases", true},
		{"NONE\nWhich passages contain information that helps?", nil, "", true},
		{"NONE", nil, "", true},
		{"The answer is in the docs.", nil, "", false},
		{"9", nil, "", false},
	} {
		rel, m, ok := parseGrade(tc.reply, 5)
		if !slices.Equal(rel, tc.relevant) || m != tc.missing || ok != tc.ok {
			t.Errorf("parseGrade(%q) = %v, %q, %v", tc.reply, rel, m, ok)
		}
	}
}

func TestGradeKeepsOrderAndSeesHistory(t *testing.T) {
	llm := &scriptedLLM{replies: []string{"3, 1"}}
	s := &State{Question: "how do I turn it off?", Standalone: "how do I turn it off?",
		History: []ragout.Turn{{Role: "user", Content: "tell me about chunk context"}}}
	for _, id := range []string{"a", "b", "c", "d"} {
		s.Evidence = append(s.Evidence, ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id}})
	}
	if err := Grade(llm).Run(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	var order []string
	for _, c := range s.Evidence {
		order = append(order, c.Chunk.ID)
	}
	if s.Verdict != Sufficient || !slices.Equal(order, []string{"a", "b", "c", "d"}) {
		t.Errorf("verdict %v, evidence order %v; want sufficient, retrieval order kept", s.Verdict, order)
	}
	if !strings.Contains(llm.prompts[0], "Conversation so far:\nuser: tell me about chunk context") {
		t.Errorf("a follow-up must be graded with its history:\n%s", llm.prompts[0])
	}
}

// --- Flow mechanics ------------------------------------------------------

func mark(name string) Step {
	return StepFunc(func(_ context.Context, s *State) error {
		s.Values["path"] = append(s.Values["path"].([]string), name)
		return nil
	})
}

func run(t *testing.T, f *Flow) []string {
	t.Helper()
	s, err := f.Run(context.Background(), "q", WithValue("path", []string{}))
	if err != nil {
		t.Fatal(err)
	}
	return s.Values["path"].([]string)
}

func TestFlowEditing(t *testing.T) {
	f := New().Add("a", mark("a")).Add("b", mark("b")).Add("c", mark("c"))
	if got := run(t, f); !slices.Equal(got, []string{"a", "b", "c"}) {
		t.Errorf("default order = %v", got)
	}

	g := f.Clone()
	if err := g.InsertAfter("a", "x", mark("x")); err != nil {
		t.Fatal(err)
	}
	if err := g.Replace("c", mark("C")); err != nil {
		t.Fatal(err)
	}
	if err := g.Remove("b"); err != nil {
		t.Fatal(err)
	}
	if got := run(t, g); !slices.Equal(got, []string{"a", "x", "C"}) {
		t.Errorf("edited = %v", got)
	}
	if got := run(t, f); !slices.Equal(got, []string{"a", "b", "c"}) {
		t.Errorf("Clone must leave the original alone, got %v", got)
	}
	if g.InsertBefore("missing", "y", mark("y")) == nil || g.Remove("missing") == nil || g.Replace("missing", mark("z")) == nil || g.InsertAfter("a", "x", mark("x")) == nil {
		t.Error("editing unknown or duplicate steps must fail")
	}
}

func TestFlowRoutingAndLimits(t *testing.T) {
	// A loop that exits through a router.
	f := New().
		Add("count", StepFunc(func(_ context.Context, s *State) error { s.Hops++; return nil })).
		Add("never", mark("never")).
		Route("count", func(s *State) string {
			if s.Hops < 3 {
				return "count"
			}
			return End
		})
	s, err := f.Run(context.Background(), "q")
	if err != nil || s.Hops != 3 {
		t.Fatalf("hops = %d, err = %v", s.Hops, err)
	}

	// A loop with no exit hits MaxSteps.
	f = New().Add("a", mark("a")).Edge("a", "a").MaxSteps(5)
	if _, err := f.Run(context.Background(), "q", WithValue("path", []string{})); !errors.Is(err, ErrMaxSteps) {
		t.Errorf("err = %v, want ErrMaxSteps", err)
	}

	// An edge to a removed step fails clearly.
	f = New().Add("a", mark("a")).Add("b", mark("b")).Edge("a", "b")
	f.Remove("b")
	if _, err := f.Run(context.Background(), "q", WithValue("path", []string{})); err == nil || !strings.Contains(err.Error(), `"b"`) {
		t.Errorf("err = %v", err)
	}

	// A failing step stops the run, and the state so far is returned.
	boom := errors.New("boom")
	f = New().Add("a", mark("a")).Add("fail", StepFunc(func(context.Context, *State) error { return boom })).Add("c", mark("c"))
	s, err = f.Run(context.Background(), "q", WithValue("path", []string{}))
	if !errors.Is(err, boom) || !slices.Equal(s.Values["path"].([]string), []string{"a"}) || len(s.Trace) != 2 {
		t.Errorf("err = %v, state = %+v", err, s)
	}
}

func TestCustomizingCRAG(t *testing.T) {
	r := &fakeRetriever{index: map[string][]string{"q": {"a", "b", "c"}}}
	f := CRAG(r, &scriptedLLM{replies: []string{"1"}}, &echoGenerator{})

	// Add a step of your own, and drop grading entirely for a plain pipeline.
	keepTwo := StepFunc(func(_ context.Context, s *State) error {
		s.Evidence = s.Evidence[:min(2, len(s.Evidence))]
		return nil
	})
	if err := f.InsertAfter(StepRetrieve, "keep-two", keepTwo); err != nil {
		t.Fatal(err)
	}
	f.Edge("keep-two", StepGenerate)
	s, err := f.Run(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	if got := steps(s); !slices.Equal(got, []string{"retrieve", "keep-two", "generate"}) || s.Answer.Text != "a b " {
		t.Errorf("steps = %v, answer = %q", got, s.Answer.Text)
	}
}
