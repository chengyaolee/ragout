package eval_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/eval"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

func TestRun(t *testing.T) {
	ctx := context.Background()
	ch, err := chunker.NewCharacterChunker(60, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	// No generator: evaluation only retrieves.
	e, err := ragout.NewEngine(
		ragout.WithReader(reader.NewTextReader(1<<20)),
		ragout.WithChunker(ch),
		ragout.WithIndexStore(store.NewBM25Index()),
	)
	if err != nil {
		t.Fatal(err)
	}
	docs := map[string]string{
		"raft.txt":  "Raft elects a leader for each term.\n\nFollowers replicate the leader log.",
		"paxos.txt": "Paxos reaches consensus with proposers and acceptors.",
	}
	for name, text := range docs {
		if err := e.Ingest(ctx, name, strings.NewReader(text), nil); err != nil {
			t.Fatal(err)
		}
	}

	cases := []eval.Case{
		{Question: "raft leader", Expected: []string{"Raft elects a leader", "replicate the leader log"}},
		{Question: "paxos acceptors", Expected: []string{"proposers and acceptors"}},
		{Question: "byzantine fault tolerance", Expected: []string{"PBFT"}},
	}
	r, err := eval.Run(ctx, e, cases, 2)
	if err != nil {
		t.Fatal(err)
	}
	if r.Cases != 3 || math.Abs(r.HitRate-2.0/3) > 1e-9 || math.Abs(r.Recall-2.0/3) > 1e-9 {
		t.Errorf("report = %v, want hit rate and recall of 2/3", r)
	}
	if r.MRR <= 0.5 || r.MRR > 2.0/3+1e-9 {
		t.Errorf("MRR = %v", r.MRR)
	}
	if len(r.Misses) != 1 || r.Misses[0] != "byzantine fault tolerance" {
		t.Errorf("misses = %q", r.Misses)
	}
}

func TestRunWithSourcesCategoriesAndNDCG(t *testing.T) {
	chunk := func(src, text string) ragout.ScoredChunk {
		return ragout.ScoredChunk{Chunk: ragout.Chunk{Content: text, Metadata: map[string]any{ragout.MetadataSource: src}}}
	}
	ranked := map[string][]ragout.ScoredChunk{
		"q1": {chunk("d9", "x"), chunk("d1", "y"), chunk("d1", "y again"), chunk("d2", "z")},
		"q2": {chunk("d5", "the answer")},
	}
	retrieve := func(_ context.Context, c eval.Case) ([]ragout.ScoredChunk, error) { return ranked[c.Question], nil }
	cases := []eval.Case{
		{Question: "q1", Category: "docs", ExpectedSources: []string{"d1", "d2"}},
		{Question: "q2", Category: "text", Expected: []string{"answer"}},
		{Question: "unanswerable", Category: "none"},
	}
	r, err := eval.RunWith(context.Background(), retrieve, cases, 4)
	if err != nil {
		t.Fatal(err)
	}
	if r.Cases != 2 || r.Skipped != 1 || r.HitRate != 1 || r.Recall != 1 {
		t.Fatalf("report = %+v", r)
	}
	// q1: relevant at ranks 2 and 4 (the repeat of d1 at rank 3 earns nothing).
	wantNDCG1 := (1/math.Log2(3) + 1/math.Log2(5)) / (1 + 1/math.Log2(3))
	if got := r.ByCategory["docs"].NDCG; math.Abs(got-wantNDCG1) > 1e-9 {
		t.Errorf("ndcg = %v, want %v", got, wantNDCG1)
	}
	if r.ByCategory["docs"].MRR != 0.5 || r.ByCategory["text"].MRR != 1 {
		t.Errorf("by category = %+v", r.ByCategory)
	}
	if got := r.Categories(); len(got) != 2 || got[0] != "docs" {
		t.Errorf("categories = %v", got)
	}
}
