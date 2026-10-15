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
