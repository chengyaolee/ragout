package flow_test

import (
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/flow"
	"github.com/chengyaolee/ragout/generator"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

var _ flow.Retriever = (*ragout.Engine)(nil)

type replies []string

func (r *replies) Complete(context.Context, string, string) (string, error) {
	out := (*r)[0]
	*r = (*r)[1:]
	return out, nil
}

func TestCRAGWithEngine(t *testing.T) {
	ctx := context.Background()
	ch, err := chunker.NewCharacterChunker(80, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	e, err := ragout.NewEngine(
		ragout.WithReader(reader.NewByExtension(1<<20)),
		ragout.WithChunker(ch),
		ragout.WithIndexStore(store.NewBM25Index()),
	)
	if err != nil {
		t.Fatal(err)
	}
	doc := "# Billing\nThe billing service is owned by the payments team.\n# Hosting\nIt runs in the eu-west region."
	if err := e.Ingest(ctx, "services.md", strings.NewReader(doc), nil); err != nil {
		t.Fatal(err)
	}

	grader := &replies{"NONE\nbilling hosting region", "1"}
	s, err := flow.CRAG(e, grader, generator.NewMockGenerator(nil)).Run(ctx, "who owns billing")
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, sc := range s.Evidence {
		text.WriteString(sc.Chunk.Content)
	}
	if s.Hops != 2 || !strings.Contains(text.String(), "payments team") || !strings.Contains(text.String(), "eu-west") {
		t.Errorf("want both sections after one corrective hop: hops=%d evidence=%q", s.Hops, text.String())
	}
}
