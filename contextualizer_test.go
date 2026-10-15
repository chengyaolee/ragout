package ragout_test

import (
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

type fixedContext string

func (f fixedContext) Contextualize(_ context.Context, _ ragout.Document, chunks []ragout.Chunk) ([]string, error) {
	out := make([]string, len(chunks))
	for i := range out {
		out[i] = string(f)
	}
	return out, nil
}

// Context must make a chunk findable by BM25 while Content (what gets cited) is untouched.
func TestEngine_ContextualizerIndexesContextNotContent(t *testing.T) {
	ctx := context.Background()
	idx := newBM25ForTest()
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithIndexStore(idx),
		ragout.WithContextualizer(fixedContext("Zorblax annual report")),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(ctx, "r.txt", strings.NewReader("Revenue grew 24 percent."), nil); err != nil {
		t.Fatal(err)
	}
	got, err := idx.SearchSparse(ctx, "zorblax", 5, nil)
	if err != nil || len(got) != 1 {
		t.Fatalf("SearchSparse(zorblax) = %v, %v; want 1 hit", got, err)
	}
	if c := got[0].Chunk.Content; c != "Revenue grew 24 percent." {
		t.Errorf("Content = %q, want the original text", c)
	}
}

func newBM25ForTest() *store.BM25Index { return store.NewBM25Index() }

func newTextReaderForTest() ragout.Reader { return reader.NewTextReader(1 << 20) }

func newChunkerForTest(t *testing.T) ragout.Chunker {
	t.Helper()
	ch, err := chunker.NewCharacterChunker(200, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	return ch
}
