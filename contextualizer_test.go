package ragout_test

import (
	"context"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/generator"
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

func TestEngine_DefaultMetadataContext(t *testing.T) {
	ctx := context.Background()
	for _, tc := range []struct {
		name string
		opts []ragout.EngineOption
		hits int
	}{
		{"default", nil, 1},
		{"disabled", []ragout.EngineOption{ragout.WithContextualizer(nil)}, 0},
	} {
		idx := newBM25ForTest()
		opts := append([]ragout.EngineOption{
			ragout.WithReader(newTextReaderForTest()),
			ragout.WithChunker(newChunkerForTest(t)),
			ragout.WithIndexStore(idx),
		}, tc.opts...)
		e, err := ragout.NewEngine(opts...)
		if err != nil {
			t.Fatal(err)
		}
		if err := e.Ingest(ctx, "dir/quarterly-earnings.txt", strings.NewReader("Revenue grew 24 percent."), nil); err != nil {
			t.Fatal(err)
		}
		// The file name only reaches the index through the breadcrumb.
		got, _ := idx.SearchSparse(ctx, "earnings", 5, nil)
		if len(got) != tc.hits {
			t.Errorf("%s: SearchSparse(earnings) = %d hits, want %d", tc.name, len(got), tc.hits)
		}
	}
}

func TestBreadcrumb(t *testing.T) {
	for _, tc := range []struct {
		meta map[string]any
		want string
	}{
		{map[string]any{ragout.MetadataTitle: "Report", ragout.MetadataSection: "A > B", ragout.MetadataSource: "x.md"}, "Report > A > B"},
		{map[string]any{ragout.MetadataSource: "docs/q3-notes.md", ragout.MetadataSection: "Risks"}, "q3-notes > Risks"},
		{map[string]any{ragout.MetadataSection: "Risks"}, "Risks"},
		{nil, ""},
	} {
		if got := ragout.Breadcrumb(tc.meta); got != tc.want {
			t.Errorf("Breadcrumb(%v) = %q, want %q", tc.meta, got, tc.want)
		}
	}
}

type upperExpander struct{}

func (upperExpander) Expand(_ context.Context, hits []ragout.ScoredChunk) ([]ragout.ScoredChunk, error) {
	for i := range hits {
		hits[i].Chunk.Content = strings.ToUpper(hits[i].Chunk.Content)
	}
	return hits, nil
}

func TestEngine_ExpanderRunsBeforeGeneration(t *testing.T) {
	ctx := context.Background()
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithIndexStore(newBM25ForTest()),
		ragout.WithGenerator(generator.NewMockGenerator(nil)),
		ragout.WithExpander(upperExpander{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(ctx, "a.txt", strings.NewReader("raft elects a leader"), nil); err != nil {
		t.Fatal(err)
	}
	ans, err := e.Query(ctx, "raft")
	if err != nil {
		t.Fatal(err)
	}
	if len(ans.Sources) != 1 || ans.Sources[0].Chunk.Content != "RAFT ELECTS A LEADER" {
		t.Fatalf("Sources = %+v, want the expanded chunk", ans.Sources)
	}
}
