package ragout_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
)

func sourceEngine(t *testing.T, replace bool) (*ragout.Engine, *store.VectorStore, *store.BM25Index) {
	t.Helper()
	emb, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	vs, idx := store.NewVectorStore(), store.NewBM25Index()
	opts := []ragout.EngineOption{
		ragout.WithReader(reader.NewByExtension(1 << 20)),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithEmbedder(emb),
		ragout.WithVectorStore(vs),
		ragout.WithIndexStore(idx),
	}
	if replace {
		opts = append(opts, ragout.WithReplaceSource())
	}
	e, err := ragout.NewEngine(opts...)
	if err != nil {
		t.Fatal(err)
	}
	return e, vs, idx
}

func hits(t *testing.T, idx *store.BM25Index, term string) int {
	t.Helper()
	got, err := idx.SearchSparse(context.Background(), term, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	return len(got)
}

func TestReplaceSourceRemovesStaleChunks(t *testing.T) {
	ctx := context.Background()
	v1 := "# Pricing\nThe plan costs ten dollars.\n# Support\nEmail only.\n"
	v2 := "# Pricing\nThe plan costs twelve dollars.\n# Support\nEmail only.\n"

	for _, replace := range []bool{false, true} {
		e, _, idx := sourceEngine(t, replace)
		for _, doc := range []string{v1, v2} {
			if err := e.Ingest(ctx, "plans.md", strings.NewReader(doc), nil); err != nil {
				t.Fatal(err)
			}
		}
		// Stale text: only gone when replacing. Unchanged sections are rewritten, not lost.
		if got, want := hits(t, idx, "ten"), map[bool]int{false: 1, true: 0}[replace]; got != want {
			t.Errorf("replace=%v: stale chunk hits = %d, want %d", replace, got, want)
		}
		if hits(t, idx, "twelve") != 1 || hits(t, idx, "email") != 1 {
			t.Errorf("replace=%v: current chunks missing", replace)
		}
	}
}

func TestDeleteSource(t *testing.T) {
	ctx := context.Background()
	e, vs, idx := sourceEngine(t, false)
	for name, text := range map[string]string{"a.txt": "alpha text", "b.txt": "beta text"} {
		if err := e.Ingest(ctx, name, strings.NewReader(text), nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.DeleteSource(ctx, "a.txt"); err != nil {
		t.Fatal(err)
	}
	if hits(t, idx, "alpha") != 0 || hits(t, idx, "beta") != 1 {
		t.Errorf("BM25 should hold only b.txt")
	}
	if got, _ := vs.SearchDense(ctx, make([]float32, 8), 10, nil); len(got) != 1 {
		t.Errorf("vector store has %d chunks, want 1", len(got))
	}
	if err := e.DeleteSource(ctx, ""); !errors.Is(err, ragout.ErrNoSource) {
		t.Errorf("empty source: %v", err)
	}
}

type noDeleteIndex struct{ ragout.IndexStore }

func TestReplaceSourceNeedsSupportingStores(t *testing.T) {
	_, err := ragout.NewEngine(
		ragout.WithIndexStore(noDeleteIndex{store.NewBM25Index()}),
		ragout.WithReplaceSource(),
	)
	if !errors.Is(err, ragout.ErrDeleteUnsupported) {
		t.Fatalf("NewEngine = %v, want ErrDeleteUnsupported", err)
	}
}
