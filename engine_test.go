package ragout_test

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/generator"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/reranker"
	"github.com/chengyaolee/ragout/store"
)

const (
	hybridFacts = "Channels prevent race conditions by transferring ownership of values between goroutines."
	answerText  = "Channels transfer ownership. "
)

func TestEngine_EndToEndHybrid(t *testing.T) {
	ctx := context.Background()
	vs := store.NewVectorStore()
	idx := store.NewBM25Index()
	emb, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := chunker.NewCharacterChunker(200, 0, nil)
	if err != nil {
		t.Fatal(err)
	}

	e, err := ragout.NewEngine(
		ragout.WithReader(reader.NewTextReader(1<<20)),
		ragout.WithChunker(ch),
		ragout.WithEmbedder(emb),
		ragout.WithVectorStore(vs),
		ragout.WithIndexStore(idx),
		ragout.WithReranker(reranker.NewMockReranker()),
		ragout.WithGenerator(generator.NewMockGenerator([]string{answerText})),
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := e.Ingest(ctx, "hybrid.txt", strings.NewReader(hybridFacts), nil); err != nil {
		t.Fatal(err)
	}

	answer, err := e.Query(ctx, "How do channels prevent race conditions?")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(answer.Text, answerText) {
		t.Fatalf("answer = %q, want synthesized text %q", answer.Text, answerText)
	}

	hits, err := idx.SearchSparse(ctx, "channels", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("ingest stored no chunks")
	}
	if !strings.Contains(answer.Text, hits[0].Chunk.ID) {
		t.Fatalf("answer = %q, want ingested chunk %s", answer.Text, hits[0].Chunk.ID)
	}

	// Sources must point back at the chunks actually placed in the prompt, and carry
	// the filename Ingest recorded.
	if len(answer.Sources) == 0 {
		t.Fatal("answer has no sources")
	}
	for _, src := range answer.Sources {
		if src.Chunk.Metadata["source"] != "hybrid.txt" {
			t.Fatalf("source chunk metadata[source] = %v, want %q", src.Chunk.Metadata["source"], "hybrid.txt")
		}
	}
	if len(answer.Retrieved) < len(answer.Sources) {
		t.Fatalf("retrieved %d chunks, fewer than %d sources", len(answer.Retrieved), len(answer.Sources))
	}
}

func TestEngine_GracefulDegradation(t *testing.T) {
	t.Run("dense only", func(t *testing.T) {
		ctx := context.Background()
		e, emb, vs, _ := testEngine(t, true, false, false)
		if err := e.Ingest(ctx, "hybrid.txt", strings.NewReader(hybridFacts), nil); err != nil {
			t.Fatal(err)
		}
		answer, err := e.Query(ctx, "How do channels prevent race conditions?")
		if err != nil {
			t.Fatal(err)
		}
		vecs, err := emb.EmbedBatch(ctx, []string{hybridFacts})
		if err != nil {
			t.Fatal(err)
		}
		hits, err := vs.SearchDense(ctx, vecs[0], 5, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(hits) == 0 || !strings.Contains(answer.Text, hits[0].Chunk.ID) {
			t.Fatalf("dense answer = %q, hits = %v", answer.Text, hits)
		}
	})

	t.Run("sparse only", func(t *testing.T) {
		ctx := context.Background()
		e, _, _, idx := testEngine(t, false, true, false)
		if err := e.Ingest(ctx, "hybrid.txt", strings.NewReader(hybridFacts), nil); err != nil {
			t.Fatal(err)
		}
		answer, err := e.Query(ctx, "How do channels prevent race conditions?")
		if err != nil {
			t.Fatal(err)
		}
		hits, err := idx.SearchSparse(ctx, "channels", 5, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(hits) == 0 || !strings.Contains(answer.Text, hits[0].Chunk.ID) {
			t.Fatalf("sparse answer = %q, hits = %v", answer.Text, hits)
		}
	})

	t.Run("no reranker", func(t *testing.T) {
		ctx := context.Background()
		const (
			topK = 50
			topN = 5
			rrfK = 60
		)
		facts := "Channels prevent race conditions via ownership.\n\nMutexes guard shared memory across goroutines."
		emb, err := embedder.NewMockEmbedder(8, 0)
		if err != nil {
			t.Fatal(err)
		}
		ch, err := chunker.NewCharacterChunker(64, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		vs := store.NewVectorStore()
		idx := store.NewBM25Index()
		e, err := ragout.NewEngine(
			ragout.WithReader(reader.NewTextReader(1<<20)),
			ragout.WithChunker(ch),
			ragout.WithEmbedder(emb),
			ragout.WithVectorStore(vs),
			ragout.WithIndexStore(idx),
			ragout.WithGenerator(generator.NewMockGenerator([]string{answerText})),
			ragout.WithTopKRecall(topK),
			ragout.WithTopNRerank(topN),
			ragout.WithRRFK(rrfK),
		)
		if err != nil {
			t.Fatal(err)
		}
		if err := e.Ingest(ctx, "facts.txt", strings.NewReader(facts), nil); err != nil {
			t.Fatal(err)
		}

		query := "How do channels prevent race conditions?"
		answer, err := e.Query(ctx, query)
		if err != nil {
			t.Fatal(err)
		}

		vecs, err := emb.EmbedBatch(ctx, []string{query})
		if err != nil {
			t.Fatal(err)
		}
		dense, err := vs.SearchDense(ctx, vecs[0], topK, nil)
		if err != nil {
			t.Fatal(err)
		}
		sparse, err := idx.SearchSparse(ctx, query, topK, nil)
		if err != nil {
			t.Fatal(err)
		}
		fused := ragout.ReciprocalRankFusion(dense, sparse, topK, rrfK)
		if len(fused) > topN {
			fused = fused[:topN]
		}
		rest := strings.TrimPrefix(answer.Text, answerText)
		for _, hit := range fused {
			if !strings.HasPrefix(rest, hit.Chunk.ID) {
				t.Fatalf("answer = %q, want RRF id %s next", answer.Text, hit.Chunk.ID)
			}
			rest = rest[len(hit.Chunk.ID):]
		}
		if rest != "" {
			t.Fatalf("answer has extra ids %q", rest)
		}
	})
}

func TestEngine_StreamIterBreak(t *testing.T) {
	ctx := context.Background()
	e, _, _, _ := testEngine(t, true, true, true)
	doc := "This query explains how channels prevent race conditions."
	if err := e.Ingest(ctx, "doc.txt", strings.NewReader(doc), nil); err != nil {
		t.Fatal(err)
	}

	before := settledGoroutines()
	n := 0
	_, tokens := e.QueryIter(ctx, "query")
	for token, err := range tokens {
		if err != nil {
			t.Fatal(err)
		}
		if token == "" {
			t.Fatal("empty token")
		}
		n++
		if n == 2 {
			break
		}
	}
	if n != 2 {
		t.Fatalf("received %d tokens, want 2", n)
	}
	if after := settledGoroutines(); after > before {
		t.Fatalf("goroutines before %d after %d", before, after)
	}
}

func TestEngine_ContextCancellation(t *testing.T) {
	t.Run("ingest", func(t *testing.T) {
		e, _, _, _ := testEngine(t, true, true, true)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := e.Ingest(ctx, "hybrid.txt", strings.NewReader(hybridFacts), nil)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Ingest err = %v, want context.Canceled", err)
		}
		_, err = e.Query(context.Background(), "channels")
		if !errors.Is(err, ragout.ErrNoResults) {
			t.Fatalf("query after canceled ingest = %v, want ErrNoResults", err)
		}
	})

	t.Run("query", func(t *testing.T) {
		e, _, _, _ := testEngine(t, true, true, true)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := e.Query(ctx, "How do channels prevent race conditions?")
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Query err = %v, want context.Canceled", err)
		}
	})
}

func TestEngine_ValidationErrors(t *testing.T) {
	if _, err := ragout.NewEngine(); !errors.Is(err, ragout.ErrNoStores) {
		t.Fatalf("NewEngine err = %v, want ErrNoStores", err)
	}

	e, _, _, _ := testEngine(t, false, true, false)
	_, err := e.Query(context.Background(), " \t\n ")
	if !errors.Is(err, ragout.ErrEmptyQuery) {
		t.Fatalf("Query err = %v, want ErrEmptyQuery", err)
	}
}

// flakyIndexStore wraps a BM25Index but can be told to fail Index calls, to exercise
// Ingest's rollback path.
type flakyIndexStore struct {
	inner     *store.BM25Index
	failIndex bool
}

func (f *flakyIndexStore) Index(ctx context.Context, chunks []ragout.Chunk) error {
	if f.failIndex {
		return errors.New("flakyIndexStore: boom")
	}
	return f.inner.Index(ctx, chunks)
}

func (f *flakyIndexStore) SearchSparse(ctx context.Context, query string, topK int, filter map[string]any) ([]ragout.ScoredChunk, error) {
	return f.inner.SearchSparse(ctx, query, topK, filter)
}

func (f *flakyIndexStore) Delete(ctx context.Context, ids []string) error {
	return f.inner.Delete(ctx, ids)
}

func TestEngine_IngestRollback(t *testing.T) {
	ctx := context.Background()
	vs := store.NewVectorStore()
	fIdx := &flakyIndexStore{inner: store.NewBM25Index(), failIndex: true}
	emb, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := chunker.NewCharacterChunker(200, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	e, err := ragout.NewEngine(
		ragout.WithReader(reader.NewTextReader(1<<20)),
		ragout.WithChunker(ch),
		ragout.WithEmbedder(emb),
		ragout.WithVectorStore(vs),
		ragout.WithIndexStore(fIdx),
		ragout.WithGenerator(generator.NewMockGenerator(nil)),
	)
	if err != nil {
		t.Fatal(err)
	}

	zeroVec := make([]float32, 8)
	countVectorChunks := func() int {
		hits, err := vs.SearchDense(ctx, zeroVec, 100, nil)
		if err != nil {
			t.Fatal(err)
		}
		return len(hits)
	}

	// IndexStore.Index fails after VectorStore.Upsert has already committed; the
	// vector store must be rolled back rather than left half-indexed.
	if err := e.Ingest(ctx, "doc.txt", strings.NewReader(hybridFacts), nil); err == nil {
		t.Fatal("expected Ingest to fail")
	}
	if n := countVectorChunks(); n != 0 {
		t.Fatalf("vector store has %d chunks after failed ingest, want 0 (rollback)", n)
	}

	// A retry with a healthy store succeeds, using the same deterministic chunk IDs.
	fIdx.failIndex = false
	if err := e.Ingest(ctx, "doc.txt", strings.NewReader(hybridFacts), nil); err != nil {
		t.Fatalf("retry Ingest failed: %v", err)
	}
	first := countVectorChunks()
	if first == 0 {
		t.Fatal("retry Ingest stored no chunks")
	}

	// Re-ingesting the same source+content must not duplicate chunks.
	if err := e.Ingest(ctx, "doc.txt", strings.NewReader(hybridFacts), nil); err != nil {
		t.Fatalf("re-ingest failed: %v", err)
	}
	if again := countVectorChunks(); again != first {
		t.Fatalf("re-ingest changed chunk count from %d to %d, want stable (idempotent IDs)", first, again)
	}
}

// testEngine builds a pipeline with the requested stores. Dense search includes the vector store; sparse search includes the BM25 index.
func testEngine(t *testing.T, dense, sparse, rerank bool) (*ragout.Engine, *embedder.MockEmbedder, *store.VectorStore, *store.BM25Index) {
	t.Helper()
	emb, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	ch, err := chunker.NewCharacterChunker(200, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	vs := store.NewVectorStore()
	idx := store.NewBM25Index()

	opts := []ragout.EngineOption{
		ragout.WithReader(reader.NewTextReader(1 << 20)),
		ragout.WithChunker(ch),
		ragout.WithGenerator(generator.NewMockGenerator([]string{"tok-a", "tok-b", "tok-c", "tok-d"})),
	}
	if dense {
		opts = append(opts, ragout.WithEmbedder(emb), ragout.WithVectorStore(vs))
	}
	if sparse {
		opts = append(opts, ragout.WithIndexStore(idx))
	}
	if rerank {
		opts = append(opts, ragout.WithReranker(reranker.NewMockReranker()))
	}
	e, err := ragout.NewEngine(opts...)
	if err != nil {
		t.Fatal(err)
	}
	return e, emb, vs, idx
}

func settledGoroutines() int {
	runtime.GC()
	var n int
	for range 5 {
		time.Sleep(10 * time.Millisecond)
		runtime.Gosched()
		n = runtime.NumGoroutine()
	}
	return n
}

func TestEngine_QueryWithFilter(t *testing.T) {
	ctx := context.Background()
	e, _, _, _ := testEngine(t, true, true, false)

	// Ingest two documents with different metadata
	docEng := "Engineering teams use distributed consensus algorithms like Raft."
	docMkt := "Marketing teams focus on customer acquisition campaigns and branding."

	if err := e.Ingest(ctx, "eng.txt", strings.NewReader(docEng), map[string]any{"department": "engineering"}); err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(ctx, "mkt.txt", strings.NewReader(docMkt), map[string]any{"department": "marketing"}); err != nil {
		t.Fatal(err)
	}

	// Query with engineering filter
	gotEng, err := e.Query(ctx, "teams", ragout.WithQueryFilter(map[string]any{"department": "engineering"}))
	if err != nil {
		t.Fatalf("Query with engineering filter failed: %v", err)
	}
	if strings.Contains(gotEng.Text, "Marketing") {
		t.Fatalf("expected only engineering results, got %s", gotEng.Text)
	}

	// Query with marketing filter
	gotMkt, err := e.Query(ctx, "teams", ragout.WithQueryFilter(map[string]any{"department": "marketing"}))
	if err != nil {
		t.Fatalf("Query with marketing filter failed: %v", err)
	}
	if strings.Contains(gotMkt.Text, "Engineering") {
		t.Fatalf("expected only marketing results, got %s", gotMkt.Text)
	}
}
