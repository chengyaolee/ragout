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

	if err := e.Ingest(ctx, strings.NewReader(hybridFacts), nil); err != nil {
		t.Fatal(err)
	}

	got, err := e.Query(ctx, "How do channels prevent race conditions?")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, answerText) {
		t.Fatalf("answer = %q, want synthesized text %q", got, answerText)
	}

	hits, err := idx.SearchSparse(ctx, "channels", 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("ingest stored no chunks")
	}
	if !strings.Contains(got, hits[0].Chunk.ID) {
		t.Fatalf("answer = %q, want ingested chunk %s", got, hits[0].Chunk.ID)
	}
}

func TestEngine_GracefulDegradation(t *testing.T) {
	t.Run("dense only", func(t *testing.T) {
		ctx := context.Background()
		e, emb, vs, _ := testEngine(t, true, false, false)
		if err := e.Ingest(ctx, strings.NewReader(hybridFacts), nil); err != nil {
			t.Fatal(err)
		}
		got, err := e.Query(ctx, "How do channels prevent race conditions?")
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
		if len(hits) == 0 || !strings.Contains(got, hits[0].Chunk.ID) {
			t.Fatalf("dense answer = %q, hits = %v", got, hits)
		}
	})

	t.Run("sparse only", func(t *testing.T) {
		ctx := context.Background()
		e, _, _, idx := testEngine(t, false, true, false)
		if err := e.Ingest(ctx, strings.NewReader(hybridFacts), nil); err != nil {
			t.Fatal(err)
		}
		got, err := e.Query(ctx, "How do channels prevent race conditions?")
		if err != nil {
			t.Fatal(err)
		}
		hits, err := idx.SearchSparse(ctx, "channels", 5, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(hits) == 0 || !strings.Contains(got, hits[0].Chunk.ID) {
			t.Fatalf("sparse answer = %q, hits = %v", got, hits)
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
		if err := e.Ingest(ctx, strings.NewReader(facts), nil); err != nil {
			t.Fatal(err)
		}

		query := "How do channels prevent race conditions?"
		got, err := e.Query(ctx, query)
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
		rest := strings.TrimPrefix(got, answerText)
		for _, hit := range fused {
			if !strings.HasPrefix(rest, hit.Chunk.ID) {
				t.Fatalf("answer = %q, want RRF id %s next", got, hit.Chunk.ID)
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
	if err := e.Ingest(ctx, strings.NewReader(doc), nil); err != nil {
		t.Fatal(err)
	}

	before := settledGoroutines()
	n := 0
	for token, err := range e.QueryIter(ctx, "query") {
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
		err := e.Ingest(ctx, strings.NewReader(hybridFacts), nil)
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
