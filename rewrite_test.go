package ragout_test

import (
	"context"
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/store"
	dto "github.com/prometheus/client_model/go"
)

type fakeTransformer struct {
	rw      ragout.Rewrite
	err     error
	history []ragout.Turn
}

func (f *fakeTransformer) Transform(_ context.Context, _ string, h []ragout.Turn) (ragout.Rewrite, error) {
	f.history = h
	return f.rw, f.err
}

type countingEmbedder struct {
	ragout.Embedder
	mu    sync.Mutex
	calls [][]string
}

func (c *countingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	c.mu.Lock()
	c.calls = append(c.calls, texts)
	c.mu.Unlock()
	return c.Embedder.EmbedBatch(ctx, texts)
}

type recordingIndex struct {
	*store.BM25Index
	mu      sync.Mutex
	queries []string
}

func (r *recordingIndex) SearchSparse(ctx context.Context, q string, k int, f map[string]any) ([]ragout.ScoredChunk, error) {
	r.mu.Lock()
	r.queries = append(r.queries, q)
	r.mu.Unlock()
	return r.BM25Index.SearchSparse(ctx, q, k, f)
}

type questionRecorder struct{ reranked, generated string }

func (q *questionRecorder) Rerank(_ context.Context, query string, c []ragout.ScoredChunk, n int) ([]ragout.ScoredChunk, error) {
	q.reranked = query
	return c[:min(n, len(c))], nil
}

func (q *questionRecorder) GenerateIter(_ context.Context, query string, c []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	q.generated = query
	return c, func(yield func(string, error) bool) { yield("ok", nil) }
}

func rewriteEngine(t *testing.T, tr ragout.QueryTransformer) (*ragout.Engine, *countingEmbedder, *recordingIndex, *questionRecorder) {
	t.Helper()
	mock, err := embedder.NewMockEmbedder(8, 0)
	if err != nil {
		t.Fatal(err)
	}
	emb := &countingEmbedder{Embedder: mock}
	idx := &recordingIndex{BM25Index: store.NewBM25Index()}
	rec := &questionRecorder{}
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithEmbedder(emb),
		ragout.WithVectorStore(store.NewVectorStore()),
		ragout.WithIndexStore(idx),
		ragout.WithReranker(rec),
		ragout.WithGenerator(rec),
		ragout.WithQueryTransformer(tr),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(context.Background(), "a.txt", strings.NewReader("Raft elects a leader to agree on a log."), nil); err != nil {
		t.Fatal(err)
	}
	emb.calls, idx.queries = nil, nil
	return e, emb, idx, rec
}

func TestQueryRewrite_RoutesSearchesAndUsesStandaloneQuestion(t *testing.T) {
	tr := &fakeTransformer{rw: ragout.Rewrite{
		Question: "how does raft agree",
		Searches: []ragout.Search{
			{Text: "A passage about Raft.", Target: ragout.SearchDense},
			{Text: "leader election", Target: ragout.SearchSparse},
			{Text: "  How does RAFT agree ", Target: ragout.SearchAll}, // duplicate of the question
			{Text: ""},
		},
	}}
	e, emb, idx, rec := rewriteEngine(t, tr)
	history := []ragout.Turn{{Role: "user", Content: "tell me about raft"}}

	ans, err := e.Query(context.Background(), "and how does it agree?", ragout.WithHistory(history))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.history) != 1 {
		t.Errorf("transformer history = %+v", tr.history)
	}
	if len(emb.calls) != 1 || !slices.Equal(emb.calls[0], []string{"how does raft agree", "A passage about Raft."}) {
		t.Errorf("want one embed call for both dense texts, got %q", emb.calls)
	}
	slices.Sort(idx.queries)
	if !slices.Equal(idx.queries, []string{"how does raft agree", "leader election"}) {
		t.Errorf("BM25 queries = %q (the HyDE passage must not reach BM25)", idx.queries)
	}
	if rec.reranked != "how does raft agree" || rec.generated != "how does raft agree" {
		t.Errorf("reranker got %q, generator got %q", rec.reranked, rec.generated)
	}
	if ans.Question != "how does raft agree" || len(ans.Searches) != 3 {
		t.Errorf("Answer.Question = %q, Searches = %+v", ans.Question, ans.Searches)
	}
}

func TestQueryRewrite_FailsOpen(t *testing.T) {
	tr := &fakeTransformer{err: errors.New("llm down")}
	e, _, idx, rec := rewriteEngine(t, tr)

	before := failures()
	ans, err := e.Query(context.Background(), "raft leader")
	if err != nil {
		t.Fatalf("a transformer error must not fail the query: %v", err)
	}
	if ans.Question != "raft leader" || rec.generated != "raft leader" || len(idx.queries) != 1 {
		t.Errorf("want the original question only, got %q / %q / %q", ans.Question, rec.generated, idx.queries)
	}
	if ans.TransformErr == nil || failures() != before+1 {
		t.Errorf("failure must be reported: TransformErr=%v", ans.TransformErr)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tr.err = context.Canceled
	if _, err := e.Query(ctx, "raft leader"); !errors.Is(err, context.Canceled) {
		t.Errorf("cancellation must still abort, got %v", err)
	}
}

func TestRRFFusesManyLists(t *testing.T) {
	c := func(id string, dense, sparse float64) ragout.ScoredChunk {
		return ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id}, DenseScore: dense, SparseScore: sparse}
	}
	lists := []ragout.RankedList{
		{Dense: true, Hits: []ragout.ScoredChunk{c("a", -0.2, 0), c("b", -0.5, 0)}},
		{Hits: []ragout.ScoredChunk{c("b", 0, 3.1)}},
		{Dense: true, Rewrite: true, Hits: []ragout.ScoredChunk{c("b", 0.4, 0), c("c", 0.1, 0)}},
	}
	got := ragout.RRF(60).Fuse(lists, 2)
	if len(got) != 2 || got[0].Chunk.ID != "b" || got[1].Chunk.ID != "a" {
		t.Fatalf("order = %+v, want b (in all three lists) then a", got)
	}
	if got[0].DenseScore != 0.4 || got[0].SparseScore != 3.1 {
		t.Errorf("b keeps its best scores, got dense %v sparse %v", got[0].DenseScore, got[0].SparseScore)
	}
	if got[1].DenseScore != -0.2 {
		t.Errorf("a negative cosine must survive fusion, got %v", got[1].DenseScore)
	}
}

type slowTransformer struct{}

func (slowTransformer) Transform(ctx context.Context, _ string, _ []ragout.Turn) (ragout.Rewrite, error) {
	<-ctx.Done()
	return ragout.Rewrite{Question: "too late"}, ctx.Err()
}

func TestQueryRewrite_TimeoutFallsBack(t *testing.T) {
	mock, _ := embedder.NewMockEmbedder(8, 0)
	rec := &questionRecorder{}
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithEmbedder(mock),
		ragout.WithVectorStore(store.NewVectorStore()),
		ragout.WithGenerator(rec),
		ragout.WithQueryTransformer(slowTransformer{}),
		ragout.WithTransformTimeout(20*time.Millisecond),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(context.Background(), "a.txt", strings.NewReader("raft leader"), nil); err != nil {
		t.Fatal(err)
	}
	ans, err := e.Query(context.Background(), "raft")
	if err != nil {
		t.Fatalf("timeout must fall back, got %v", err)
	}
	if !errors.Is(ans.TransformErr, context.DeadlineExceeded) || rec.generated == "" {
		t.Errorf("TransformErr = %v, generated %q", ans.TransformErr, rec.generated)
	}
}

type historyRecorder struct {
	questionRecorder
	history []ragout.Turn
	plain   bool
}

func (h *historyRecorder) GenerateIter(ctx context.Context, q string, c []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	h.plain = true
	return h.questionRecorder.GenerateIter(ctx, q, c)
}

func (h *historyRecorder) GenerateIterWithHistory(ctx context.Context, q string, hist []ragout.Turn, c []ragout.ScoredChunk) ([]ragout.ScoredChunk, iter.Seq2[string, error]) {
	h.history = hist
	return h.questionRecorder.GenerateIter(ctx, q, c)
}

func TestHistoryReachesHistoryGenerator(t *testing.T) {
	gen := &historyRecorder{}
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithIndexStore(newBM25ForTest()),
		ragout.WithGenerator(gen),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(context.Background(), "a.txt", strings.NewReader("raft leader"), nil); err != nil {
		t.Fatal(err)
	}

	if _, err := e.Query(context.Background(), "raft"); err != nil || !gen.plain || gen.history != nil {
		t.Fatalf("no history: want GenerateIter, got plain=%v history=%v err=%v", gen.plain, gen.history, err)
	}
	gen.plain = false
	turns := []ragout.Turn{{Role: "user", Content: "hi"}}
	if _, err := e.Query(context.Background(), "raft", ragout.WithHistory(turns)); err != nil || gen.plain || len(gen.history) != 1 {
		t.Fatalf("with history: want GenerateIterWithHistory, got plain=%v history=%v err=%v", gen.plain, gen.history, err)
	}
}

func TestRetrieveNeedsNoGenerator(t *testing.T) {
	e, err := ragout.NewEngine(
		ragout.WithReader(newTextReaderForTest()),
		ragout.WithChunker(newChunkerForTest(t)),
		ragout.WithIndexStore(newBM25ForTest()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Ingest(context.Background(), "a.txt", strings.NewReader("raft leader"), nil); err != nil {
		t.Fatal(err)
	}
	ans, err := e.Retrieve(context.Background(), "raft")
	if err != nil || len(ans.Retrieved) != 1 || ans.Text != "" || ans.Question != "raft" {
		t.Fatalf("Retrieve = %+v, %v", ans, err)
	}
	if _, err := e.Query(context.Background(), "raft"); !errors.Is(err, ragout.ErrNilGenerator) {
		t.Errorf("Query without a generator = %v, want ErrNilGenerator", err)
	}
}

func failures() float64 {
	var m dto.Metric
	_ = ragout.TransformFailuresTotal.Write(&m)
	return m.GetCounter().GetValue()
}

func TestWeightedRRFDiscountsRewrites(t *testing.T) {
	c := func(id string) ragout.ScoredChunk { return ragout.ScoredChunk{Chunk: ragout.Chunk{ID: id}} }
	// The question's dense and BM25 lists both rank "q" first; three rewrites rank "r" first.
	lists := []ragout.RankedList{
		{Dense: true, Hits: []ragout.ScoredChunk{c("q")}},
		{Hits: []ragout.ScoredChunk{c("q")}},
		{Dense: true, Rewrite: true, Hits: []ragout.ScoredChunk{c("r")}},
		{Dense: true, Rewrite: true, Hits: []ragout.ScoredChunk{c("r")}},
		{Dense: true, Rewrite: true, Hits: []ragout.ScoredChunk{c("r")}},
	}
	if got := ragout.RRF(60).Fuse(lists, 1); got[0].Chunk.ID != "r" {
		t.Errorf("at equal weight three rewrites outvote the question, got %s", got[0].Chunk.ID)
	}
	if got := ragout.WeightedRRF(60, 0.5).Fuse(lists, 1); got[0].Chunk.ID != "q" {
		t.Errorf("weighted, the question's top hit should win, got %s", got[0].Chunk.ID)
	}
	// Without rewrites the two are identical.
	plain := lists[:2]
	if a, b := ragout.RRF(60).Fuse(plain, 1), ragout.WeightedRRF(60, 0.5).Fuse(plain, 1); a[0].Score != b[0].Score {
		t.Errorf("no rewrites: %v vs %v", a[0].Score, b[0].Score)
	}
}
