package ragout

import (
	"context"
	"io"
	"iter"
	"os"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// spanRec is installed once. The global tracer delegates only to the first provider.
var spanRec = tracetest.NewSpanRecorder()

func TestMain(m *testing.M) {
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(spanRec)))
	os.Exit(m.Run())
}

func TestQuerySpanTreeAndStageMetrics(t *testing.T) {
	spanRec.Reset()

	before := map[string]uint64{}
	for _, stage := range []string{"embed", "dense", "sparse", "fusion", "rerank", "generation"} {
		before[stage] = histogramCount(t, stage)
	}

	e := testEngine(t)
	got, err := e.Query(context.Background(), "channels")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if got != "channels serialize access" {
		t.Fatalf("answer = %q", got)
	}

	spans := map[string]sdktrace.ReadOnlySpan{}
	for _, sp := range spanRec.Ended() {
		spans[sp.Name()] = sp
	}
	root := spans["ragout.Engine.Query"]
	if root == nil {
		t.Fatalf("missing root span, got %v", spanNames(spans))
	}
	if root.Status().Code != codes.Unset {
		t.Fatalf("root status = %s", root.Status().Code)
	}
	assertAttr(t, root, "ragout.query", "channels")

	parentOf := map[string]string{
		"query.embed":         "ragout.Engine.Query",
		"query.retrieval":     "ragout.Engine.Query",
		"query.dense_search":  "query.retrieval",
		"query.sparse_search": "query.retrieval",
		"query.rrf_fusion":    "ragout.Engine.Query",
		"query.reranker":      "ragout.Engine.Query",
		"query.generation":    "ragout.Engine.Query",
	}
	for name, parent := range parentOf {
		sp := spans[name]
		if sp == nil {
			t.Fatalf("missing span %s", name)
		}
		p := spans[parent]
		if sp.Parent().SpanID() != p.SpanContext().SpanID() {
			t.Fatalf("%s parent = %s, want %s", name, sp.Parent().SpanID(), parent)
		}
	}
	assertAttr(t, spans["query.embed"], "text.len", len("channels"))
	assertAttr(t, spans["query.dense_search"], "top_k", 50)
	assertAttr(t, spans["query.dense_search"], "candidates_found", 1)
	assertAttr(t, spans["query.sparse_search"], "terms_matched", 2)
	assertAttr(t, spans["query.sparse_search"], "candidates_found", 1)
	assertAttr(t, spans["query.rrf_fusion"], "rrf_k", 60)
	assertAttr(t, spans["query.rrf_fusion"], "fused_count", 1)
	assertAttr(t, spans["query.reranker"], "top_n", 5)
	if v, ok := attrString(spans["query.reranker"], "reranker_type"); !ok || !strings.Contains(v, "echoReranker") {
		t.Fatalf("reranker_type = %q", v)
	}
	if v, ok := attrString(spans["query.embed"], "model"); !ok || v == "" {
		t.Fatalf("model = %q", v)
	}
	if _, ok := attrFloat(spans["query.generation"], "first_token_latency_ms"); !ok {
		t.Fatal("missing first_token_latency_ms")
	}
	if n, ok := attrInt(spans["query.generation"], "prompt_tokens"); !ok || n <= 0 {
		t.Fatalf("prompt_tokens = %d", n)
	}

	for stage, was := range before {
		got := histogramCount(t, stage)
		if got <= was {
			t.Fatalf("stage %s samples = %d, want increase from %d", stage, got, was)
		}
	}
}

func TestQuerySpanRecordsError(t *testing.T) {
	spanRec.Reset()

	e := testEngine(t)
	_, err := e.Query(context.Background(), "   ")
	if err != ErrEmptyQuery {
		t.Fatalf("err = %v", err)
	}
	var root sdktrace.ReadOnlySpan
	for _, sp := range spanRec.Ended() {
		if sp.Name() == "ragout.Engine.Query" {
			root = sp
		}
	}
	if root == nil {
		t.Fatal("missing root span")
	}
	if root.Status().Code != codes.Error {
		t.Fatalf("status = %s", root.Status().Code)
	}
}

func TestIngestCountsChunks(t *testing.T) {
	before := counterValue(t)
	e := testEngine(t)
	if err := e.Ingest(context.Background(), strings.NewReader("ignored"), nil); err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	if got := counterValue(t) - before; got != 2 {
		t.Fatalf("ingested delta = %v, want 2", got)
	}
}

func BenchmarkEngine_Query(b *testing.B) {
	e := testEngine(b)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := e.Query(ctx, "channels"); err != nil {
			b.Fatal(err)
		}
	}
}

type testTB interface {
	Helper()
	Fatal(args ...any)
}

func testEngine(tb testTB) *Engine {
	tb.Helper()
	e, err := NewEngine(
		WithReader(staticReader{}),
		WithChunker(staticChunker{}),
		WithEmbedder(staticEmbedder{}),
		WithVectorStore(staticDense{}),
		WithIndexStore(staticSparse{terms: 2}),
		WithReranker(echoReranker{}),
		WithGenerator(echoGenerator{}),
	)
	if err != nil {
		tb.Fatal(err)
	}
	return e
}

type staticReader struct{}

func (staticReader) Read(context.Context, io.Reader, map[string]any) ([]Document, error) {
	return []Document{{ID: "d", Content: "channels serialize access"}}, nil
}

type staticChunker struct{}

func (staticChunker) Chunk(context.Context, Document) ([]Chunk, error) {
	return []Chunk{
		{ID: "c1", Content: "channels serialize access"},
		{ID: "c2", Content: "mutexes too"},
	}, nil
}

type staticEmbedder struct{}

func (staticEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0}
	}
	return out, nil
}
func (staticEmbedder) Dimension() int { return 2 }

type staticDense struct{}

func (staticDense) Upsert(context.Context, []Chunk) error { return nil }
func (staticDense) SearchDense(context.Context, []float32, int, map[string]any) ([]ScoredChunk, error) {
	return []ScoredChunk{{
		Chunk:      Chunk{ID: "c1", Content: "channels serialize access"},
		DenseScore: 1,
		Score:      1,
	}}, nil
}

type staticSparse struct{ terms int }

func (staticSparse) Index(context.Context, []Chunk) error { return nil }
func (staticSparse) SearchSparse(context.Context, string, int, map[string]any) ([]ScoredChunk, error) {
	return []ScoredChunk{{
		Chunk:       Chunk{ID: "c1", Content: "channels serialize access"},
		SparseScore: 1,
		Score:       1,
	}}, nil
}
func (s staticSparse) TermsMatched(string) int { return s.terms }

type echoReranker struct{}

func (echoReranker) Rerank(context.Context, string, []ScoredChunk, int) ([]ScoredChunk, error) {
	return []ScoredChunk{{
		Chunk: Chunk{ID: "c1", Content: "channels serialize access"},
		Score: 1,
	}}, nil
}

type echoGenerator struct{}

func (echoGenerator) Generate(context.Context, string, []ScoredChunk) (string, error) {
	return "", nil
}
func (echoGenerator) GenerateStream(context.Context, string, []ScoredChunk, StreamCallback) error {
	return nil
}
func (echoGenerator) GenerateIter(_ context.Context, _ string, candidates []ScoredChunk) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if len(candidates) == 0 || candidates[0].Chunk.Content == "" {
			yield("", ErrNoResults)
			return
		}
		yield(candidates[0].Chunk.Content, nil)
	}
}

func histogramCount(t *testing.T, stage string) uint64 {
	t.Helper()
	child := QueryDuration.WithLabelValues(stage)
	metric, ok := child.(prometheus.Metric)
	if !ok {
		t.Fatalf("stage %s observer is not a metric", stage)
	}
	var m dto.Metric
	if err := metric.Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetHistogram().GetSampleCount()
}

func counterValue(t *testing.T) float64 {
	t.Helper()
	var m dto.Metric
	if err := IngestedChunksTotal.Write(&m); err != nil {
		t.Fatal(err)
	}
	return m.GetCounter().GetValue()
}

func assertAttr(t *testing.T, sp sdktrace.ReadOnlySpan, key string, want any) {
	t.Helper()
	switch want := want.(type) {
	case string:
		got, ok := attrString(sp, key)
		if !ok || got != want {
			t.Fatalf("%s = %q, want %q", key, got, want)
		}
	case int:
		got, ok := attrInt(sp, key)
		if !ok || got != want {
			t.Fatalf("%s = %d, want %d", key, got, want)
		}
	default:
		t.Fatalf("unsupported attr type %T", want)
	}
}

func attrString(sp sdktrace.ReadOnlySpan, key string) (string, bool) {
	for _, a := range sp.Attributes() {
		if string(a.Key) == key && a.Value.Type() == attribute.STRING {
			return a.Value.AsString(), true
		}
	}
	return "", false
}

func attrInt(sp sdktrace.ReadOnlySpan, key string) (int, bool) {
	for _, a := range sp.Attributes() {
		if string(a.Key) == key && a.Value.Type() == attribute.INT64 {
			return int(a.Value.AsInt64()), true
		}
	}
	return 0, false
}

func attrFloat(sp sdktrace.ReadOnlySpan, key string) (float64, bool) {
	for _, a := range sp.Attributes() {
		if string(a.Key) == key && a.Value.Type() == attribute.FLOAT64 {
			return a.Value.AsFloat64(), true
		}
	}
	return 0, false
}

func spanNames(spans map[string]sdktrace.ReadOnlySpan) []string {
	names := make([]string, 0, len(spans))
	for name := range spans {
		names = append(names, name)
	}
	return names
}
