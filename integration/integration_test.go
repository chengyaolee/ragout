// Package integration runs ragout against real services. Every test is skipped unless
// its environment variable is set, so `go test ./...` stays offline:
//
//	RAGOUT_IT_CHROMA_URL         e.g. http://localhost:8000
//	RAGOUT_IT_QDRANT_URL         e.g. http://localhost:6333
//	RAGOUT_IT_OLLAMA_URL         e.g. http://localhost:11434
//	RAGOUT_IT_OLLAMA_MODEL       a chat model pulled into Ollama, e.g. qwen2.5:0.5b
//	RAGOUT_IT_OLLAMA_EMBED_MODEL optional embedding model, e.g. nomic-embed-text
//	RAGOUT_IT_OPENAI_API_KEY     spends a few cents of OpenAI usage
//
// Example: docker run -p 6333:6333 qdrant/qdrant, then
// RAGOUT_IT_QDRANT_URL=http://localhost:6333 go test ./integration -v
package integration_test

import (
	"context"
	"errors"
	"hash/fnv"
	"math"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/contextual"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/expand"
	"github.com/chengyaolee/ragout/generator"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/store"
	"github.com/chengyaolee/ragout/transform"
	"github.com/google/uuid"
)

func env(t *testing.T, name string) string {
	t.Helper()
	v := os.Getenv(name)
	if v == "" {
		t.Skipf("%s not set", name)
	}
	return v
}

func ctxFor(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)
	return ctx
}

// hashEmbedder is a deterministic bag-of-words embedder, so store tests need no model.
type hashEmbedder struct{}

const hashDim = 32

func (hashEmbedder) Dimension() int { return hashDim }

func (hashEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		v := make([]float32, hashDim)
		for _, w := range strings.Fields(strings.ToLower(text)) {
			h := fnv.New32a()
			h.Write([]byte(strings.Trim(w, ".,?!")))
			v[h.Sum32()%hashDim]++
		}
		var norm float64
		for _, x := range v {
			norm += float64(x * x)
		}
		for j := range v {
			v[j] /= float32(math.Max(math.Sqrt(norm), 1e-9))
		}
		out[i] = v
	}
	return out, nil
}

const (
	plansV1 = "# Plans\n## Pricing\nThe basic plan costs ten dollars a month. It includes one seat.\n\nAnnual billing gives two months free.\n## Support\nSupport is by email only, answered within one business day.\n"
	plansV2 = "# Plans\n## Pricing\nThe basic plan costs twelve dollars a month. It includes one seat.\n\nAnnual billing gives two months free.\n## Support\nSupport is by email only, answered within one business day.\n"
)

// storeRoundTrip exercises a remote vector store through the engine: section metadata,
// search, neighbor lookups, replace-on-ingest and delete-by-source.
func storeRoundTrip(t *testing.T, vs ragout.VectorStore) {
	ctx := ctxFor(t)
	ch, err := chunker.NewCharacterChunker(70, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	e, err := ragout.NewEngine(
		ragout.WithReader(reader.NewByExtension(1<<20)),
		ragout.WithChunker(ch),
		ragout.WithEmbedder(hashEmbedder{}),
		ragout.WithVectorStore(vs),
		ragout.WithReplaceSource(),
		ragout.WithExpander(expand.Neighbors(vs.(ragout.ChunkGetter), 1)),
	)
	if err != nil {
		t.Fatal(err)
	}
	all := []ragout.QueryOption{ragout.WithQueryTopKRecall(50), ragout.WithQueryTopNRerank(50)}

	if err := e.Ingest(ctx, "plans.md", strings.NewReader(plansV1), nil); err != nil {
		t.Fatal(err)
	}
	ans, err := e.Retrieve(ctx, "basic plan price per month", all...)
	if err != nil {
		t.Fatal(err)
	}
	var pricing *ragout.Chunk
	for i := range ans.Retrieved {
		if strings.Contains(ans.Retrieved[i].Chunk.Content, "ten dollars") {
			pricing = &ans.Retrieved[i].Chunk
			break
		}
	}
	if pricing == nil {
		t.Fatalf("pricing chunk not retrieved: %+v", ans.Retrieved)
	}
	if pricing.Metadata[ragout.MetadataSection] != "Plans > Pricing" || pricing.Metadata[ragout.MetadataSource] != "plans.md" || pricing.DocumentID == "" {
		t.Errorf("metadata did not round-trip: %+v", pricing)
	}
	// The section splits into three chunks (heading, price, billing); neighbor expansion
	// joins the hit with at least one of the others, fetched through GetChunks.
	pieces := 0
	for _, p := range []string{"## Pricing", "ten dollars", "Annual billing"} {
		if strings.Contains(pricing.Content, p) {
			pieces++
		}
	}
	if pieces < 2 {
		t.Errorf("expected the hit widened with a neighbor, got %q", pricing.Content)
	}
	next, err := vs.(ragout.ChunkGetter).GetChunks(ctx, []string{ragout.ChunkID(pricing.DocumentID, pricing.Index+1), "missing"})
	if err != nil || len(next) != 1 || next[0].DocumentID != pricing.DocumentID {
		t.Errorf("GetChunks = %+v, %v", next, err)
	}

	if err := e.Ingest(ctx, "plans.md", strings.NewReader(plansV2), nil); err != nil {
		t.Fatal(err)
	}
	ans, err = e.Retrieve(ctx, "basic plan price", all...)
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for _, sc := range ans.Retrieved {
		text.WriteString(sc.Chunk.Content)
	}
	if strings.Contains(text.String(), "ten dollars") || !strings.Contains(text.String(), "twelve dollars") || !strings.Contains(text.String(), "email") {
		t.Errorf("after replace, want only current text:\n%s", text.String())
	}

	if err := e.DeleteSource(ctx, "plans.md"); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Retrieve(ctx, "basic plan price", all...); !errors.Is(err, ragout.ErrNoResults) {
		t.Errorf("after DeleteSource: %v, want ErrNoResults", err)
	}
}

func cleanup(t *testing.T, method, url string) {
	t.Cleanup(func() {
		req, _ := http.NewRequest(method, url, nil)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	})
}

func TestQdrant(t *testing.T) {
	url := env(t, "RAGOUT_IT_QDRANT_URL")
	name := "ragout_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	qs, err := store.NewQdrantStore(ctxFor(t), url, name, hashDim)
	if err != nil {
		t.Fatal(err)
	}
	cleanup(t, http.MethodDelete, url+"/collections/"+name)
	storeRoundTrip(t, qs)
}

func TestChroma(t *testing.T) {
	url := env(t, "RAGOUT_IT_CHROMA_URL")
	name := "ragout_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	cs, err := store.NewChromaStore(ctxFor(t), url, name)
	if err != nil {
		t.Fatal(err)
	}
	cleanup(t, http.MethodDelete, url+"/api/v2/tenants/default_tenant/databases/default_database/collections/"+name)
	storeRoundTrip(t, cs)
}

// llmRoundTrip checks the calls ragout makes to a chat model: Complete, streamed
// generation with citations, contextualization and condensing.
func llmRoundTrip(t *testing.T, llm interface {
	ragout.Completer
	ragout.Generator
}) {
	ctx := ctxFor(t)

	out, err := llm.Complete(ctx, "<document>\nAcme Corp annual report 2024.\n</document>", "In one word, which company is this document about?")
	if err != nil || !strings.Contains(strings.ToLower(out), "acme") {
		t.Errorf("Complete = %q, %v", out, err)
	}

	chunks := []ragout.ScoredChunk{{Chunk: ragout.Chunk{ID: "c1", Content: "The basic plan costs twelve dollars a month."}}}
	sources, tokens := llm.GenerateIter(ctx, "How much is the basic plan?", chunks)
	var answer strings.Builder
	for tok, err := range tokens {
		if err != nil {
			t.Fatal(err)
		}
		answer.WriteString(tok)
	}
	if len(sources) != 1 || answer.Len() == 0 {
		t.Errorf("GenerateIter: %d sources, answer %q", len(sources), answer.String())
	}

	doc := ragout.Document{Content: plansV2, Metadata: map[string]any{ragout.MetadataTitle: "Plans"}}
	ctxs, err := contextual.New(llm).Contextualize(ctx, doc, []ragout.Chunk{{Content: "Support is by email only."}})
	if err != nil || len(ctxs) != 1 || !strings.HasPrefix(ctxs[0], "Plans\n") || len(ctxs[0]) <= len("Plans\n") {
		t.Errorf("Contextualize = %q, %v", ctxs, err)
	}

	history := []ragout.Turn{
		{Role: "user", Content: "What does the Acme basic plan cost?"},
		{Role: "assistant", Content: "Twelve dollars a month."},
	}
	rw, err := transform.Condense(llm).Transform(ctx, "Does it include support?", history)
	if err != nil || rw.Question == "" {
		t.Errorf("Condense = %+v, %v", rw, err)
	}
	t.Logf("condensed: %q", rw.Question)
}

func TestOllama(t *testing.T) {
	url := strings.TrimSuffix(env(t, "RAGOUT_IT_OLLAMA_URL"), "/")
	model := env(t, "RAGOUT_IT_OLLAMA_MODEL")
	llmRoundTrip(t, generator.NewOllamaGenerator(model, url+"/api/generate", generator.WithOllamaNumCtx(8192)))

	if em := os.Getenv("RAGOUT_IT_OLLAMA_EMBED_MODEL"); em != "" {
		emb := embedder.NewOllamaEmbedder(em, embedder.WithOllamaEndpoint(url+"/api/embed"))
		vecs, err := emb.EmbedBatch(ctxFor(t), []string{"hello", "world"})
		if err != nil || len(vecs) != 2 || len(vecs[0]) == 0 {
			t.Errorf("Ollama EmbedBatch = %d vectors, %v", len(vecs), err)
		}
	}
}

func TestOpenAI(t *testing.T) {
	key := env(t, "RAGOUT_IT_OPENAI_API_KEY")
	llmRoundTrip(t, generator.NewOpenAIGenerator(key))

	vecs, err := embedder.NewOpenAIEmbedder(key).EmbedBatch(ctxFor(t), []string{"hello", "world"})
	if err != nil || len(vecs) != 2 || len(vecs[0]) == 0 {
		t.Errorf("OpenAI EmbedBatch = %d vectors, %v", len(vecs), err)
	}
}
