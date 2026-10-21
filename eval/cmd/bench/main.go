// Command bench measures ragout's retrieval methods against a question set and writes
// a markdown report. Data lives in eval/data (gitignored); reports go to eval/results.
//
//	go run ./eval/cmd/bench -suite docs    -out eval/results/docs.md
//	go run ./eval/cmd/bench -suite scifact -out eval/results/scifact.md
//
// It needs Ollama with an embedding model and a chat model, and optionally rerank
// servers (llama-server --reranking) for the reranker configurations.
package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/chunker"
	"github.com/chengyaolee/ragout/contextual"
	"github.com/chengyaolee/ragout/embedder"
	"github.com/chengyaolee/ragout/eval"
	"github.com/chengyaolee/ragout/expand"
	"github.com/chengyaolee/ragout/flow"
	"github.com/chengyaolee/ragout/reader"
	"github.com/chengyaolee/ragout/reranker"
	"github.com/chengyaolee/ragout/store"
	"github.com/chengyaolee/ragout/transform"
)

var (
	suiteFlag  = flag.String("suite", "docs", "question set: docs or scifact")
	dataDir    = flag.String("data", "eval/data", "directory holding question sets and benchmarks")
	ollamaURL  = flag.String("ollama", "http://localhost:11434", "Ollama base URL")
	embedModel = flag.String("embed", "nomic-embed-text", "Ollama embedding model")
	embedDim   = flag.Int("dim", 768, "embedding dimension")
	llmModel   = flag.String("llm", "qwen2.5:3b", "Ollama chat model for LLM methods")
	bgeURL     = flag.String("bge", "", "rerank URL serving bge-reranker-v2-m3; empty skips it")
	qwenURL    = flag.String("qwen", "", "rerank URL serving Qwen3-Reranker-4B; empty skips it")
	kFlag      = flag.Int("k", 0, "cut-off k (default 5 for docs, 10 for scifact)")
	limit      = flag.Int("limit", 0, "use at most this many questions (0: all)")
	only       = flag.String("configs", "", "comma-separated config IDs to run (default: all)")
	outPath    = flag.String("out", "", "write the markdown report here (default: stdout)")
	rewriteW   = flag.Float64("rewrite-weight", 0, "fuse with WeightedRRF(60, w) instead of the engine default (0: default)")
)

// suite is a corpus plus questions.
type suite struct {
	name      string
	docs      []doc
	cases     []eval.Case
	k         int
	chunkSize int
	overlap   int
	reader    ragout.Reader
	note      string
}

type doc struct {
	source string
	body   string
	meta   map[string]any
}

// config is one retrieval setup to score.
type config struct {
	id      string
	label   string
	variant string // index variant: noctx, meta or llmctx
	stores  string // bm25, dense or hybrid
	suites  []string
	needs   string // llm, bge or qwen: skipped when that model is not available
	recall  int    // candidates handed to the reranker
	opts    func(r *resources) []ragout.EngineOption
	flow    bool // run CRAG instead of plain retrieval
}

type resources struct {
	llm      ragout.Completer
	emb      ragout.Embedder
	bge, qwn ragout.Reranker
}

func configs() []config {
	all := []string{"docs", "scifact"}
	docsOnly := []string{"docs"}
	none := func(*resources) []ragout.EngineOption { return nil }
	tr := func(t func(r *resources) ragout.QueryTransformer) func(*resources) []ragout.EngineOption {
		return func(r *resources) []ragout.EngineOption {
			return []ragout.EngineOption{ragout.WithQueryTransformer(t(r))}
		}
	}
	return []config{
		{id: "bm25-noctx", label: "BM25, no context", variant: "noctx", stores: "bm25", suites: all, opts: none},
		{id: "dense-noctx", label: "Dense, no context", variant: "noctx", stores: "dense", suites: all, opts: none},
		{id: "hybrid-noctx", label: "Hybrid, no context", variant: "noctx", stores: "hybrid", suites: all, opts: none},
		{id: "bm25-meta", label: "BM25 + metadata", variant: "meta", stores: "bm25", suites: all, opts: none},
		{id: "dense-meta", label: "Dense + metadata", variant: "meta", stores: "dense", suites: all, opts: none},
		{id: "hybrid-meta", label: "Hybrid + metadata (default)", variant: "meta", stores: "hybrid", suites: all, opts: none},
		{id: "multiquery", label: "Default + MultiQuery(3)", variant: "meta", stores: "hybrid", suites: all, needs: "llm",
			opts: tr(func(r *resources) ragout.QueryTransformer { return transform.MultiQuery(r.llm, 3) })},
		{id: "hyde", label: "Default + HyDE", variant: "meta", stores: "hybrid", suites: all, needs: "llm",
			opts: tr(func(r *resources) ragout.QueryTransformer { return transform.HyDE(r.llm) })},
		{id: "decompose", label: "Default + Decompose(4)", variant: "meta", stores: "hybrid", suites: docsOnly, needs: "llm",
			opts: tr(func(r *resources) ragout.QueryTransformer { return transform.Decompose(r.llm, 4) })},
		{id: "condense", label: "Default + Condense", variant: "meta", stores: "hybrid", suites: docsOnly, needs: "llm",
			opts: tr(func(r *resources) ragout.QueryTransformer { return transform.Condense(r.llm) })},
		{id: "bge", label: "Default + bge-reranker-v2-m3", variant: "meta", stores: "hybrid", suites: all, needs: "bge", recall: 30,
			opts: func(r *resources) []ragout.EngineOption { return []ragout.EngineOption{ragout.WithReranker(r.bge)} }},
		{id: "qwen3", label: "Default + Qwen3-Reranker-4B", variant: "meta", stores: "hybrid", suites: all, needs: "qwen", recall: 30,
			opts: func(r *resources) []ragout.EngineOption { return []ragout.EngineOption{ragout.WithReranker(r.qwn)} }},
		{id: "neighbors", label: "Default + Neighbors(1)", variant: "meta", stores: "hybrid", suites: docsOnly, opts: none},
		{id: "llmctx", label: "Hybrid + LLM context", variant: "llmctx", stores: "hybrid", suites: docsOnly, needs: "llm", opts: none},
		{id: "crag", label: "Default + CRAG (2 hops)", variant: "meta", stores: "hybrid", suites: docsOnly, needs: "llm", opts: none, flow: true},
	}
}

func main() {
	flag.Parse()
	ctx := context.Background()

	s, err := loadSuite(*suiteFlag)
	if err != nil {
		log.Fatal(err)
	}
	if *kFlag > 0 {
		s.k = *kFlag
	}
	if *limit > 0 && len(s.cases) > *limit {
		s.cases = s.cases[:*limit]
	}

	res := &resources{
		emb: embedder.NewOllamaEmbedder(*embedModel,
			embedder.WithOllamaEndpoint(*ollamaURL+"/api/embed"), embedder.WithOllamaDimension(*embedDim)),
		llm: &ollamaLLM{url: *ollamaURL, model: *llmModel},
	}
	if *bgeURL != "" {
		res.bge = reranker.NewHTTP(*bgeURL)
	}
	if *qwenURL != "" {
		res.qwn = reranker.NewHTTP(*qwenURL, reranker.WithModel(reranker.Qwen3Reranker4B))
	}

	var run []config
	for _, c := range configs() {
		if !slices.Contains(c.suites, s.name) || (*only != "" && !slices.Contains(strings.Split(*only, ","), c.id)) {
			continue
		}
		if (c.needs == "bge" && res.bge == nil) || (c.needs == "qwen" && res.qwn == nil) {
			log.Printf("skip %s: no %s server", c.id, c.needs)
			continue
		}
		run = append(run, c)
	}

	rep := &report{suite: s, started: time.Now(), ingest: map[string]ingestStat{}}
	indexes := map[string]*index{}
	for _, c := range run {
		ix, ok := indexes[c.variant]
		if !ok {
			log.Printf("indexing %s (%s)", s.name, c.variant)
			ix, err = buildIndex(ctx, s, c.variant, res)
			if err != nil {
				log.Fatalf("index %s: %v", c.variant, err)
			}
			indexes[c.variant] = ix
			rep.ingest[c.variant] = ix.stat
		}
		log.Printf("running %s", c.id)
		row, err := runConfig(ctx, s, c, ix, res)
		if err != nil {
			log.Fatalf("%s: %v", c.id, err)
		}
		rep.rows = append(rep.rows, row)
		log.Printf("  %s", row.report)
	}

	md := rep.markdown()
	if *outPath == "" {
		fmt.Print(md)
		return
	}
	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*outPath, []byte(md), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %s", *outPath)
}

// ---- indexing -------------------------------------------------------------

type ingestStat struct {
	docs   int
	chunks int
	took   time.Duration
}

type index struct {
	vs   *store.VectorStore
	bm25 *store.BM25Index
	stat ingestStat
}

func buildIndex(ctx context.Context, s suite, variant string, r *resources) (*index, error) {
	ix := &index{vs: store.NewVectorStore(), bm25: store.NewBM25Index()}
	var cz ragout.Contextualizer = ragout.MetadataContext()
	switch variant {
	case "noctx":
		cz = nil
	case "llmctx":
		cz = contextual.New(r.llm, contextual.WithConcurrency(2))
	}
	ch, err := chunker.NewCharacterChunker(s.chunkSize, s.overlap, nil)
	if err != nil {
		return nil, err
	}
	e, err := ragout.NewEngine(
		ragout.WithReader(s.reader),
		ragout.WithChunker(ch),
		ragout.WithEmbedder(r.emb),
		ragout.WithVectorStore(ix.vs),
		ragout.WithIndexStore(ix.bm25),
		ragout.WithContextualizer(cz),
		ragout.WithEmbedBatchSize(64),
	)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	jobs := make(chan doc)
	errs := make(chan error, 1)
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for d := range jobs {
				err := e.Ingest(ctx, d.source, strings.NewReader(d.body), d.meta)
				if err != nil && !errors.Is(err, ragout.ErrEmptyDocument) {
					select {
					case errs <- fmt.Errorf("%s: %w", d.source, err):
					default:
					}
				}
			}
		}()
	}
	for i, d := range s.docs {
		select {
		case err := <-errs:
			close(jobs)
			return nil, err
		case jobs <- d:
		}
		if (i+1)%1000 == 0 {
			log.Printf("  ingested %d/%d", i+1, len(s.docs))
		}
	}
	close(jobs)
	wg.Wait()
	select {
	case err := <-errs:
		return nil, err
	default:
	}

	all, _ := ix.vs.SearchDense(ctx, make([]float32, *embedDim), 1<<30, nil)
	ix.stat = ingestStat{docs: len(s.docs), chunks: len(all), took: time.Since(start)}
	return ix, nil
}

// ---- running --------------------------------------------------------------

type row struct {
	cfg    config
	report eval.Report
	perQ   time.Duration
	crag   *cragStat
}

type cragStat struct {
	answerable, unanswerable    int
	abstainAns, abstainUnans    int
	hops, evidence, graderFails int
	evidenceReport              eval.Report
}

func runConfig(ctx context.Context, s suite, c config, ix *index, r *resources) (row, error) {
	opts := []ragout.EngineOption{ragout.WithReranker(nil)}
	switch c.stores {
	case "bm25":
		opts = append(opts, ragout.WithIndexStore(ix.bm25))
	case "dense":
		opts = append(opts, ragout.WithEmbedder(r.emb), ragout.WithVectorStore(ix.vs))
	default:
		opts = append(opts, ragout.WithEmbedder(r.emb), ragout.WithVectorStore(ix.vs), ragout.WithIndexStore(ix.bm25))
	}
	opts = append(opts, c.opts(r)...)
	if *rewriteW > 0 {
		opts = append(opts, ragout.WithFusion(ragout.WeightedRRF(60, *rewriteW)))
	}
	if c.id == "neighbors" {
		opts = append(opts, ragout.WithExpander(expand.Neighbors(ix.vs, 1)))
	}
	e, err := ragout.NewEngine(opts...)
	if err != nil {
		return row{}, err
	}
	recall := 50
	if c.recall > 0 {
		recall = c.recall
	}

	if c.flow {
		return runCRAG(ctx, s, c, e, r, recall)
	}

	start := time.Now()
	retrieve := func(ctx context.Context, cs eval.Case) ([]ragout.ScoredChunk, error) {
		ans, err := e.Retrieve(ctx, cs.Question,
			ragout.WithHistory(cs.History),
			ragout.WithQueryTopKRecall(recall),
			ragout.WithQueryTopNRerank(s.k))
		return ans.Retrieved, err
	}
	rep, err := eval.RunWith(ctx, retrieve, s.cases, s.k)
	if err != nil {
		return row{}, err
	}
	return row{cfg: c, report: rep, perQ: perQuery(time.Since(start), rep.Cases)}, nil
}

// runCRAG runs the corrective flow without its generation step, since answers are not
// scored here: it measures what the loop retrieves and when it abstains.
func runCRAG(ctx context.Context, s suite, c config, e *ragout.Engine, r *resources, recall int) (row, error) {
	f := flow.CRAG(retrieverWith(e, recall, s.k), r.llm, nil)
	if err := f.Replace(flow.StepGenerate, flow.StepFunc(func(context.Context, *flow.State) error { return nil })); err != nil {
		return row{}, err
	}

	st := &cragStat{}
	evidence := map[string][]ragout.ScoredChunk{}
	start := time.Now()
	for _, cs := range s.cases {
		state, err := f.Run(ctx, cs.Question, flow.WithHistory(cs.History))
		if err != nil {
			return row{}, fmt.Errorf("%q: %w", cs.Question, err)
		}
		evidence[cs.Question] = state.Evidence
		st.hops += state.Hops
		st.evidence += len(state.Evidence)
		st.graderFails += len(state.Errors)
		if len(cs.Expected)+len(cs.ExpectedSources) == 0 {
			st.unanswerable++
			if state.Abstained {
				st.abstainUnans++
			}
		} else {
			st.answerable++
			if state.Abstained {
				st.abstainAns++
			}
		}
	}
	took := time.Since(start)
	// Score the first k chunks the loop found, like every other config.
	rep, err := eval.RunWith(ctx, func(_ context.Context, cs eval.Case) ([]ragout.ScoredChunk, error) {
		return evidence[cs.Question], nil
	}, s.cases, s.k)
	if err != nil {
		return row{}, err
	}
	// And everything it gathered across hops.
	st.evidenceReport, _ = eval.RunWith(ctx, func(_ context.Context, cs eval.Case) ([]ragout.ScoredChunk, error) {
		return evidence[cs.Question], nil
	}, s.cases, 1<<20)
	return row{cfg: c, report: rep, perQ: perQuery(took, len(s.cases)), crag: st}, nil
}

type fixedRetriever struct {
	e         *ragout.Engine
	recall, k int
}

func retrieverWith(e *ragout.Engine, recall, k int) flow.Retriever {
	return fixedRetriever{e, recall, k}
}

func (f fixedRetriever) Retrieve(ctx context.Context, q string, opts ...ragout.QueryOption) (ragout.Answer, error) {
	opts = append(opts, ragout.WithQueryTopKRecall(f.recall), ragout.WithQueryTopNRerank(f.k))
	return f.e.Retrieve(ctx, q, opts...)
}

func perQuery(d time.Duration, n int) time.Duration {
	if n == 0 {
		return 0
	}
	return d / time.Duration(n)
}

// ---- deterministic model calls -------------------------------------------

// ollamaLLM calls Ollama at temperature 0 with a fixed seed, so LLM-based methods give
// the same result on every run. Thinking is off.
type ollamaLLM struct{ url, model string }

func (o *ollamaLLM) Complete(ctx context.Context, prefix, prompt string) (string, error) {
	if prefix != "" {
		prompt = prefix + "\n\n" + prompt
	}
	body, _ := json.Marshal(map[string]any{
		"model": o.model, "prompt": prompt, "stream": false, "think": false,
		"options": map[string]any{"temperature": 0, "seed": 7, "num_ctx": 8192},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.url+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var out struct {
		Response string `json:"response"`
		Error    string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if out.Error != "" {
		return "", errors.New(out.Error)
	}
	return out.Response, nil
}

// ---- suites ---------------------------------------------------------------

func loadSuite(name string) (suite, error) {
	switch name {
	case "docs":
		return loadDocs()
	case "scifact":
		return loadSciFact()
	}
	return suite{}, fmt.Errorf("unknown suite %q", name)
}

// loadDocs reads ragout's own documentation and the hand-written question set.
func loadDocs() (suite, error) {
	s := suite{name: "docs", k: 5, chunkSize: 800, overlap: 0, reader: reader.NewMarkdownReader(1 << 20),
		note: "ragout's own docs/ (Markdown, split by heading). Questions are hand-written, grouped by the method each targets; expected strings each occur exactly once in the corpus."}
	var file struct {
		Cases []eval.Case `json:"cases"`
	}
	raw, err := os.ReadFile(filepath.Join(*dataDir, "ragout-docs.json"))
	if err != nil {
		return s, err
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return s, err
	}
	s.cases = file.Cases

	err = filepath.WalkDir("docs", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || filepath.Ext(path) != ".md" {
			return err
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		s.docs = append(s.docs, doc{source: path, body: string(b)})
		return nil
	})
	if err != nil {
		return s, err
	}
	return s, validate(s)
}

// validate checks that every expected string occurs in the corpus.
func validate(s suite) error {
	var corpus strings.Builder
	for _, d := range s.docs {
		corpus.WriteString(strings.ToLower(strings.Join(strings.Fields(d.body), " ")))
		corpus.WriteString(" ")
	}
	text := corpus.String()
	for _, c := range s.cases {
		for _, e := range c.Expected {
			if !strings.Contains(text, strings.ToLower(strings.Join(strings.Fields(e), " "))) {
				return fmt.Errorf("question %q: expected text %q is not in the corpus", c.Question, e)
			}
		}
	}
	return nil
}

const sciFactURL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

// loadSciFact reads BEIR SciFact (CC BY-SA 4.0), downloading it on first use.
func loadSciFact() (suite, error) {
	s := suite{name: "scifact", k: 10, chunkSize: 1000, overlap: 100, reader: reader.NewTextReader(1 << 20),
		note: "BEIR SciFact test split: scientific claims against 5,183 paper abstracts (CC BY-SA 4.0). Relevance is by document; the title is stored as metadata, so it is indexed only when chunk context is on."}
	zipPath := filepath.Join(*dataDir, "beir", "scifact.zip")
	if _, err := os.Stat(zipPath); err != nil {
		if err := download(sciFactURL, zipPath); err != nil {
			return s, err
		}
	}
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return s, err
	}
	defer zr.Close()
	open := func(name string) (io.ReadCloser, error) { return zr.Open("scifact/" + name) }

	corpus, err := open("corpus.jsonl")
	if err != nil {
		return s, err
	}
	sc := bufio.NewScanner(corpus)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		var d struct{ ID, Title, Text string }
		var raw map[string]any
		if err := json.Unmarshal(sc.Bytes(), &raw); err != nil {
			return s, err
		}
		d.ID, _ = raw["_id"].(string)
		d.Title, _ = raw["title"].(string)
		d.Text, _ = raw["text"].(string)
		s.docs = append(s.docs, doc{source: d.ID, body: d.Text, meta: map[string]any{ragout.MetadataTitle: d.Title}})
	}
	corpus.Close()

	queries := map[string]string{}
	qf, err := open("queries.jsonl")
	if err != nil {
		return s, err
	}
	qs := bufio.NewScanner(qf)
	for qs.Scan() {
		var raw map[string]any
		if err := json.Unmarshal(qs.Bytes(), &raw); err != nil {
			return s, err
		}
		id, _ := raw["_id"].(string)
		queries[id], _ = raw["text"].(string)
	}
	qf.Close()

	rf, err := open("qrels/test.tsv")
	if err != nil {
		return s, err
	}
	rows, err := func() ([][]string, error) {
		r := csv.NewReader(rf)
		r.Comma = '\t'
		return r.ReadAll()
	}()
	rf.Close()
	if err != nil {
		return s, err
	}
	rel := map[string][]string{}
	var order []string
	for _, r := range rows[1:] { // header: query-id corpus-id score
		if r[2] == "0" {
			continue
		}
		if _, ok := rel[r[0]]; !ok {
			order = append(order, r[0])
		}
		rel[r[0]] = append(rel[r[0]], r[1])
	}
	for _, qid := range order {
		s.cases = append(s.cases, eval.Case{Question: queries[qid], ExpectedSources: rel[qid]})
	}
	return s, nil
}

func download(url, path string) error {
	log.Printf("downloading %s", url)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}
	f, err := os.Create(path + ".tmp")
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	f.Close()
	return os.Rename(path+".tmp", path)
}

// ---- report ---------------------------------------------------------------

type report struct {
	suite   suite
	started time.Time
	ingest  map[string]ingestStat
	rows    []row
}

func (r *report) markdown() string {
	var b strings.Builder
	commit, _ := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	k := r.suite.k
	fmt.Fprintf(&b, "# Retrieval evaluation: %s\n\n", r.suite.name)
	fmt.Fprintf(&b, "%s\n\n", r.suite.note)
	fmt.Fprintf(&b, "| Setting | Value |\n| --- | --- |\n")
	fmt.Fprintf(&b, "| Date | %s |\n| Commit | `%s` |\n", r.started.Format("2006-01-02"), strings.TrimSpace(string(commit)))
	fmt.Fprintf(&b, "| Questions scored | %d |\n| Cut-off k | %d |\n", r.rows[0].report.Cases, k)
	fmt.Fprintf(&b, "| Chunking | character, %d bytes, %d overlap |\n", r.suite.chunkSize, r.suite.overlap)
	fmt.Fprintf(&b, "| Embedder | `%s` via Ollama (no task prefixes) |\n| LLM methods | `%s` via Ollama, temperature 0, thinking off |\n", *embedModel, *llmModel)
	fmt.Fprintf(&b, "| Rerankers | `bge-reranker-v2-m3` Q8_0 and `Qwen3-Reranker-4B` Q4_K_M via llama.cpp, top 30 fused candidates |\n")
	for _, v := range []string{"noctx", "meta", "llmctx"} {
		if st, ok := r.ingest[v]; ok {
			fmt.Fprintf(&b, "| Index `%s` | %d documents, %d chunks, built in %s |\n", v, st.docs, st.chunks, st.took.Round(time.Second))
		}
	}

	fmt.Fprintf(&b, "\n## Overall\n\n| Config | Hit@%d | Recall@%d | MRR | nDCG@%d | Time per question |\n| --- | --- | --- | --- | --- | --- |\n", k, k, k)
	for _, rw := range r.rows {
		m := rw.report.Metrics
		fmt.Fprintf(&b, "| %s | %.3f | %.3f | %.3f | %.3f | %.0f ms |\n", rw.cfg.label, m.HitRate, m.Recall, m.MRR, m.NDCG, float64(rw.perQ)/float64(time.Millisecond))
	}

	if cats := r.rows[0].report.Categories(); len(cats) > 0 {
		fmt.Fprintf(&b, "\n## Recall@%d by question category\n\nEach category targets one method; see the question set.\n\n| Config |", k)
		for _, c := range cats {
			fmt.Fprintf(&b, " %s (%d) |", c, r.rows[0].report.ByCategory[c].Cases)
		}
		fmt.Fprintf(&b, "\n| --- |%s\n", strings.Repeat(" --- |", len(cats)))
		for _, rw := range r.rows {
			fmt.Fprintf(&b, "| %s |", rw.cfg.label)
			for _, c := range cats {
				fmt.Fprintf(&b, " %.3f |", rw.report.ByCategory[c].Recall)
			}
			b.WriteString("\n")
		}
	}

	for _, rw := range r.rows {
		if c := rw.crag; c != nil {
			fmt.Fprintf(&b, "\n## CRAG (corrective loop)\n\nThe loop runs without its generation step; answers are not scored here.\n\n")
			fmt.Fprintf(&b, "| Measure | Value |\n| --- | --- |\n")
			fmt.Fprintf(&b, "| Abstained on answerable questions (lower is better) | %d / %d |\n", c.abstainAns, c.answerable)
			fmt.Fprintf(&b, "| Abstained on unanswerable questions (higher is better) | %d / %d |\n", c.abstainUnans, c.unanswerable)
			n := c.answerable + c.unanswerable
			fmt.Fprintf(&b, "| Retrievals per question | %.2f |\n| Evidence chunks per question | %.1f |\n", float64(c.hops)/float64(n), float64(c.evidence)/float64(n))
			fmt.Fprintf(&b, "| Recall over all gathered evidence | %.3f |\n| Grader failures | %d |\n", c.evidenceReport.Recall, c.graderFails)
		}
	}
	return b.String()
}
