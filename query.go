package ragout

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"golang.org/x/sync/errgroup"
)

// Query executes a synchronous end-to-end RAG query and returns the complete
// synthesized answer together with the chunks that produced it.
func (e *Engine) Query(ctx context.Context, query string, opts ...QueryOption) (Answer, error) {
	answer, tokens := e.QueryIter(ctx, query, opts...)
	for _, err := range tokens {
		if err != nil {
			return *answer, err
		}
	}
	return *answer, nil
}

// QueryStream pushes generated tokens to the provided StreamCallback and returns the
// completed answer (including its sources) once streaming finishes.
func (e *Engine) QueryStream(ctx context.Context, query string, cb StreamCallback, opts ...QueryOption) (Answer, error) {
	if cb == nil {
		return Answer{}, ErrNilCallback
	}
	answer, tokens := e.QueryIter(ctx, query, opts...)
	for token, err := range tokens {
		if err != nil {
			return *answer, err
		}
		if err := cb(token); err != nil {
			return *answer, err
		}
	}
	return *answer, nil
}

// QueryIter runs retrieval and starts generation, returning the Answer (with Retrieved
// and Sources already populated, before any token is produced) alongside a lazy token
// stream using modern Go 1.23+ range-over-func iterators. Answer.Text is filled in as
// the stream is consumed, and is complete once the iterator finishes.
func (e *Engine) QueryIter(ctx context.Context, query string, opts ...QueryOption) (*Answer, iter.Seq2[string, error]) {
	params := e.queryParams(opts)
	ctx, endQuery := e.traceQuery(ctx, "ragout.Engine.Query", query)

	answer, err := e.retrieve(ctx, query, params, true)
	if err != nil {
		endQuery(err)
		return answer, func(yield func(string, error) bool) {
			yield("", err)
		}
	}

	genStart := time.Now()
	genCtx, genSpan := startSpan(ctx, "query.generation")

	var sources []ScoredChunk
	var tokens iter.Seq2[string, error]
	if hg, ok := e.generator.(HistoryGenerator); ok && len(params.History) > 0 {
		sources, tokens = hg.GenerateIterWithHistory(genCtx, answer.Question, params.History, answer.Retrieved)
	} else {
		sources, tokens = e.generator.GenerateIter(genCtx, answer.Question, answer.Retrieved)
	}
	answer.Sources = sources
	genSpan.SetAttributes(attribute.Int("prompt_tokens", promptTokens(answer.Question, sources)))

	var text strings.Builder
	first := true

	wrapped := func(yield func(string, error) bool) {
		var genErr error
		defer func() {
			QueryDuration.WithLabelValues("generation").Observe(time.Since(genStart).Seconds())
			endSpan(genSpan, genErr)
			answer.Text = text.String()
			endQuery(genErr)
		}()

		for token, tokenErr := range tokens {
			if first {
				first = false
				genSpan.SetAttributes(attribute.Float64(
					"first_token_latency_ms",
					float64(time.Since(genStart))/float64(time.Millisecond),
				))
			}
			if tokenErr != nil {
				genErr = tokenErr
			}
			text.WriteString(token)
			if !yield(token, tokenErr) {
				return
			}
		}
	}

	return answer, wrapped
}

// Retrieve runs everything Query does before generation (rewrite, search, fusion,
// rerank, expansion) and returns the Answer without Text or Sources. It needs no
// generator, so it suits retrieval-only use and evaluation (see the eval package).
func (e *Engine) Retrieve(ctx context.Context, query string, opts ...QueryOption) (answer Answer, err error) {
	ctx, endQuery := e.traceQuery(ctx, "ragout.Engine.Retrieve", query)
	defer func() { endQuery(err) }()
	a, err := e.retrieve(ctx, query, e.queryParams(opts), false)
	return *a, err
}

func (e *Engine) queryParams(opts []QueryOption) QueryParams {
	params := QueryParams{
		TopKRecall: e.topKRecall,
		TopNRerank: e.topNRerank,
	}
	for _, opt := range opts {
		opt(&params)
	}
	return params
}

// retrieve fills Question, Searches, TransformErr and Retrieved. needGenerator makes a
// missing generator an error before any work is done.
func (e *Engine) retrieve(ctx context.Context, query string, params QueryParams, needGenerator bool) (*Answer, error) {
	answer := &Answer{}
	if strings.TrimSpace(query) == "" {
		return answer, ErrEmptyQuery
	}
	if needGenerator && e.generator == nil {
		return answer, ErrNilGenerator
	}
	if err := ctx.Err(); err != nil {
		return answer, err
	}

	question, searches, transformErr, err := e.rewrite(ctx, query, params.History)
	if err != nil {
		return answer, err
	}
	answer.Question = question
	answer.Searches = searches
	answer.TransformErr = transformErr

	candidates, err := e.retrieveCandidates(ctx, question, searches, params)
	if err != nil {
		return answer, err
	}
	// Candidates are backed by a pooled buffer that's about to be recycled; clone the
	// values out so Answer.Retrieved stays valid for the caller.
	retrieved := append([]ScoredChunk(nil), candidates...)
	releaseCandidates(candidates)
	answer.Retrieved = retrieved

	if len(retrieved) == 0 {
		return answer, ErrNoResults
	}

	if e.expander != nil {
		expStart := time.Now()
		expCtx, expSpan := startSpan(ctx, "query.expand",
			attribute.String("expander_type", fmt.Sprintf("%T", e.expander)),
		)
		retrieved, err = e.expander.Expand(expCtx, retrieved)
		QueryDuration.WithLabelValues("expand").Observe(time.Since(expStart).Seconds())
		endSpan(expSpan, err)
		if err != nil {
			return answer, fmt.Errorf("expansion failed: %w", err)
		}
		answer.Retrieved = retrieved
	}
	return answer, nil
}

// rewrite runs the QueryTransformer, if any, and returns the standalone question, the
// searches to run (the question first) and the transformer's error. A transformer error
// or timeout fails open; only cancellation of ctx itself is returned as err.
func (e *Engine) rewrite(ctx context.Context, query string, history []Turn) (question string, searches []Search, transformErr, err error) {
	question = query
	var extra []Search
	if e.transformer != nil {
		start := time.Now()
		tCtx, span := startSpan(ctx, "query.transform",
			attribute.String("transformer_type", fmt.Sprintf("%T", e.transformer)),
			attribute.Int("history_turns", len(history)),
		)
		if e.transformTimeout > 0 {
			var cancel context.CancelFunc
			tCtx, cancel = context.WithTimeout(tCtx, e.transformTimeout)
			defer cancel()
		}
		rw, tErr := e.transformer.Transform(tCtx, query, history)
		QueryDuration.WithLabelValues("transform").Observe(time.Since(start).Seconds())
		if q := strings.TrimSpace(rw.Question); q != "" {
			question = q
		}
		extra = rw.Searches
		span.SetAttributes(attribute.Int("searches", len(extra)+1))
		endSpan(span, tErr)
		if tErr != nil {
			if ctx.Err() != nil {
				return "", nil, nil, ctx.Err()
			}
			TransformFailuresTotal.Inc()
			transformErr = fmt.Errorf("query transform: %w", tErr)
		}
	}

	searches = []Search{{Text: question, Target: SearchAll}}
	seen := map[string]SearchTarget{normalize(question): SearchAll}
	for _, s := range extra {
		s.Text = strings.TrimSpace(s.Text)
		key := normalize(s.Text)
		if prev, dup := seen[key]; s.Text == "" || dup && (prev == SearchAll || prev == s.Target) {
			continue
		}
		seen[key] = s.Target
		searches = append(searches, s)
	}
	return question, searches, transformErr, nil
}

func normalize(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }

// retrieveCandidates runs every search against the indexes it targets, concurrently,
// fuses the result lists, and reranks against question.
func (e *Engine) retrieveCandidates(ctx context.Context, question string, searches []Search, params QueryParams) (candidates []ScoredChunk, err error) {
	// lists[2i] holds search i's dense hits, lists[2i+1] its sparse hits.
	lists := make([][]ScoredChunk, 2*len(searches))
	releaseAll := func() {
		for _, l := range lists {
			releaseCandidates(l)
		}
	}

	retrCtx, retrSpan := startSpan(ctx, "query.retrieval", attribute.Int("searches", len(searches)))
	defer func() { endSpan(retrSpan, err) }()

	g, searchCtx := errgroup.WithContext(retrCtx)

	if e.vStore != nil && e.embedder != nil {
		var idx []int
		var texts []string
		textLen := 0
		for i, s := range searches {
			if s.Target != SearchSparse {
				idx = append(idx, i)
				texts = append(texts, s.Text)
				textLen += len(s.Text)
			}
		}
		if len(texts) > 0 {
			g.Go(func() error {
				// One embedding call for every dense search.
				embedStart := time.Now()
				_, embedSpan := startSpan(ctx, "query.embed",
					attribute.Int("text.len", textLen),
					attribute.Int("texts", len(texts)),
					attribute.String("model", fmt.Sprintf("%T", e.embedder)),
				)
				vecs, embedErr := e.embedder.EmbedBatch(searchCtx, texts)
				QueryDuration.WithLabelValues("embed").Observe(time.Since(embedStart).Seconds())
				if embedErr != nil {
					endSpan(embedSpan, embedErr)
					return fmt.Errorf("embed query failed: %w", embedErr)
				}
				if len(vecs) != len(texts) {
					endSpan(embedSpan, ErrEmptyEmbeddings)
					return ErrEmptyEmbeddings
				}
				endSpan(embedSpan, nil)

				dg, denseCtx := errgroup.WithContext(searchCtx)
				for j, i := range idx {
					dg.Go(func() error {
						denseStart := time.Now()
						_, denseSpan := startSpan(denseCtx, "query.dense_search",
							attribute.Int("top_k", params.TopKRecall),
						)
						res, denseErr := e.vStore.SearchDense(denseCtx, vecs[j], params.TopKRecall, params.Filter)
						QueryDuration.WithLabelValues("dense").Observe(time.Since(denseStart).Seconds())
						if denseErr != nil {
							endSpan(denseSpan, denseErr)
							return fmt.Errorf("dense search failed: %w", denseErr)
						}
						denseSpan.SetAttributes(attribute.Int("candidates_found", len(res)))
						endSpan(denseSpan, nil)
						lists[2*i] = res
						return nil
					})
				}
				return dg.Wait()
			})
		}
	}

	if e.iStore != nil {
		for i, s := range searches {
			if s.Target == SearchDense {
				continue
			}
			g.Go(func() error {
				sparseStart := time.Now()
				_, sparseSpan := startSpan(searchCtx, "query.sparse_search")
				res, sparseErr := e.iStore.SearchSparse(searchCtx, s.Text, params.TopKRecall, params.Filter)
				QueryDuration.WithLabelValues("sparse").Observe(time.Since(sparseStart).Seconds())
				if sparseErr != nil {
					endSpan(sparseSpan, sparseErr)
					return fmt.Errorf("sparse search failed: %w", sparseErr)
				}
				sparseSpan.SetAttributes(
					attribute.Int("terms_matched", termsMatched(e.iStore, s.Text)),
					attribute.Int("candidates_found", len(res)),
				)
				endSpan(sparseSpan, nil)
				lists[2*i+1] = res
				return nil
			})
		}
	}

	if err = g.Wait(); err != nil {
		releaseAll()
		return nil, err
	}

	var hits [][]ScoredChunk
	for _, l := range lists {
		if len(l) > 0 {
			hits = append(hits, l)
		}
	}
	switch len(hits) {
	case 0:
	case 1:
		// A single ranking needs no fusion.
		candidates = hits[0]
	default:
		fuseStart := time.Now()
		attrs := []attribute.KeyValue{attribute.Int("lists", len(hits))}
		if _, isRRF := e.fusion.(rrf); isRRF {
			attrs = append(attrs, attribute.Int("rrf_k", e.rrfK))
		}
		_, fuseSpan := startSpan(ctx, "query.rrf_fusion", attrs...)
		candidates = e.fusion.Fuse(hits, params.TopKRecall)
		fuseSpan.SetAttributes(attribute.Int("fused_count", len(candidates)))
		endSpan(fuseSpan, nil)
		QueryDuration.WithLabelValues("fusion").Observe(time.Since(fuseStart).Seconds())
		for _, l := range hits {
			if !sameBacking(l, candidates) {
				releaseCandidates(l)
			}
		}
	}

	if e.reranker != nil && len(candidates) > 0 {
		rrStart := time.Now()
		_, rrSpan := startSpan(ctx, "query.reranker",
			attribute.String("reranker_type", fmt.Sprintf("%T", e.reranker)),
			attribute.Int("top_n", params.TopNRerank),
		)
		var ranked []ScoredChunk
		ranked, err = e.reranker.Rerank(ctx, question, candidates, params.TopNRerank)
		QueryDuration.WithLabelValues("rerank").Observe(time.Since(rrStart).Seconds())
		endSpan(rrSpan, err)
		if err != nil {
			releaseCandidates(candidates)
			return nil, fmt.Errorf("reranking failed: %w", err)
		}
		if !sameBacking(ranked, candidates) {
			releaseCandidates(candidates)
		}
		return ranked, nil
	}

	if len(candidates) > params.TopNRerank {
		candidates = candidates[:params.TopNRerank]
	}
	return candidates, nil
}

func termsMatched(store IndexStore, query string) int {
	counter, ok := store.(interface{ TermsMatched(string) int })
	if !ok {
		return 0
	}
	return counter.TermsMatched(query)
}

func promptTokens(query string, candidates []ScoredChunk) int {
	n := estimatedTokens(query)
	for i := range candidates {
		n += estimatedTokens(candidates[i].Chunk.Content)
	}
	return n
}

func estimatedTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return max(1, len(text)/4)
}
