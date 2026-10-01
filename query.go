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
	answer := &Answer{}

	params := QueryParams{
		TopKRecall: e.topKRecall,
		TopNRerank: e.topNRerank,
	}
	for _, opt := range opts {
		opt(&params)
	}

	ctx, endQuery := e.traceQuery(ctx, query)

	fail := func(err error) (*Answer, iter.Seq2[string, error]) {
		endQuery(err)
		return answer, func(yield func(string, error) bool) {
			yield("", err)
		}
	}

	if strings.TrimSpace(query) == "" {
		return fail(ErrEmptyQuery)
	}
	if e.generator == nil {
		return fail(ErrNilGenerator)
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}

	candidates, err := e.retrieveCandidates(ctx, query, params)
	if err != nil {
		return fail(err)
	}
	// Candidates are backed by a pooled buffer that's about to be recycled; clone the
	// values out so Answer.Retrieved stays valid for the caller.
	retrieved := append([]ScoredChunk(nil), candidates...)
	releaseCandidates(candidates)
	answer.Retrieved = retrieved

	if len(retrieved) == 0 {
		return fail(ErrNoResults)
	}

	genStart := time.Now()
	genCtx, genSpan := startSpan(ctx, "query.generation")

	sources, tokens := e.generator.GenerateIter(genCtx, query, retrieved)
	answer.Sources = sources
	genSpan.SetAttributes(attribute.Int("prompt_tokens", promptTokens(query, sources)))

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

// retrieveCandidates executes fan-out parallel retrieval, RRF fusion, and quality reranking.
func (e *Engine) retrieveCandidates(ctx context.Context, query string, params QueryParams) (candidates []ScoredChunk, err error) {
	var denseResults, sparseResults []ScoredChunk

	retrCtx, retrSpan := startSpan(ctx, "query.retrieval")
	defer func() { endSpan(retrSpan, err) }()

	g, searchCtx := errgroup.WithContext(retrCtx)

	if e.vStore != nil && e.embedder != nil {
		g.Go(func() error {
			embedStart := time.Now()
			_, embedSpan := startSpan(ctx, "query.embed",
				attribute.Int("text.len", len(query)),
				attribute.String("model", fmt.Sprintf("%T", e.embedder)),
			)
			vecs, embedErr := e.embedder.EmbedBatch(searchCtx, []string{query})
			QueryDuration.WithLabelValues("embed").Observe(time.Since(embedStart).Seconds())
			if embedErr != nil {
				endSpan(embedSpan, embedErr)
				return fmt.Errorf("embed query failed: %w", embedErr)
			}
			if len(vecs) == 0 {
				endSpan(embedSpan, ErrEmptyEmbeddings)
				return ErrEmptyEmbeddings
			}
			endSpan(embedSpan, nil)

			denseStart := time.Now()
			_, denseSpan := startSpan(searchCtx, "query.dense_search",
				attribute.Int("top_k", params.TopKRecall),
			)
			res, denseErr := e.vStore.SearchDense(searchCtx, vecs[0], params.TopKRecall, params.Filter)
			QueryDuration.WithLabelValues("dense").Observe(time.Since(denseStart).Seconds())
			if denseErr != nil {
				endSpan(denseSpan, denseErr)
				return fmt.Errorf("dense search failed: %w", denseErr)
			}
			denseSpan.SetAttributes(attribute.Int("candidates_found", len(res)))
			endSpan(denseSpan, nil)
			denseResults = res
			return nil
		})
	}

	if e.iStore != nil {
		g.Go(func() error {
			sparseStart := time.Now()
			_, sparseSpan := startSpan(searchCtx, "query.sparse_search")
			res, sparseErr := e.iStore.SearchSparse(searchCtx, query, params.TopKRecall, params.Filter)
			QueryDuration.WithLabelValues("sparse").Observe(time.Since(sparseStart).Seconds())
			if sparseErr != nil {
				endSpan(sparseSpan, sparseErr)
				return fmt.Errorf("sparse search failed: %w", sparseErr)
			}
			sparseSpan.SetAttributes(
				attribute.Int("terms_matched", termsMatched(e.iStore, query)),
				attribute.Int("candidates_found", len(res)),
			)
			endSpan(sparseSpan, nil)
			sparseResults = res
			return nil
		})
	}

	if err = g.Wait(); err != nil {
		releaseCandidates(denseResults)
		releaseCandidates(sparseResults)
		return nil, err
	}

	if len(denseResults) > 0 && len(sparseResults) > 0 {
		fuseStart := time.Now()
		_, fuseSpan := startSpan(ctx, "query.rrf_fusion",
			attribute.Int("rrf_k", e.rrfK),
		)
		candidates = ReciprocalRankFusion(denseResults, sparseResults, params.TopKRecall, e.rrfK)
		fuseSpan.SetAttributes(attribute.Int("fused_count", len(candidates)))
		endSpan(fuseSpan, nil)
		QueryDuration.WithLabelValues("fusion").Observe(time.Since(fuseStart).Seconds())
		releaseCandidates(denseResults)
		releaseCandidates(sparseResults)
	} else if len(denseResults) > 0 {
		candidates = denseResults
	} else {
		candidates = sparseResults
	}

	if e.reranker != nil && len(candidates) > 0 {
		rrStart := time.Now()
		_, rrSpan := startSpan(ctx, "query.reranker",
			attribute.String("reranker_type", fmt.Sprintf("%T", e.reranker)),
			attribute.Int("top_n", params.TopNRerank),
		)
		var ranked []ScoredChunk
		ranked, err = e.reranker.Rerank(ctx, query, candidates, params.TopNRerank)
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
