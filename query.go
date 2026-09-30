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

// Query executes a synchronous end-to-end RAG query and returns the complete synthesized answer.
func (e *Engine) Query(ctx context.Context, query string, opts ...QueryOption) (string, error) {
	var b strings.Builder
	for token, err := range e.QueryIter(ctx, query, opts...) {
		if err != nil {
			return "", err
		}
		b.WriteString(token)
	}
	return b.String(), nil
}

// QueryStream pushes generated tokens to the provided StreamCallback.
func (e *Engine) QueryStream(ctx context.Context, query string, cb StreamCallback, opts ...QueryOption) error {
	if cb == nil {
		return ErrNilCallback
	}
	for token, err := range e.QueryIter(ctx, query, opts...) {
		if err != nil {
			return err
		}
		if err := cb(token); err != nil {
			return err
		}
	}
	return nil
}

// QueryIter streams generated tokens using modern Go 1.23+ range-over-func iterators.
func (e *Engine) QueryIter(ctx context.Context, query string, opts ...QueryOption) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		params := QueryParams{
			TopKRecall: e.topKRecall,
			TopNRerank: e.topNRerank,
		}
		for _, opt := range opts {
			opt(&params)
		}

		ctx, end := e.traceQuery(ctx, query)
		var queryErr error
		defer func() { end(queryErr) }()

		if strings.TrimSpace(query) == "" {
			queryErr = ErrEmptyQuery
			yield("", queryErr)
			return
		}
		if e.generator == nil {
			queryErr = ErrNilGenerator
			yield("", queryErr)
			return
		}
		if err := ctx.Err(); err != nil {
			queryErr = err
			yield("", queryErr)
			return
		}

		candidates, err := e.retrieveCandidates(ctx, query, params)
		if err != nil {
			queryErr = err
			yield("", queryErr)
			return
		}
		defer releaseCandidates(candidates)
		if len(candidates) == 0 {
			queryErr = ErrNoResults
			yield("", queryErr)
			return
		}

		genStart := time.Now()
		genCtx, genSpan := startSpan(ctx, "query.generation",
			attribute.Int("prompt_tokens", promptTokens(query, candidates)),
		)
		var genErr error
		defer func() {
			QueryDuration.WithLabelValues("generation").Observe(time.Since(genStart).Seconds())
			endSpan(genSpan, genErr)
		}()

		first := true
		for token, tokenErr := range e.generator.GenerateIter(genCtx, query, candidates) {
			if first {
				first = false
				genSpan.SetAttributes(attribute.Float64(
					"first_token_latency_ms",
					float64(time.Since(genStart))/float64(time.Millisecond),
				))
			}
			if tokenErr != nil {
				genErr = tokenErr
				queryErr = tokenErr
			}
			if !yield(token, tokenErr) {
				return
			}
		}
	}
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
