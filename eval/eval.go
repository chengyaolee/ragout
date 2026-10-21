// Package eval measures retrieval quality: does the engine put the chunks that answer a
// question near the top? It calls Engine.Retrieve, so it needs no generator and makes no
// generation calls. Run the same cases against differently configured engines to compare
// contextualizers, query transformers, rerankers or expanders on your own data.
package eval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/chengyaolee/ragout"
)

// Case is one question and what a relevant chunk looks like.
type Case struct {
	Question string
	Category string        // optional; Report.ByCategory groups by it
	History  []ragout.Turn // optional, passed with ragout.WithHistory
	// Expected lists text a relevant chunk contains, ignoring case and whitespace.
	// Several entries describe an answer spread over several chunks.
	Expected []string
	// ExpectedSources lists the metadata["source"] of relevant documents, for
	// benchmarks that label relevance by document rather than by text.
	ExpectedSources []string
}

// relevant reports which expectations chunk satisfies, as indexes into the case's
// combined Expected then ExpectedSources list.
func (c Case) matches(chunk ragout.Chunk) []int {
	var out []int
	text := normalize(chunk.Content)
	for i, want := range c.Expected {
		if strings.Contains(text, normalize(want)) {
			out = append(out, i)
		}
	}
	src, _ := chunk.Metadata[ragout.MetadataSource].(string)
	for i, want := range c.ExpectedSources {
		if src == want {
			out = append(out, len(c.Expected)+i)
		}
	}
	return out
}

func (c Case) wants() int { return len(c.Expected) + len(c.ExpectedSources) }

// Metrics are averages over the cases they cover.
type Metrics struct {
	Cases   int
	HitRate float64 // share of cases with at least one relevant chunk in the top K
	Recall  float64 // mean share of each case's expectations found in the top K
	MRR     float64 // mean of 1/rank of the first relevant chunk, 0 when there is none
	NDCG    float64 // mean nDCG@K with binary gains; each expectation counts once
}

// Report summarizes a run.
type Report struct {
	K int
	Metrics
	ByCategory map[string]Metrics // only when some case has a Category
	Misses     []string           // questions with no relevant chunk in the top K
	Skipped    int                // cases with no expectations (e.g. unanswerable); not scored
}

func (r Report) String() string {
	return fmt.Sprintf("cases=%d k=%d hit_rate=%.3f recall=%.3f mrr=%.3f ndcg=%.3f misses=%d",
		r.Cases, r.K, r.HitRate, r.Recall, r.MRR, r.NDCG, len(r.Misses))
}

// Categories returns the category names in ByCategory, sorted.
func (r Report) Categories() []string {
	out := make([]string, 0, len(r.ByCategory))
	for c := range r.ByCategory {
		out = append(out, c)
	}
	sort.Strings(out)
	return out
}

// Run retrieves the top k chunks for every case (k <= 0 means 5) and scores them.
// A question that retrieves nothing counts as a miss; any other error stops the run.
// Cases with no expectations are skipped and counted in Report.Skipped.
func Run(ctx context.Context, e *ragout.Engine, cases []Case, k int) (Report, error) {
	return RunWith(ctx, func(ctx context.Context, c Case) ([]ragout.ScoredChunk, error) {
		ans, err := e.Retrieve(ctx, c.Question,
			ragout.WithHistory(c.History),
			ragout.WithQueryTopKRecall(max(k, 50)),
			ragout.WithQueryTopNRerank(k),
		)
		return ans.Retrieved, err
	}, cases, k)
}

// RetrieveFunc returns the ranked chunks for a case, best first.
type RetrieveFunc func(ctx context.Context, c Case) ([]ragout.ScoredChunk, error)

// RunWith scores any retrieval, for example a flow's gathered evidence, the same way
// Run scores Engine.Retrieve. ragout.ErrNoResults counts as an empty result.
func RunWith(ctx context.Context, retrieve RetrieveFunc, cases []Case, k int) (Report, error) {
	if k <= 0 {
		k = 5
	}
	r := Report{K: k}
	total := &acc{}
	cats := map[string]*acc{}

	for _, c := range cases {
		if c.wants() == 0 {
			r.Skipped++
			continue
		}
		chunks, err := retrieve(ctx, c)
		if err != nil && !errors.Is(err, ragout.ErrNoResults) {
			return r, fmt.Errorf("eval: %q: %w", c.Question, err)
		}
		s := score(c, chunks[:min(k, len(chunks))], k)
		total.add(s)
		if c.Category != "" {
			if cats[c.Category] == nil {
				cats[c.Category] = &acc{}
			}
			cats[c.Category].add(s)
		}
		if s.first == 0 {
			r.Misses = append(r.Misses, c.Question)
		}
	}

	r.Metrics = total.metrics()
	if len(cats) > 0 {
		r.ByCategory = make(map[string]Metrics, len(cats))
		for name, a := range cats {
			r.ByCategory[name] = a.metrics()
		}
	}
	return r, nil
}

type caseScore struct {
	first  int // rank of the first relevant chunk, 0 if none
	recall float64
	ndcg   float64
}

func score(c Case, top []ragout.ScoredChunk, k int) caseScore {
	found := map[int]bool{}
	var s caseScore
	var dcg float64
	for rank, sc := range top {
		gain := 0.0
		for _, m := range c.matches(sc.Chunk) {
			if !found[m] {
				found[m] = true
				gain = 1 // a chunk counts once, however many expectations it meets
			}
		}
		if gain > 0 {
			dcg += gain / math.Log2(float64(rank)+2)
			if s.first == 0 {
				s.first = rank + 1
			}
		}
	}
	var ideal float64
	for i := range min(c.wants(), k) {
		ideal += 1 / math.Log2(float64(i)+2)
	}
	s.recall = float64(len(found)) / float64(c.wants())
	s.ndcg = dcg / ideal
	return s
}

type acc struct {
	n                       int
	hits, recall, mrr, ndcg float64
}

func (a *acc) add(s caseScore) {
	a.n++
	a.recall += s.recall
	a.ndcg += s.ndcg
	if s.first > 0 {
		a.hits++
		a.mrr += 1 / float64(s.first)
	}
}

func (a *acc) metrics() Metrics {
	if a.n == 0 {
		return Metrics{}
	}
	n := float64(a.n)
	return Metrics{Cases: a.n, HitRate: a.hits / n, Recall: a.recall / n, MRR: a.mrr / n, NDCG: a.ndcg / n}
}

func normalize(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }
