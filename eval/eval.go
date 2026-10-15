// Package eval measures retrieval quality: does the engine put the chunks that answer a
// question near the top? It calls Engine.Retrieve, so it needs no generator and makes no
// generation calls. Run the same cases against differently configured engines to compare
// contextualizers, query transformers, rerankers or expanders on your own data.
package eval

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/chengyaolee/ragout"
)

// Case is one question and the text that relevant chunks contain.
type Case struct {
	Question string
	History  []ragout.Turn // optional, passed with ragout.WithHistory
	// Expected lists text a relevant chunk contains. A retrieved chunk is relevant if
	// it contains any entry, ignoring case and whitespace. Several entries describe a
	// question whose answer is spread over several chunks.
	Expected []string
}

// Report summarizes a run.
type Report struct {
	K       int
	Cases   int
	HitRate float64  // share of cases with at least one relevant chunk in the top K
	Recall  float64  // mean share of each case's Expected entries found in the top K
	MRR     float64  // mean of 1/rank of the first relevant chunk, 0 when there is none
	Misses  []string // questions with no relevant chunk in the top K
}

func (r Report) String() string {
	return fmt.Sprintf("cases=%d k=%d hit_rate=%.3f recall=%.3f mrr=%.3f misses=%d",
		r.Cases, r.K, r.HitRate, r.Recall, r.MRR, len(r.Misses))
}

// Run retrieves the top k chunks for every case (k <= 0 means 5) and scores them.
// A question that retrieves nothing counts as a miss; any other error stops the run.
func Run(ctx context.Context, e *ragout.Engine, cases []Case, k int) (Report, error) {
	if k <= 0 {
		k = 5
	}
	r := Report{K: k, Cases: len(cases)}
	if len(cases) == 0 {
		return r, nil
	}

	var hits, recall, mrr float64
	for _, c := range cases {
		ans, err := e.Retrieve(ctx, c.Question,
			ragout.WithHistory(c.History),
			ragout.WithQueryTopKRecall(max(k, 50)),
			ragout.WithQueryTopNRerank(k),
		)
		if err != nil && !errors.Is(err, ragout.ErrNoResults) {
			return r, fmt.Errorf("eval: %q: %w", c.Question, err)
		}

		found := make([]bool, len(c.Expected))
		first := 0
		for rank, sc := range ans.Retrieved[:min(k, len(ans.Retrieved))] {
			text := normalize(sc.Chunk.Content)
			for i, want := range c.Expected {
				if strings.Contains(text, normalize(want)) {
					found[i] = true
					if first == 0 {
						first = rank + 1
					}
				}
			}
		}

		if first > 0 {
			hits++
			mrr += 1 / float64(first)
		} else {
			r.Misses = append(r.Misses, c.Question)
		}
		if len(c.Expected) > 0 {
			n := 0
			for _, f := range found {
				if f {
					n++
				}
			}
			recall += float64(n) / float64(len(c.Expected))
		}
	}

	total := float64(len(cases))
	r.HitRate, r.Recall, r.MRR = hits/total, recall/total, mrr/total
	return r, nil
}

func normalize(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }
