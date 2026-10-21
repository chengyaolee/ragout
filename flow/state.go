package flow

import (
	"time"

	"github.com/chengyaolee/ragout"
)

// Verdict is a grader's judgment of the evidence gathered so far.
type Verdict int

const (
	// Ungraded means no grader has run, or the grader failed (see State.Errors).
	Ungraded Verdict = iota
	// Sufficient means the evidence answers the question.
	Sufficient
	// Insufficient means something is missing; State.Missing says what to search for.
	Insufficient
)

func (v Verdict) String() string {
	switch v {
	case Sufficient:
		return "sufficient"
	case Insufficient:
		return "insufficient"
	}
	return "ungraded"
}

// State is what a run's steps share. Built-in steps use the named fields; custom steps
// may use them too, and keep anything else in Values.
type State struct {
	Question   string        // as asked
	History    []ragout.Turn // earlier turns, from WithHistory
	Standalone string        // the question reranked and answered; the first retrieval may rewrite it
	Query      string        // what the next retrieval searches for; empty means Standalone

	Evidence []ragout.ScoredChunk // every chunk gathered, de-duplicated, in the order found
	Searches []ragout.Search      // every search the retrievals ran
	Hops     int                  // retrievals run so far

	Verdict Verdict
	Missing string // a grader's description of what to search for next

	Answer    ragout.Answer // set by Generate or Abstain
	Abstained bool

	Errors []error        // failures a step absorbed instead of stopping the run
	Trace  []Event        // one entry per step that ran
	Values map[string]any // free slot for custom steps

	onToken func(string) error
}

// Event records one step of a run.
type Event struct {
	Step     string
	Next     string
	Duration time.Duration
	Err      error
}

// AddEvidence appends chunks not already present, by chunk ID. It returns how many
// were new.
func (s *State) AddEvidence(chunks []ragout.ScoredChunk) int {
	seen := make(map[string]bool, len(s.Evidence))
	for _, c := range s.Evidence {
		seen[c.Chunk.ID] = true
	}
	added := 0
	for _, c := range chunks {
		if !seen[c.Chunk.ID] {
			seen[c.Chunk.ID] = true
			s.Evidence = append(s.Evidence, c)
			added++
		}
	}
	return added
}
