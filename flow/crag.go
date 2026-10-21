package flow

import (
	"strings"

	"github.com/chengyaolee/ragout"
)

// Step names used by CRAG. Use them to change a CRAG flow:
//
//	f := flow.CRAG(engine, llm, gen)
//	f.Replace(flow.StepGrade, myGrader)
//	f.InsertAfter(flow.StepRetrieve, "dedupe", myStep)
const (
	StepRetrieve = "retrieve"
	StepGrade    = "grade"
	StepRefine   = "refine"
	StepGenerate = "generate"
	StepAbstain  = "abstain"
)

type cragConfig struct {
	maxHops   int
	exhausted string
	message   string
}

// CRAGOption configures CRAG.
type CRAGOption func(*cragConfig)

// MaxHops caps how many retrievals one question may take. Default 2: the first search
// and one corrective search.
func MaxHops(n int) CRAGOption {
	return func(c *cragConfig) {
		if n > 0 {
			c.maxHops = n
		}
	}
}

// AnswerWhenExhausted answers from whatever evidence was found when the hops run out
// and the grader still says something is missing. By default the flow abstains then.
func AnswerWhenExhausted() CRAGOption { return func(c *cragConfig) { c.exhausted = StepGenerate } }

// AbstainMessage replaces DefaultAbstainMessage.
func AbstainMessage(msg string) CRAGOption { return func(c *cragConfig) { c.message = msg } }

// CRAG returns a corrective-RAG flow:
//
//	retrieve -> grade -+-> generate -> end            (sufficient, or the grader failed)
//	   ^               +-> refine -> retrieve          (insufficient, hops left)
//	   |               +-> abstain -> end              (insufficient, hops used up)
//
// grader judges the evidence and names what is missing; it can be a smaller, cheaper
// model than the one behind gen. The flow is a normal Flow: every step can be replaced,
// removed, or joined by others.
func CRAG(r Retriever, grader ragout.Completer, gen ragout.Generator, opts ...CRAGOption) *Flow {
	cfg := cragConfig{maxHops: 2, exhausted: StepAbstain}
	for _, opt := range opts {
		opt(&cfg)
	}
	return New().
		Add(StepRetrieve, Retrieve(r)).
		Add(StepGrade, Grade(grader)).
		Add(StepRefine, Refine()).
		Add(StepGenerate, Generate(gen)).
		Add(StepAbstain, Abstain(cfg.message)).
		Route(StepGrade, CRAGRoute(cfg.maxHops, cfg.exhausted)).
		Edge(StepRefine, StepRetrieve).
		Edge(StepGenerate, End)
}

// CRAGRoute is CRAG's decision after grading. Sufficient or ungraded evidence goes to
// StepGenerate. Insufficient evidence goes to StepRefine while hops remain and the
// grader named a search not already run; otherwise to exhausted, or to StepAbstain
// when there is no evidence at all.
func CRAGRoute(maxHops int, exhausted string) Router {
	return func(s *State) string {
		switch {
		case s.Verdict != Insufficient && len(s.Evidence) > 0:
			return StepGenerate
		case s.Hops < maxHops && s.Missing != "" && !searched(s, s.Missing):
			return StepRefine
		case len(s.Evidence) == 0:
			return StepAbstain
		}
		return exhausted
	}
}

// searched reports whether query was already run, ignoring case and spacing.
func searched(s *State, query string) bool {
	norm := func(q string) string { return strings.ToLower(strings.Join(strings.Fields(q), " ")) }
	q := norm(query)
	if q == norm(s.Standalone) {
		return true
	}
	for _, sr := range s.Searches {
		if norm(sr.Text) == q {
			return true
		}
	}
	return false
}
