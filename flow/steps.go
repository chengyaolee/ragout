package flow

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"regexp"
	"slices"
	"strings"

	"github.com/chengyaolee/ragout"
)

// Retriever is what Retrieve searches with. *ragout.Engine implements it, so the
// engine's query transformer, fusion, reranker and expander all apply.
type Retriever interface {
	Retrieve(ctx context.Context, query string, opts ...ragout.QueryOption) (ragout.Answer, error)
}

// Retrieve searches for State.Query (State.Standalone when empty) and adds the hits to
// State.Evidence. The first retrieval passes State.History, so a query transformer
// such as transform.Condense can resolve a follow-up, and records the standalone
// question it produced. Finding nothing is not an error: it just adds no evidence.
func Retrieve(r Retriever, opts ...ragout.QueryOption) Step {
	return StepFunc(func(ctx context.Context, s *State) error {
		query := s.Query
		if query == "" {
			query = s.Standalone
		}
		first := s.Hops == 0
		o := opts
		if first && len(s.History) > 0 {
			o = append(slices.Clone(opts), ragout.WithHistory(s.History))
		}

		ans, err := r.Retrieve(ctx, query, o...)
		s.Hops++
		if err != nil && !errors.Is(err, ragout.ErrNoResults) {
			return err
		}
		if first && ans.Question != "" {
			s.Standalone = ans.Question
		}
		if ans.TransformErr != nil {
			s.Errors = append(s.Errors, ans.TransformErr)
		}
		s.Searches = append(s.Searches, ans.Searches...)
		s.AddEvidence(ans.Retrieved)
		return nil
	})
}

const (
	gradeChunks     = 12   // evidence chunks shown to the grader, in the order found
	gradeChunkBytes = 1500 // bytes kept from each
	gradeTurns      = 6    // most recent turns shown when the question is a follow-up
)

// Grade asks a model which passages of State.Evidence help answer State.Standalone,
// the way corrective RAG grades each retrieved document. If at least one does, the
// verdict is Sufficient. If none does, the verdict is Insufficient and State.Missing
// holds the model's suggested search, if it gave one. The evidence keeps its retrieval
// order: on the docs benchmark, moving the grader's picks to the front lowered MRR from
// 0.60 to 0.39, because a small grader often picks a neighbor of the passage that answers.
//
// Grading relevance passage by passage is a simpler judgment than "is this enough to
// answer?", and small models make it far more reliably: on this module's docs
// benchmark, a 3B model judging sufficiency abstained on 27 of 42 answerable questions;
// judging relevance, on 5. When the run has history and no query transformer rewrote
// the question, the grader sees the conversation, so it can judge a follow-up.
//
// The reply is read as plain lines, so any chat model works. Grade fails open: if the
// model errors or the reply cannot be read, the verdict stays Ungraded, the error goes to
// State.Errors, and the run continues.
func Grade(c ragout.Completer) Step {
	return StepFunc(func(ctx context.Context, s *State) error {
		s.Verdict, s.Missing = Ungraded, ""
		if len(s.Evidence) == 0 {
			s.Verdict = Insufficient
			return nil
		}

		var b strings.Builder
		if len(s.History) > 0 && s.Standalone == s.Question {
			b.WriteString("Conversation so far:\n")
			for _, t := range s.History[max(0, len(s.History)-gradeTurns):] {
				fmt.Fprintf(&b, "%s: %s\n", t.Role, truncate(strings.TrimSpace(t.Content), gradeChunkBytes))
			}
			b.WriteString("\n")
		}
		shown := s.Evidence[:min(gradeChunks, len(s.Evidence))]
		fmt.Fprintf(&b, "Question: %s\n\nPassages:\n", s.Standalone)
		for i, sc := range shown {
			fmt.Fprintf(&b, "[%d] %s\n\n", i+1, truncate(sc.Chunk.IndexText(), gradeChunkBytes))
		}
		b.WriteString(gradeInstruction)

		reply, err := c.Complete(ctx, "", b.String())
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			s.Errors = append(s.Errors, fmt.Errorf("grade: %w", err))
			return nil
		}
		relevant, missing, ok := parseGrade(reply, len(shown))
		if !ok {
			s.Errors = append(s.Errors, fmt.Errorf("grade: unreadable reply %q", truncate(reply, 200)))
			return nil
		}
		if len(relevant) == 0 {
			s.Verdict, s.Missing = Insufficient, missing
			return nil
		}
		s.Verdict = Sufficient
		return nil
	})
}

const gradeInstruction = "Which passages contain information that helps answer the question? Reply with their numbers separated by commas, for example: 2, 5. If none of them do, reply NONE, and on the next line write a different search query that might find the answer."

var (
	thinkBlock = regexp.MustCompile(`(?is)<think>.*?</think>`)
	listMarker = regexp.MustCompile(`^(?:\d+[.)]|[-*•])\s+`)
	number     = regexp.MustCompile(`\d+`)
	// A label in front of the suggested search, e.g. "Query:" or "A better search query could be:".
	queryLabel = regexp.MustCompile(`(?i)^(?:[^:]{0,60}\b(?:query|search)\b[^:]{0,40}|missing)\s*:\s*`)
)

// parseGrade reads a grader reply. The first line holds passage numbers (1-based, at
// most n) or NONE; after NONE, the next usable line is the suggested search. It returns
// the relevant passages as 0-based indexes.
func parseGrade(reply string, n int) (relevant []int, missing string, ok bool) {
	var lines []string
	for _, l := range strings.Split(thinkBlock.ReplaceAllString(reply, ""), "\n") {
		if l = strings.Trim(strings.TrimSpace(l), "*`\"'“” "); l != "" {
			lines = append(lines, l)
		}
	}
	if len(lines) == 0 {
		return nil, "", false
	}
	head := strings.ToUpper(lines[0])
	if strings.Contains(head, "NONE") || strings.Contains(head, "INSUFFICIENT") {
		for _, l := range lines[1:] {
			if q := searchLine(l); q != "" {
				return nil, q, true
			}
		}
		return nil, "", true
	}
	seen := map[int]bool{}
	for _, m := range number.FindAllString(lines[0], -1) {
		var i int
		fmt.Sscan(m, &i)
		if i >= 1 && i <= n && !seen[i-1] {
			seen[i-1] = true
			relevant = append(relevant, i-1)
		}
	}
	return relevant, "", len(relevant) > 0
}

// searchLine cleans a suggested search: drops list markers, labels and quotes, and
// rejects a line that only echoes the instruction.
func searchLine(l string) string {
	l = listMarker.ReplaceAllString(l, "")
	l = strings.Trim(queryLabel.ReplaceAllString(l, ""), "*`\"'“” ")
	if l == "" || strings.Contains(strings.ToLower(l), "which passages") {
		return ""
	}
	return l
}

// Refine makes the grader's Missing text the next search.
func Refine() Step {
	return StepFunc(func(_ context.Context, s *State) error {
		s.Query = s.Missing
		return nil
	})
}

// Generate answers State.Standalone from State.Evidence and fills State.Answer. With
// history passed to the run and a generator that implements ragout.HistoryGenerator,
// the history goes into the prompt too. Tokens stream to WithTokens as they arrive.
// No evidence at all is ragout.ErrNoResults.
func Generate(g ragout.Generator) Step {
	return StepFunc(func(ctx context.Context, s *State) error {
		if len(s.Evidence) == 0 {
			return ragout.ErrNoResults
		}
		var sources []ragout.ScoredChunk
		var tokens iter.Seq2[string, error]
		if hg, ok := g.(ragout.HistoryGenerator); ok && len(s.History) > 0 {
			sources, tokens = hg.GenerateIterWithHistory(ctx, s.Standalone, s.History, s.Evidence)
		} else {
			sources, tokens = g.GenerateIter(ctx, s.Standalone, s.Evidence)
		}
		var text strings.Builder
		for tok, err := range tokens {
			if err != nil {
				return err
			}
			text.WriteString(tok)
			if s.onToken != nil {
				if err := s.onToken(tok); err != nil {
					return err
				}
			}
		}
		s.Answer = s.answer(text.String(), sources)
		return nil
	})
}

// DefaultAbstainMessage is what Abstain answers when given no message.
const DefaultAbstainMessage = "I couldn't find enough information in the documents to answer that."

// Abstain answers that the documents do not hold the answer, instead of guessing.
// It sets State.Abstained and streams the message to WithTokens.
func Abstain(message string) Step {
	if message == "" {
		message = DefaultAbstainMessage
	}
	return StepFunc(func(_ context.Context, s *State) error {
		s.Abstained = true
		s.Answer = s.answer(message, nil)
		if s.onToken != nil {
			return s.onToken(message)
		}
		return nil
	})
}

func (s *State) answer(text string, sources []ragout.ScoredChunk) ragout.Answer {
	return ragout.Answer{Text: text, Sources: sources, Retrieved: s.Evidence, Question: s.Standalone, Searches: s.Searches}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return strings.ToValidUTF8(s[:n], "")
}
