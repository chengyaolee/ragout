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
)

// Grade asks a model whether State.Evidence answers State.Standalone, setting
// State.Verdict and, when something is missing, State.Missing. The reply is read as
// plain lines, so any chat model works.
//
// It fails open: if the model errors or the reply cannot be read, the verdict stays
// Ungraded, the error goes to State.Errors, and the run continues. A grader outage
// therefore degrades to a plain retrieve-and-answer, and stays visible.
func Grade(c ragout.Completer) Step {
	return StepFunc(func(ctx context.Context, s *State) error {
		s.Verdict, s.Missing = Ungraded, ""
		if len(s.Evidence) == 0 {
			s.Verdict = Insufficient
			return nil
		}

		var b strings.Builder
		for i, sc := range s.Evidence[:min(gradeChunks, len(s.Evidence))] {
			fmt.Fprintf(&b, "[%d] %s\n\n", i+1, truncate(sc.Chunk.IndexText(), gradeChunkBytes))
		}
		reply, err := c.Complete(ctx, "", fmt.Sprintf(gradePrompt, s.Standalone, b.String()))
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			s.Errors = append(s.Errors, fmt.Errorf("grade: %w", err))
			return nil
		}
		verdict, missing, ok := parseVerdict(reply)
		if !ok {
			s.Errors = append(s.Errors, fmt.Errorf("grade: unreadable reply %q", truncate(reply, 200)))
			return nil
		}
		s.Verdict, s.Missing = verdict, missing
		return nil
	})
}

const gradePrompt = `You check whether retrieved passages contain enough information to answer a question. Do not answer the question.

Question: %s

Passages:
%s
If the passages contain the information needed to answer the question, reply with the single word SUFFICIENT.
Otherwise reply with INSUFFICIENT on the first line, and on the second line a short search query for the missing information.`

var (
	thinkBlock = regexp.MustCompile(`(?is)<think>.*?</think>`)
	listMarker = regexp.MustCompile(`^(?:\d+[.)]|[-*•])\s+`)
	labelled   = regexp.MustCompile(`(?i)^(?:missing|search query|query|search)\s*:\s*`)
)

// parseVerdict reads a grader reply: the first line holds the verdict, the next
// non-empty line (if any) what to search for.
func parseVerdict(reply string) (Verdict, string, bool) {
	var lines []string
	for _, l := range strings.Split(thinkBlock.ReplaceAllString(reply, ""), "\n") {
		l = strings.TrimSpace(listMarker.ReplaceAllString(strings.TrimSpace(l), ""))
		if l = strings.Trim(l, "*`\"'“” "); l != "" {
			lines = append(lines, l)
		}
	}
	if len(lines) == 0 {
		return Ungraded, "", false
	}
	head := strings.ToUpper(lines[0])
	switch {
	case strings.Contains(head, "INSUFFICIENT"), strings.Contains(head, "NOT SUFFICIENT"):
		missing := ""
		if len(lines) > 1 {
			missing = strings.TrimSpace(labelled.ReplaceAllString(lines[1], ""))
		} else if _, rest, ok := strings.Cut(lines[0], ":"); ok {
			missing = strings.TrimSpace(rest)
		}
		return Insufficient, missing, true
	case strings.Contains(head, "SUFFICIENT"):
		return Sufficient, "", true
	}
	return Ungraded, "", false
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
