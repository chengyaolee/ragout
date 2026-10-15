// Package transform implements ragout.QueryTransformer: rewriting a question before
// retrieval. Each transformer makes one Completer call; combine them with Chain.
//
// Model output is read as plain lines, never JSON, so any chat model works: <think>
// blocks, list numbering, bullets, quotes and preamble lines are ignored.
package transform

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/chengyaolee/ragout"
)

const (
	historyTurns    = 6    // most recent turns Condense sends
	historyTurnSize = 1000 // bytes kept from each turn
	passageSize     = 2000 // bytes kept from a HyDE passage
)

// Condense rewrites a follow-up into a standalone question using the conversation
// passed with ragout.WithHistory. With no history it makes no call.
func Condense(c ragout.Completer) ragout.QueryTransformer { return condense{c} }

// MultiQuery adds up to n rephrasings of the question, each searched on every index.
// n <= 0 means 3.
func MultiQuery(c ragout.Completer, n int) ragout.QueryTransformer {
	if n <= 0 {
		n = 3
	}
	return multiQuery{c, n}
}

// Decompose splits a compound question into up to n simpler sub-questions, each
// searched on every index. A question about one thing adds nothing. n <= 0 means 4.
func Decompose(c ragout.Completer, n int) ragout.QueryTransformer {
	if n <= 0 {
		n = 4
	}
	return decompose{c, n}
}

// HyDE adds a hypothetical answer passage, searched on the vector store only
// (Hypothetical Document Embeddings). Its gains are smaller with strong modern
// embedders, and it can mislead on private data the model does not know.
func HyDE(c ragout.Completer) ragout.QueryTransformer { return hyde{c} }

// Chain runs transformers in order. Each sees the standalone question produced so far;
// their searches accumulate. It stops at the first error, returning what it has.
func Chain(ts ...ragout.QueryTransformer) ragout.QueryTransformer { return chain(ts) }

type condense struct{ c ragout.Completer }

func (t condense) Transform(ctx context.Context, question string, history []ragout.Turn) (ragout.Rewrite, error) {
	if len(history) == 0 {
		return ragout.Rewrite{}, nil
	}
	var b strings.Builder
	for _, turn := range history[max(0, len(history)-historyTurns):] {
		fmt.Fprintf(&b, "%s: %s\n", turn.Role, truncate(strings.TrimSpace(turn.Content), historyTurnSize))
	}
	reply, err := t.c.Complete(ctx, "", fmt.Sprintf(`Given the conversation and a follow-up question, rewrite the follow-up as a standalone question that can be understood without the conversation. Keep any names, numbers and dates from the conversation that the follow-up refers to. If it is already standalone, repeat it unchanged. Output only the question.

<conversation>
%s</conversation>
Follow-up question: %s`, b.String(), question))
	if err != nil {
		return ragout.Rewrite{}, err
	}
	if q := lines(reply, 1, ""); len(q) == 1 {
		return ragout.Rewrite{Question: q[0]}, nil
	}
	return ragout.Rewrite{}, nil
}

type multiQuery struct {
	c ragout.Completer
	n int
}

func (t multiQuery) Transform(ctx context.Context, question string, _ []ragout.Turn) (ragout.Rewrite, error) {
	reply, err := t.c.Complete(ctx, "", fmt.Sprintf(`Write %d different search queries that would find documents answering the question below. Vary the wording, use synonyms and domain terms, and keep each query short. Output one query per line and nothing else.

Question: %s`, t.n, question))
	if err != nil {
		return ragout.Rewrite{}, err
	}
	return ragout.Rewrite{Searches: searches(lines(reply, t.n, question), ragout.SearchAll)}, nil
}

type decompose struct {
	c ragout.Completer
	n int
}

func (t decompose) Transform(ctx context.Context, question string, _ []ragout.Turn) (ragout.Rewrite, error) {
	reply, err := t.c.Complete(ctx, "", fmt.Sprintf(`Break the question below into the separate, simpler questions that must each be looked up to answer it, at most %d. If it asks about only one thing, output it unchanged. Output one question per line and nothing else.

Question: %s`, t.n, question))
	if err != nil {
		return ragout.Rewrite{}, err
	}
	return ragout.Rewrite{Searches: searches(lines(reply, t.n, question), ragout.SearchAll)}, nil
}

type hyde struct{ c ragout.Completer }

func (t hyde) Transform(ctx context.Context, question string, _ []ragout.Turn) (ragout.Rewrite, error) {
	reply, err := t.c.Complete(ctx, "", fmt.Sprintf(`Write a short passage (under 100 words) that answers the question below, written like an excerpt from a document that contains the answer. Output only the passage.

Question: %s`, question))
	if err != nil {
		return ragout.Rewrite{}, err
	}
	passage := truncate(strings.TrimSpace(thinkBlock.ReplaceAllString(reply, "")), passageSize)
	if passage == "" {
		return ragout.Rewrite{}, nil
	}
	return ragout.Rewrite{Searches: []ragout.Search{{Text: passage, Target: ragout.SearchDense}}}, nil
}

type chain []ragout.QueryTransformer

func (ts chain) Transform(ctx context.Context, question string, history []ragout.Turn) (ragout.Rewrite, error) {
	var out ragout.Rewrite
	for _, t := range ts {
		rw, err := t.Transform(ctx, question, history)
		if q := strings.TrimSpace(rw.Question); q != "" {
			question, out.Question = q, q
		}
		out.Searches = append(out.Searches, rw.Searches...)
		if err != nil {
			return out, err
		}
	}
	return out, nil
}

var (
	thinkBlock = regexp.MustCompile(`(?is)<think>.*?</think>`)
	listMarker = regexp.MustCompile(`^(?:\d+[.)]|[-*•])\s+`)
)

// lines extracts up to max distinct lines from a model reply, skipping reasoning
// blocks, list markers, surrounding quotes, preamble lines ending in ':' and any
// line equal to exclude.
func lines(reply string, max int, exclude string) []string {
	reply = thinkBlock.ReplaceAllString(reply, "")
	seen := map[string]bool{normalize(exclude): true}
	var out []string
	for _, line := range strings.Split(reply, "\n") {
		line = listMarker.ReplaceAllString(strings.TrimSpace(line), "")
		line = strings.TrimSpace(strings.Trim(line, "\"'`“”*"))
		key := normalize(line)
		if line == "" || strings.HasSuffix(line, ":") || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, line)
		if len(out) == max {
			break
		}
	}
	return out
}

func searches(texts []string, target ragout.SearchTarget) []ragout.Search {
	out := make([]ragout.Search, len(texts))
	for i, t := range texts {
		out[i] = ragout.Search{Text: t, Target: target}
	}
	return out
}

func normalize(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }

// truncate cuts s to at most n bytes without splitting a UTF-8 sequence.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return strings.ToValidUTF8(s[:n], "")
}
