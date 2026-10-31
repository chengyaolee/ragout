package generator

import (
	"iter"
	"strings"

	"github.com/chengyaolee/ragout"
)

func joinPrompt(prefix, prompt string) string {
	if prefix == "" {
		return prompt
	}
	return prefix + "\n\n" + prompt
}

// collect drains a token stream into one string.
func collect(tokens iter.Seq2[string, error]) (string, error) {
	var b strings.Builder
	for token, err := range tokens {
		if err != nil {
			return b.String(), err
		}
		b.WriteString(token)
	}
	return b.String(), nil
}

// lastTurns returns the final n turns of history.
func lastTurns(history []ragout.Turn, n int) []ragout.Turn {
	return history[max(0, len(history)-n):]
}
