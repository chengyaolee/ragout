package generator

import (
	"testing"

	"github.com/pkoukk/tiktoken-go"
)

// TestTiktokenCounter_MatchesTiktoken checks TiktokenCounter against the same
// cl100k_base encoding chunker/token.go uses, so the generator's prompt budget and
// the ingestion-time chunker agree on what a "token" is.
func TestTiktokenCounter_MatchesTiktoken(t *testing.T) {
	text := "Channels prevent race conditions by transferring ownership of values between goroutines."

	enc, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		t.Fatal(err)
	}
	want := len(enc.Encode(text, nil, nil))

	counter, err := NewTiktokenCounter("cl100k_base")
	if err != nil {
		t.Fatal(err)
	}
	if got := counter.CountTokens(text); got != want {
		t.Fatalf("CountTokens = %d, want %d", got, want)
	}
}

func TestTiktokenCounter_Empty(t *testing.T) {
	counter, err := NewTiktokenCounter("cl100k_base")
	if err != nil {
		t.Fatal(err)
	}
	if got := counter.CountTokens(""); got != 0 {
		t.Fatalf("CountTokens(\"\") = %d, want 0", got)
	}
}

func TestNewContextBudgeter_DefaultsToTiktoken(t *testing.T) {
	b := NewContextBudgeter(nil, 100)
	if _, ok := b.counter.(*TiktokenCounter); !ok {
		t.Fatalf("default counter = %T, want *TiktokenCounter", b.counter)
	}
}
