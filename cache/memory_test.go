package cache_test

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	"github.com/chengyaolee/ragout"
	"github.com/chengyaolee/ragout/cache"
)

func TestMemory(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx := context.Background()
		m := cache.NewMemory(2)
		has := func(key string) bool { _, ok, _ := m.Get(ctx, key); return ok }

		m.Set(ctx, "a", ragout.Answer{Question: "qa"}, time.Hour)
		m.Set(ctx, "forever", ragout.Answer{}, 0)
		if got, ok, _ := m.Get(ctx, "a"); !ok || got.Question != "qa" {
			t.Fatalf("Get(a) = %v, %v", got, ok)
		}

		// Reading does not extend an entry's ttl; ttl 0 never expires.
		time.Sleep(59 * time.Minute)
		if !has("a") {
			t.Fatal("a expired early")
		}
		time.Sleep(1000 * time.Hour)
		if has("a") {
			t.Fatal("a outlived its ttl")
		}
		if !has("forever") {
			t.Fatal("an entry with ttl 0 expired")
		}

		// Past maxEntries the oldest write goes, and rewriting a key makes it the newest.
		m.Set(ctx, "b", ragout.Answer{}, 0)
		m.Set(ctx, "forever", ragout.Answer{}, 0)
		m.Set(ctx, "c", ragout.Answer{}, 0)
		if has("b") || !has("forever") || !has("c") {
			t.Fatalf("after eviction: b=%v forever=%v c=%v, want false true true", has("b"), has("forever"), has("c"))
		}
	})
}
