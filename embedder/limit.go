package embedder

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/chengyaolee/ragout"
	"golang.org/x/time/rate"
)

type RateLimiter struct {
	inner   ragout.Embedder
	limit   *rate.Limiter
	attemps int
	base    time.Duration
	max     time.Duration
}

func NewRateLimiter(inner ragout.Embedder, limit *rate.Limiter, attemps int, base time.Duration, max time.Duration) (*RateLimiter, error) {
	if inner == nil {
		return nil, fmt.Errorf("inner embedder is nil")
	}
	if limit == nil {
		return nil, fmt.Errorf("limiter is nil")
	}
	if attemps <= 0 {
		return nil, fmt.Errorf("attempts must be >= 1")
	}
	if base <= 0 || max < base {
		return nil, fmt.Errorf("backoff base must be > 0 and max >= base")
	}
	return &RateLimiter{
		inner:   inner,
		limit:   limit,
		attemps: attemps,
		base:    base,
		max:     max,
	}, nil
}

func (l *RateLimiter) Dimension() int {
	return l.inner.Dimension()
}

func (l *RateLimiter) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	// Last error to return
	var last error

	for attempt := 0; attempt < l.attemps; attempt++ {
		// Wait for the rate limit
		if err := l.limit.Wait(ctx); err != nil {
			return nil, err
		}
		embeddings, err := l.inner.EmbedBatch(ctx, texts)
		if err == nil {
			return embeddings, nil
		}
		// If error is not retryable or this is the last attempt, return the error
		if !retryable(err) || attempt == l.attemps-1 {
			return nil, err
		}
		// Last error to return
		last = err
		// Sleep for the backoff duration (Exponential backoff with full jitter)
		// Jitter prevents thundering herd problem when upstream provider recovers
		if err := sleep(ctx, fullJitter(attempt, l.base, l.max)); err != nil {
			return nil, err
		}
	}
	return nil, last
}

// fullJitter returns a random backoff duration (with jitter) based on the attempt number.
// It uses exponential backoff with full jitter, doubling the base up to the max.
func fullJitter(attempt int, base time.Duration, max time.Duration) time.Duration {
	capacity := base
	// Exponentially increase capacity for each attempt (without exceeding max)
	for range attempt {
		if capacity > max/2 {
			capacity = max
			break
		}
		capacity *= 2
	}
	// Ensure capacity does not exceed max
	if capacity > max {
		capacity = max
	}
	// If capacity is not positive, return 0
	if capacity <= 0 {
		return 0
	}
	// Return a random duration between 0 and capacity (full jitter)
	return time.Duration(rand.Int64N(int64(capacity)))
}

// Sleep function that can exit early if the context is cancelled.
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 { // No delay needed
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done(): // Context cancelled/timed out
		return ctx.Err()
	case <-t.C: // Timer completed
		return nil
	}
}

// temporaryError is an error wrapper type that indicates the error is temporary.
// It implements the interface { Temporary() bool } with Temporary() always returning true.
type temporaryError struct{ error }

func (e temporaryError) Temporary() bool { return true }

// retryable determines if an error is safe to retry.
// Returns false for context cancellations, timeouts, and fundamental embedding errors.
// Returns true if the error implements Temporary() bool and it is temporary (expected to recover).
func retryable(err error) bool {
	// Not retryable: context canceled or deadline exceeded
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	// Not retryable: embedding-specific permanent errors
	if errors.Is(err, ragout.ErrCountMismatch) ||
		errors.Is(err, ragout.ErrDimensionMismatch) ||
		errors.Is(err, ragout.ErrEmptyEmbeddings) {
		return false
	}
	// Retryable: error implements Temporary() and returns true
	var t interface{ Temporary() bool }
	return errors.As(err, &t) && t.Temporary()
}

// // Wait allows LimiterFunc to satisfy an interface requiring a Wait method.
// // It simply calls the underlying function with the provided context.
// func (f LimiterFunc) Wait(ctx context.Context) error {
// 	return f(ctx)
// }
