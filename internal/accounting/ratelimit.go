package accounting

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Rate limit kinds.
const (
	KindRPM = "rpm" // requests per minute
	KindTPM = "tpm" // tokens per minute
)

// RateDecision is the answer of a RateLimiter.
type RateDecision struct {
	OK bool
	// RetryAfter is, for a denial, how long until the bucket holds enough
	// tokens: whole seconds, at least one, and a minute for a request larger
	// than the bucket can ever hold.
	RetryAfter time.Duration
}

// RateLimiter is a set of token buckets, one per tenant and kind. A bucket
// holds perMinute tokens when full and refills at perMinute/60 tokens a
// second. A denial takes nothing.
type RateLimiter interface {
	Allow(ctx context.Context, tenant, kind string, n, perMinute int64) (RateDecision, error)
	// Ping checks that the limiter is reachable.
	Ping(ctx context.Context) error
}

type limiterKey struct{ tenant, kind string }

type bucket struct {
	tokens float64
	last   time.Time // when tokens was last brought up to date
}

// MemoryLimiter is a RateLimiter in process memory. It is safe for concurrent
// use.
type MemoryLimiter struct {
	now     func() time.Time
	mu      sync.Mutex
	buckets map[limiterKey]*bucket
}

// NewMemoryLimiter returns an empty MemoryLimiter that reads time from now.
func NewMemoryLimiter(now func() time.Time) *MemoryLimiter {
	return &MemoryLimiter{now: now, buckets: make(map[limiterKey]*bucket)}
}

// Allow implements RateLimiter.
func (l *MemoryLimiter) Allow(_ context.Context, tenant, kind string, n, perMinute int64) (RateDecision, error) {
	if perMinute <= 0 || n < 0 {
		return RateDecision{}, fmt.Errorf("rate limit %s for %q: need a positive rate and a non-negative amount, got n=%d perMinute=%d", kind, tenant, n, perMinute)
	}
	capacity := float64(perMinute)

	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	key := limiterKey{tenant, kind}
	b, ok := l.buckets[key]
	if !ok {
		b = &bucket{tokens: capacity, last: now}
		l.buckets[key] = b
	}
	// A clock that goes back refills nothing and is not followed back.
	if elapsed := now.Sub(b.last); elapsed > 0 {
		// Multiply before dividing so whole seconds of refill stay exact.
		b.tokens += float64(elapsed) * float64(perMinute) / float64(time.Minute)
		b.last = now
	}
	b.tokens = min(b.tokens, capacity)

	if n > perMinute {
		return RateDecision{RetryAfter: time.Minute}, nil
	}
	if need := float64(n); b.tokens >= need {
		b.tokens -= need
		return RateDecision{OK: true}, nil
	}
	secs := math.Ceil((float64(n) - b.tokens) * 60 / float64(perMinute))
	return RateDecision{RetryAfter: max(time.Duration(secs), 1) * time.Second}, nil
}

// Ping implements RateLimiter.
func (l *MemoryLimiter) Ping(context.Context) error { return nil }
