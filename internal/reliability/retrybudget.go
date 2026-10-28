package reliability

import (
	"math"
	"sync"
	"time"
)

// RetryBudget caps how many retries the gateway adds on top of its traffic, so
// that failing providers are not hit harder just when they struggle. Within a
// sliding window it allows floor(ratio * requests) + a fixed floor retries; the
// floor (Ruling R5) lets an isolated request still fail over. It is safe for
// concurrent use.
type RetryBudget struct {
	ratio      float64
	minRetries int
	now        func() time.Time

	mu  sync.Mutex
	win window // requests, and the retries granted
}

// NewRetryBudget returns a budget of ratio retries per request over span, plus
// minRetries. A nil now uses time.Now.
func NewRetryBudget(ratio float64, span time.Duration, minRetries int, now func() time.Time) *RetryBudget {
	if now == nil {
		now = time.Now
	}
	return &RetryBudget{ratio: ratio, minRetries: minRetries, now: now, win: newWindow(span)}
}

// RecordRequest counts one client request, however many attempts it takes.
func (r *RetryBudget) RecordRequest() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.win.add(r.now(), 1, 0)
}

// TryRetry asks for one retry. It grants it, and counts it against the window,
// while the retries already granted number fewer than the budget allows.
func (r *RetryBudget) TryRetry() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now()
	requests, retries := r.win.totals(now)
	if retries >= int(math.Floor(r.ratio*float64(requests)))+r.minRetries {
		return false
	}
	r.win.add(now, 0, 1)
	return true
}
