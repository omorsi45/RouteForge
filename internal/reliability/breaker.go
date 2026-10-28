// Package reliability keeps the gateway standing when providers do not:
// per-model circuit breakers, a retry budget, jittered backoff, bounded
// per-model queues and a global in-flight limit.
//
// Every time-based type reads an injected clock, so tests drive it by hand.
package reliability

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

// State is the state of a circuit breaker. The values are what the
// breaker_state metric reports.
type State int

// Breaker states.
const (
	StateClosed   State = 0 // requests flow
	StateHalfOpen State = 1 // a few probes test the model
	StateOpen     State = 2 // requests are refused
)

// BreakerConfig sets when a breaker opens and for how long.
type BreakerConfig struct {
	// ConsecutiveFailures failures in a row open the breaker.
	ConsecutiveFailures int
	// ErrorRate opens it too, once MinRequests requests in the last Window
	// failed at this rate or more.
	ErrorRate   float64
	MinRequests int
	Window      time.Duration
	// OpenFor is how long an open breaker refuses requests before it lets
	// probes through.
	OpenFor time.Duration
	// HalfOpenProbes is how many probes a half-open breaker admits at once.
	HalfOpenProbes int
}

// CountsAsFailure reports whether err says the model is unhealthy. Unavailable,
// timed out, rate limited and internal failures do. A nil error does not, nor
// does a canceled call (the caller left, not the model), nor does a request the
// model rejected on its merits: bad_request, auth and context_length say
// something about the request or the key, not about the model's health.
func CountsAsFailure(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	switch provider.KindOf(err) {
	case provider.KindUnavailable, provider.KindTimeout, provider.KindRateLimited, provider.KindInternal:
		return true
	}
	return false
}

// Breakers holds one circuit breaker per model, created on first use. It is
// safe for concurrent use.
type Breakers struct {
	cfg BreakerConfig
	now func() time.Time

	mu      sync.RWMutex
	byModel map[string]*breaker
}

// NewBreakers returns breakers that open as cfg says and read time from now. A
// nil now uses time.Now.
func NewBreakers(cfg BreakerConfig, now func() time.Time) *Breakers {
	if now == nil {
		now = time.Now
	}
	return &Breakers{cfg: cfg, now: now, byModel: map[string]*breaker{}}
}

// Allow asks to send a request to model. When it reports true the caller must
// call Done on the ticket exactly once with the outcome. A closed breaker
// admits everything, an open one nothing, and a half-open one up to
// HalfOpenProbes requests at a time.
func (b *Breakers) Allow(model string) (Ticket, bool) {
	return b.breaker(model).allow()
}

// State returns the state of model's breaker. An open breaker whose OpenFor has
// elapsed is already half-open. A model never used is closed.
func (b *Breakers) State(model string) State {
	b.mu.RLock()
	br := b.byModel[model]
	b.mu.RUnlock()
	if br == nil {
		return StateClosed
	}
	return br.current()
}

// IsOpen reports whether model's breaker is open, the routing signal for
// eligibility. A half-open breaker is not open: it needs probes to heal, and
// only a request routed to it can be one.
func (b *Breakers) IsOpen(model string) bool { return b.State(model) == StateOpen }

// breaker returns the breaker of model, creating it on first use.
func (b *Breakers) breaker(model string) *breaker {
	b.mu.RLock()
	br := b.byModel[model]
	b.mu.RUnlock()
	if br != nil {
		return br
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if br = b.byModel[model]; br == nil {
		br = &breaker{cfg: b.cfg, now: b.now, win: newWindow(b.cfg.Window)}
		b.byModel[model] = br
	}
	return br
}

// Ticket is permission to send one request. The zero Ticket, which Allow
// returns when it refuses, ignores Done.
type Ticket struct {
	b     *breaker
	epoch uint64 // the breaker epoch the ticket was issued in
	probe bool   // issued by a half-open breaker
}

// Done reports the outcome of the request: true for a success. A caller with an
// error passes !CountsAsFailure(err), except for a canceled call, which has no
// outcome to report and ends with Cancel instead. An outcome from before the
// breaker last changed state is ignored, since it describes a model that has
// since been judged again.
func (t Ticket) Done(success bool) {
	if t.b != nil {
		t.b.done(t, success)
	}
}

// Cancel gives the ticket back without a verdict, for a request that ended
// before it said anything about the model: the client left, or the request was
// never sent. A half-open breaker gets its probe slot back; a closed breaker
// records nothing. Like Done, call it exactly once, and not after Done.
func (t Ticket) Cancel() {
	if t.b != nil {
		t.b.cancel(t)
	}
}

// breaker is the circuit breaker of one model.
type breaker struct {
	cfg BreakerConfig
	now func() time.Time

	mu          sync.Mutex
	state       State
	epoch       uint64    // bumped on every state change
	openedAt    time.Time // when the breaker last opened
	probes      int       // probes in flight while half-open
	consecutive int       // failures in a row while closed
	win         window    // requests and failures while closed
}

func (b *breaker) allow() (Ticket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refresh(b.now())
	switch b.state {
	case StateOpen:
		return Ticket{}, false
	case StateHalfOpen:
		if b.probes >= b.cfg.HalfOpenProbes {
			return Ticket{}, false
		}
		b.probes++
		return Ticket{b: b, epoch: b.epoch, probe: true}, true
	}
	return Ticket{b: b, epoch: b.epoch}, true
}

func (b *breaker) current() State {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refresh(b.now())
	return b.state
}

func (b *breaker) done(t Ticket, success bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if t.epoch != b.epoch {
		return
	}
	now := b.now()
	// The epoch is unchanged, so the state is the one the ticket was issued in.
	if t.probe {
		if success {
			b.close()
		} else {
			b.open(now)
		}
		return
	}
	b.record(now, success)
}

// cancel frees the probe slot of a half-open ticket. A ticket from an older
// epoch holds no slot of the current one: the state change reset the probe
// count.
func (b *breaker) cancel(t Ticket) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if t.probe && t.epoch == b.epoch && b.probes > 0 {
		b.probes--
	}
}

// refresh turns an open breaker whose OpenFor has elapsed into a half-open one.
func (b *breaker) refresh(now time.Time) {
	if b.state == StateOpen && now.Sub(b.openedAt) >= b.cfg.OpenFor {
		b.transition(StateHalfOpen)
	}
}

// record counts the outcome of a request admitted while closed and opens the
// breaker when it crosses a threshold. Both thresholds are checked on every
// outcome: a success can be the request that reaches MinRequests.
func (b *breaker) record(now time.Time, success bool) {
	// A streak is only as old as the window: when every request has left it,
	// the failures that made up the streak have gone with them.
	if requests, _ := b.win.totals(now); requests == 0 {
		b.consecutive = 0
	}
	failed := 0
	if !success {
		failed = 1
	}
	b.win.add(now, 1, failed)
	if success {
		b.consecutive = 0
	} else {
		b.consecutive++
	}
	if b.consecutive >= b.cfg.ConsecutiveFailures {
		b.open(now)
		return
	}
	requests, failures := b.win.totals(now)
	if requests >= b.cfg.MinRequests && float64(failures)/float64(requests) >= b.cfg.ErrorRate {
		b.open(now)
	}
}

func (b *breaker) open(now time.Time) {
	b.transition(StateOpen)
	b.openedAt = now
}

// close returns the breaker to normal with a clean slate.
func (b *breaker) close() {
	b.transition(StateClosed)
	b.consecutive = 0
	b.win.reset()
}

func (b *breaker) transition(s State) {
	b.state = s
	b.epoch++
	b.probes = 0
}

// window counts requests and events (failures, or granted retries) in one
// second buckets over a sliding span. It is not safe for concurrent use.
type window struct {
	buckets []bucket
}

type bucket struct {
	sec      int64 // the Unix second the bucket counts
	requests int
	events   int
}

// newWindow returns a window of span, rounded up to whole seconds.
func newWindow(span time.Duration) window {
	n := max(int((span+time.Second-1)/time.Second), 1)
	return window{buckets: make([]bucket, n)}
}

// add counts requests and events at now.
func (w *window) add(now time.Time, requests, events int) {
	sec := now.Unix()
	n := int64(len(w.buckets))
	b := &w.buckets[(sec%n+n)%n]
	if b.sec != sec {
		*b = bucket{sec: sec}
	}
	b.requests += requests
	b.events += events
}

// totals returns the requests and events of the buckets still inside the span
// that ends at now.
func (w *window) totals(now time.Time) (requests, events int) {
	sec := now.Unix()
	for _, b := range w.buckets {
		if b.sec <= sec && sec-b.sec < int64(len(w.buckets)) {
			requests += b.requests
			events += b.events
		}
	}
	return requests, events
}

func (w *window) reset() { clear(w.buckets) }
