// Package executor runs a routing decision against its upstream providers.
//
// A request makes up to MaxAttempts calls. The targets are the decision's
// chosen model followed by its fallbacks; attempt i goes to targets[min(i,
// len-1)], so a failure fails over down the list and a pinned model is retried
// itself. A target whose circuit breaker is open, or whose queue is full or
// times out, is skipped for the next one without costing an attempt.
//
// A retry needs a retryable failure (rate_limited, unavailable, timeout), a
// client that is still there, a grant from the retry budget and a backoff that
// fits in the time left. A stream is retried only while it is being opened and
// its first event read: once a byte has been delivered the attempt is final.
//
// Each attempt is metered through an optional Meter, which accounting uses to
// reserve before the call and settle after it.
package executor

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/health"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
)

// ErrNoTargetAccepted means every target was skipped, by an open breaker or a
// full or timed-out queue, so no upstream call was made. The API maps it to
// 503 overloaded.
var ErrNoTargetAccepted = errors.New("no model accepted the request")

// Meter charges the attempts of a request. It is the seam accounting plugs
// into; a nil Meter leaves a call unmetered.
type Meter interface {
	// Begin is called before each attempt, once a model has accepted it, with
	// the configured model name. An error means the attempt must not be made:
	// on the first attempt the executor returns it, on a later one it stops
	// retrying and returns the last upstream error.
	Begin(ctx context.Context, model string) (Charge, error)
}

// Charge is one attempt's open charge and Settlement says what the attempt did.
// Accounting defines both, because it cannot import this package (the executor
// imports the router, which imports accounting).
type (
	Charge     = accounting.Charge
	Settlement = accounting.Settlement
)

// Target binds a configured model name to its provider and upstream model id.
type Target struct {
	Model    string // configured model name
	Upstream string // provider model id sent as Request.Model
	Provider provider.Provider
}

// Result is a completed non-streaming call.
type Result struct {
	Model    string // served configured model name
	Response *provider.Response
}

// Options configures an Executor. A nil field disables its feature.
type Options struct {
	// Targets are the callable models, keyed by configured name.
	Targets map[string]Target
	// Now is the clock for attempt timings; nil uses time.Now. Request
	// deadlines come from the context, which runs on the real clock.
	Now func() time.Time

	Breakers    *reliability.Breakers
	Queues      *reliability.Queues
	RetryBudget *reliability.RetryBudget
	Backoff     reliability.Backoff
	Health      *health.Registry

	// MaxAttempts is the most upstream calls per request; 0 means 1.
	MaxAttempts int
	// Sleep waits for d or until ctx ends, returning ctx's error then. Nil
	// waits on a timer.
	Sleep func(ctx context.Context, d time.Duration) error
}

// Executor calls providers on behalf of routing decisions. It is safe for
// concurrent use.
type Executor struct {
	targets     map[string]Target
	now         func() time.Time
	breakers    *reliability.Breakers
	queues      *reliability.Queues
	retryBudget *reliability.RetryBudget
	backoff     reliability.Backoff
	health      *health.Registry
	maxAttempts int
	sleep       func(context.Context, time.Duration) error
}

// New returns an Executor over targets keyed by configured model name that
// makes a single attempt with no breakers, queues or health tracking. A nil now
// uses time.Now.
func New(targets map[string]Target, now func() time.Time) *Executor {
	return NewWithOptions(Options{Targets: targets, Now: now})
}

// NewWithOptions returns an Executor configured by o.
func NewWithOptions(o Options) *Executor {
	e := &Executor{
		targets:     o.Targets,
		now:         o.Now,
		breakers:    o.Breakers,
		queues:      o.Queues,
		retryBudget: o.RetryBudget,
		backoff:     o.Backoff,
		health:      o.Health,
		maxAttempts: max(o.MaxAttempts, 1),
		sleep:       o.Sleep,
	}
	if e.now == nil {
		e.now = time.Now
	}
	if e.sleep == nil {
		e.sleep = sleepContext
	}
	return e
}

// sleepContext waits for d, or until ctx ends.
func sleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Complete makes a non-streaming call for d, retrying and failing over as
// described in the package comment, and records each attempt on d. m meters
// the attempts; nil leaves them unmetered.
func (e *Executor) Complete(ctx context.Context, d *router.Decision, req provider.Request, m Meter) (*Result, error) {
	var res *Result
	err := e.run(ctx, d, m, func(a *attempt) error {
		req.Model = a.target.Upstream
		resp, err := a.target.Provider.Complete(ctx, req)
		if err != nil {
			return err
		}
		res = &Result{Model: a.target.Model, Response: resp}
		a.end(ending{outcome: router.OutcomeOK, usage: resp.Usage, delivered: len(resp.Content), observe: true})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Stream opens a streaming call for d and reads its first event before
// returning, so a failure before the first event is a returned error, and is
// retried, while nothing has reached the client. The returned Stream owns the
// attempt: it records it, settles its charge and frees its queue slot and
// breaker ticket when it ends or is closed.
func (e *Executor) Stream(ctx context.Context, d *router.Decision, req provider.Request, m Meter) (*Stream, error) {
	var s *Stream
	err := e.run(ctx, d, m, func(a *attempt) error {
		req.Model = a.target.Upstream
		inner, err := a.target.Provider.Stream(ctx, req)
		if err != nil {
			return err
		}
		primed := false
		defer func() {
			if !primed { // a failed first read, or a panic
				_ = inner.Close()
			}
		}()
		ev, err := inner.Recv()
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		primed = true
		s = &Stream{
			Model:     a.target.Model,
			a:         a,
			inner:     inner,
			hasPrimed: true,
			primedEv:  ev,
			primedErr: err,
		}
		if err == nil {
			s.ttft, s.hasTTFT = e.now().Sub(a.start), true
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

// run drives the attempt loop. try performs the upstream call of one admitted
// attempt: on success it must end the attempt or hand it to a Stream; on
// failure it returns the error and run ends the attempt.
func (e *Executor) run(ctx context.Context, d *router.Decision, m Meter, try func(*attempt) error) error {
	targets, err := e.resolve(d)
	if err != nil {
		return err
	}
	if e.retryBudget != nil {
		e.retryBudget.RecordRequest()
	}

	var (
		calls int   // upstream calls made
		pos   int   // the next target to try
		prev  error // the error of the last upstream call
	)
	for calls < e.maxAttempts {
		if calls > 0 && !e.mayRetry(ctx, calls, prev) {
			return prev
		}
		a, err := e.admit(ctx, d, m, targets, &pos, prev)
		if err != nil {
			return err
		}
		calls++
		if err = a.run(try); err == nil {
			return nil
		}
		prev = err
		pos = min(pos+1, len(targets)-1)
	}
	return prev
}

// resolve returns the targets of d in order: the chosen model, then the
// fallbacks. A model without a target is an internal error before any call.
func (e *Executor) resolve(d *router.Decision) ([]Target, error) {
	names := make([]string, 0, 1+len(d.Fallbacks))
	names = append(append(names, d.Chosen), d.Fallbacks...)
	targets := make([]Target, len(names))
	for i, name := range names {
		t, ok := e.targets[name]
		if !ok {
			return nil, &provider.Error{Kind: provider.KindInternal, Message: "no target for model " + name}
		}
		targets[i] = t
	}
	return targets, nil
}

// mayRetry decides whether attempt number retry (counted from 1) goes ahead
// after prev failed, and if so waits out the backoff. The delay is the larger
// of the jittered backoff and the provider's Retry-After; one that does not fit
// in the time left stops the request instead of burning it asleep.
func (e *Executor) mayRetry(ctx context.Context, retry int, prev error) bool {
	if !provider.KindOf(prev).Retryable() || ctx.Err() != nil {
		return false
	}
	delay := e.backoff.Delay(retry)
	var pe *provider.Error
	if errors.As(prev, &pe) {
		delay = max(delay, pe.RetryAfter)
	}
	// Checked before the budget is asked, so a retry that cannot happen costs
	// none of it. The context deadline is on the real clock.
	if deadline, ok := ctx.Deadline(); ok && delay >= time.Until(deadline) {
		return false
	}
	if e.retryBudget != nil && !e.retryBudget.TryRetry() {
		return false
	}
	return e.sleep(ctx, delay) == nil
}

// admit finds the first target from *pos on that accepts the request: its
// breaker allows it, a queue slot is free in time and the meter agrees. Targets
// that refuse are skipped, which is not an attempt. When none accepts it
// returns prev, the last upstream error, or ErrNoTargetAccepted if no call was
// made. prev is also what a meter refusal returns after a first attempt; a
// context that ends in the queue returns its own error.
func (e *Executor) admit(ctx context.Context, d *router.Decision, m Meter, targets []Target, pos *int, prev error) (*attempt, error) {
	for ; *pos < len(targets); *pos++ {
		t := targets[*pos]

		var ticket reliability.Ticket
		if e.breakers != nil {
			var allowed bool
			if ticket, allowed = e.breakers.Allow(t.Model); !allowed {
				continue
			}
		}

		release := func() {}
		if e.queues != nil {
			rel, err := e.queues.Acquire(ctx, t.Model)
			if err != nil {
				ticket.Cancel() // no call was made, so the breaker has no verdict
				if errors.Is(err, reliability.ErrQueueFull) || errors.Is(err, reliability.ErrQueueTimeout) {
					continue
				}
				return nil, fmt.Errorf("wait for model %s: %w", t.Model, err)
			}
			release = rel
		}

		var charge Charge
		if m != nil {
			c, err := m.Begin(ctx, t.Model)
			if err != nil {
				release()
				ticket.Cancel()
				if prev != nil {
					return nil, prev
				}
				return nil, err
			}
			charge = c
		}
		return &attempt{e: e, d: d, target: t, ticket: ticket, release: release, charge: charge, start: e.now()}, nil
	}
	if prev != nil {
		return nil, prev
	}
	return nil, ErrNoTargetAccepted
}

func (e *Executor) record(d *router.Decision, model, outcome string, kind provider.ErrorKind, dur time.Duration) {
	d.Attempts = append(d.Attempts, router.Attempt{
		Model:      model,
		Outcome:    outcome,
		ErrorKind:  string(kind),
		DurationMS: float64(dur.Microseconds()) / 1000,
	})
}

// classify maps a failed call to an attempt outcome.
func classify(err error) string {
	if errors.Is(err, context.Canceled) {
		return router.OutcomeCanceled
	}
	return router.OutcomeError
}

// outputTokens is the number of completion tokens an attempt produced: what the
// provider reported, else the spec 11 estimate from the bytes delivered, so a
// provider that sends no usage does not make the health estimators charge a
// whole response's decode time to one token.
func outputTokens(u *provider.Usage, deliveredBytes int) int {
	if u != nil {
		return u.CompletionTokens
	}
	return accounting.EstimateTokens(deliveredBytes)
}

// attempt is one admitted upstream call and everything it holds: the model's
// queue slot, its breaker ticket and its charge. end gives them all back.
type attempt struct {
	e      *Executor
	d      *router.Decision
	target Target
	start  time.Time

	ticket  reliability.Ticket
	release func()
	charge  Charge // nil when unmetered

	once sync.Once
}

// ending says how an attempt finished.
type ending struct {
	outcome   string // a router.Outcome* value
	err       error  // the failure, nil for ok and for a close before the end
	usage     *provider.Usage
	delivered int

	// observe reports the timing of a successful attempt to the health
	// estimators; only ok endings set it. A stream sets streamed and ttft, the
	// time to its first event.
	observe  bool
	streamed bool
	ttft     time.Duration
}

// run makes the upstream call through try and ends the attempt when it fails.
// If try panics the attempt ends as an internal failure, so the charge,
// breaker ticket and queue slot are never leaked, and the panic goes on.
func (a *attempt) run(try func(*attempt) error) error {
	returned := false
	defer func() {
		if !returned {
			a.end(ending{
				outcome: router.OutcomeError,
				err:     &provider.Error{Kind: provider.KindInternal, Message: "provider call panicked"},
			})
		}
	}()
	err := try(a)
	returned = true
	if err != nil {
		a.end(ending{outcome: classify(err), err: err})
	}
	return err
}

// end finishes the attempt exactly once: it records it on the decision, tells
// the breaker, feeds the health estimators, settles the charge and frees the
// queue slot. Later calls do nothing.
func (a *attempt) end(r ending) {
	a.once.Do(func() {
		defer a.release()

		dur := a.e.now().Sub(a.start)
		var kind provider.ErrorKind
		if r.err != nil && r.outcome != router.OutcomeCanceled {
			kind = provider.KindOf(r.err)
		}
		a.e.record(a.d, a.target.Model, r.outcome, kind, dur)

		switch {
		case r.outcome == router.OutcomeOK:
			a.ticket.Done(true)
		case r.err == nil || errors.Is(r.err, context.Canceled):
			a.ticket.Cancel() // the caller left: no verdict on the model
		default:
			a.ticket.Done(!reliability.CountsAsFailure(r.err))
		}

		if r.observe && a.e.health != nil {
			a.e.health.Observe(a.target.Model, health.Observation{
				TTFT:     r.ttft,
				Total:    dur,
				Tokens:   outputTokens(r.usage, r.delivered),
				Streamed: r.streamed,
			})
		}

		if a.charge != nil {
			a.charge.End(Settlement{Outcome: r.outcome, Usage: r.usage, DeliveredBytes: r.delivered})
		}
	})
}

// Stream is a primed provider stream. Recv must be called from one goroutine
// at a time; Close may be called from any goroutine.
type Stream struct {
	Model string // served configured model name

	a     *attempt
	inner provider.Stream

	// Written before the Stream is returned, then used only by Recv.
	hasPrimed bool
	primedEv  provider.StreamEvent
	primedErr error
	ttft      time.Duration // time from the attempt's start to the first event
	hasTTFT   bool          // false for a stream that ended without an event

	// What the caller has received so far, read when the attempt ends.
	delivered atomic.Int64
	usage     atomic.Pointer[provider.Usage]

	closeOnce sync.Once
	closeErr  error
}

// Recv returns the next event: the primed one first, then the provider's.
func (s *Stream) Recv() (provider.StreamEvent, error) {
	var (
		ev  provider.StreamEvent
		err error
	)
	if s.hasPrimed {
		s.hasPrimed = false
		ev, err = s.primedEv, s.primedErr
	} else {
		ev, err = s.inner.Recv()
	}
	if err == nil {
		s.delivered.Add(int64(len(ev.Delta)))
		if ev.Usage != nil {
			u := *ev.Usage
			s.usage.Store(&u)
		}
		return ev, nil
	}
	switch {
	case errors.Is(err, io.EOF):
		s.end(router.OutcomeOK, nil)
	case errors.Is(err, context.Canceled):
		s.end(router.OutcomeCanceled, err)
	default:
		s.end(router.OutcomePartial, err)
	}
	return ev, err
}

// Close ends the stream. Closing before the end records the attempt as
// canceled. It is idempotent and safe to call concurrently with Recv.
func (s *Stream) Close() error {
	// Record before closing the provider stream so a Recv failure caused by
	// this Close cannot claim the attempt first.
	s.end(router.OutcomeCanceled, nil)
	s.closeOnce.Do(func() { s.closeErr = s.inner.Close() })
	return s.closeErr
}

// end ends the attempt, once, with what the caller has received so far.
func (s *Stream) end(outcome string, err error) {
	r := ending{
		outcome:   outcome,
		err:       err,
		usage:     s.usage.Load(),
		delivered: int(s.delivered.Load()),
	}
	if outcome == router.OutcomeOK {
		// A stream with no first event has no TTFT, so there is nothing to time.
		r.observe, r.streamed, r.ttft = s.hasTTFT, true, s.ttft
	}
	s.a.end(r)
}
