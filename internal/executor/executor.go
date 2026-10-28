// Package executor runs a routing decision against its upstream provider.
//
// This is the single-attempt executor: it calls d.Chosen once and records the
// attempt on the decision. Queueing, breakers, retries and failover will be
// added behind the same signatures.
package executor

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
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

// Executor calls providers on behalf of routing decisions.
type Executor struct {
	targets map[string]Target
	now     func() time.Time
}

// New returns an Executor over targets keyed by configured model name. A nil
// now uses time.Now.
func New(targets map[string]Target, now func() time.Time) *Executor {
	if now == nil {
		now = time.Now
	}
	return &Executor{targets: targets, now: now}
}

func (e *Executor) target(d *router.Decision) (Target, error) {
	t, ok := e.targets[d.Chosen]
	if !ok {
		return Target{}, &provider.Error{
			Kind:    provider.KindInternal,
			Message: "no target for model " + d.Chosen,
		}
	}
	return t, nil
}

// Complete makes one non-streaming call to d.Chosen and records the attempt.
func (e *Executor) Complete(ctx context.Context, d *router.Decision, req provider.Request) (*Result, error) {
	t, err := e.target(d)
	if err != nil {
		return nil, err
	}
	req.Model = t.Upstream
	start := e.now()
	resp, err := t.Provider.Complete(ctx, req)
	if err != nil {
		outcome, kind := classify(err)
		e.record(d, t.Model, outcome, kind, start)
		return nil, err
	}
	e.record(d, t.Model, router.OutcomeOK, "", start)
	return &Result{Model: t.Model, Response: resp}, nil
}

// Stream opens a streaming call to d.Chosen and reads its first event before
// returning, so a failure before the first event is a returned error and
// nothing has reached the client. The attempt is recorded when the returned
// stream ends or is closed.
func (e *Executor) Stream(ctx context.Context, d *router.Decision, req provider.Request) (*Stream, error) {
	t, err := e.target(d)
	if err != nil {
		return nil, err
	}
	req.Model = t.Upstream
	start := e.now()
	inner, err := t.Provider.Stream(ctx, req)
	if err != nil {
		outcome, kind := classify(err)
		e.record(d, t.Model, outcome, kind, start)
		return nil, err
	}
	ev, err := inner.Recv()
	if err != nil && !errors.Is(err, io.EOF) {
		_ = inner.Close()
		outcome, kind := classify(err)
		e.record(d, t.Model, outcome, kind, start)
		return nil, err
	}
	return &Stream{
		Model:     t.Model,
		inner:     inner,
		hasPrimed: true,
		primedEv:  ev,
		primedErr: err,
		finish: func(outcome string, kind provider.ErrorKind) {
			e.record(d, t.Model, outcome, kind, start)
		},
	}, nil
}

func (e *Executor) record(d *router.Decision, model, outcome string, kind provider.ErrorKind, start time.Time) {
	d.Attempts = append(d.Attempts, router.Attempt{
		Model:      model,
		Outcome:    outcome,
		ErrorKind:  string(kind),
		DurationMS: float64(e.now().Sub(start).Microseconds()) / 1000,
	})
}

// classify maps a failed call to an attempt outcome and error kind.
func classify(err error) (string, provider.ErrorKind) {
	if errors.Is(err, context.Canceled) {
		return router.OutcomeCanceled, ""
	}
	return router.OutcomeError, provider.KindOf(err)
}

// Stream is a primed provider stream. Recv must be called from one goroutine
// at a time; Close may be called from any goroutine.
type Stream struct {
	Model string // served configured model name

	inner provider.Stream

	// Written before the Stream is returned, then used only by Recv.
	hasPrimed bool
	primedEv  provider.StreamEvent
	primedErr error

	endOnce   sync.Once
	finish    func(outcome string, kind provider.ErrorKind)
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
	if err != nil {
		switch {
		case errors.Is(err, io.EOF):
			s.end(router.OutcomeOK, "")
		case errors.Is(err, context.Canceled):
			s.end(router.OutcomeCanceled, "")
		default:
			s.end(router.OutcomePartial, provider.KindOf(err))
		}
	}
	return ev, err
}

// Close ends the stream. Closing before the end records the attempt as
// canceled. It is idempotent and safe to call concurrently with Recv.
func (s *Stream) Close() error {
	// Record before closing the provider stream so a Recv failure caused by
	// this Close cannot claim the attempt first.
	s.end(router.OutcomeCanceled, "")
	s.closeOnce.Do(func() { s.closeErr = s.inner.Close() })
	return s.closeErr
}

// end records the attempt exactly once.
func (s *Stream) end(outcome string, kind provider.ErrorKind) {
	s.endOnce.Do(func() { s.finish(outcome, kind) })
}
