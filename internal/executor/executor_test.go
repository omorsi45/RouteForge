package executor_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

type step struct {
	ev  provider.StreamEvent
	err error
}

// stubStream replays scripted steps; once they run out it returns io.EOF.
// It is safe for concurrent Recv and Close.
type stubStream struct {
	mu     sync.Mutex
	steps  []step
	recvs  int
	closes int
	closed bool
}

func (s *stubStream) Recv() (provider.StreamEvent, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.recvs++
	if s.closed {
		return provider.StreamEvent{}, fmt.Errorf("stub: %w", context.Canceled)
	}
	if len(s.steps) == 0 {
		return provider.StreamEvent{}, io.EOF
	}
	st := s.steps[0]
	s.steps = s.steps[1:]
	return st.ev, st.err
}

func (s *stubStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	s.closed = true
	return nil
}

func (s *stubStream) counts() (recvs, closes int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.recvs, s.closes
}

type stubProvider struct {
	complete func(ctx context.Context, req provider.Request) (*provider.Response, error)
	stream   func(ctx context.Context, req provider.Request) (provider.Stream, error)

	mu    sync.Mutex
	calls int
	seen  []provider.Request
}

func (p *stubProvider) record(req provider.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.seen = append(p.seen, req)
}

func (p *stubProvider) Complete(ctx context.Context, req provider.Request) (*provider.Response, error) {
	p.record(req)
	return p.complete(ctx, req)
}

func (p *stubProvider) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	p.record(req)
	return p.stream(ctx, req)
}

func newExec(p provider.Provider) *executor.Executor {
	return executor.New(map[string]executor.Target{
		"m": {Model: "m", Upstream: "up-1", Provider: p},
	}, nil)
}

func decision(chosen string) *router.Decision {
	return &router.Decision{Chosen: chosen, Fallbacks: []string{}, Attempts: []router.Attempt{}}
}

func okComplete(context.Context, provider.Request) (*provider.Response, error) {
	return &provider.Response{Content: "hi"}, nil
}

func streamOf(s *stubStream) func(context.Context, provider.Request) (provider.Stream, error) {
	return func(context.Context, provider.Request) (provider.Stream, error) { return s, nil }
}

func oneAttempt(t *testing.T, d *router.Decision) router.Attempt {
	t.Helper()
	if len(d.Attempts) != 1 {
		t.Fatalf("attempts = %+v, want exactly one", d.Attempts)
	}
	if d.Attempts[0].DurationMS < 0 {
		t.Fatalf("negative duration: %+v", d.Attempts[0])
	}
	return d.Attempts[0]
}

func TestCompleteSetsUpstreamAndRecordsAttempt(t *testing.T) {
	p := &stubProvider{complete: okComplete}
	d := decision("m")
	res, err := newExec(p).Complete(context.Background(), d, provider.Request{Model: "client-name"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if p.seen[0].Model != "up-1" {
		t.Fatalf("provider saw model %q, want up-1", p.seen[0].Model)
	}
	if res.Model != "m" || res.Response == nil || res.Response.Content != "hi" {
		t.Fatalf("result = %+v", res)
	}
	a := oneAttempt(t, d)
	if a.Model != "m" || a.Outcome != router.OutcomeOK || a.ErrorKind != "" {
		t.Fatalf("attempt = %+v", a)
	}
}

// steppingClock returns a clock that advances by step on every call.
func steppingClock(step time.Duration) func() time.Time {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	return func() time.Time {
		now = now.Add(step)
		return now
	}
}

// The recorded duration is fractional milliseconds with microsecond
// resolution, so a sub-millisecond upstream time survives.
func TestAttemptDurationHasMicrosecondResolution(t *testing.T) {
	cases := []struct {
		name string
		step time.Duration
		want float64
	}{
		{"one and a half milliseconds", 1500 * time.Microsecond, 1.5},
		{"one microsecond", time.Microsecond, 0.001},
		{"below a microsecond", 900 * time.Nanosecond, 0},
		{"whole milliseconds", 7 * time.Millisecond, 7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := &stubProvider{complete: okComplete}
			e := executor.New(map[string]executor.Target{
				"m": {Model: "m", Upstream: "up-1", Provider: p},
			}, steppingClock(tc.step))
			d := decision("m")
			if _, err := e.Complete(context.Background(), d, provider.Request{}, nil); err != nil {
				t.Fatal(err)
			}
			if got := oneAttempt(t, d).DurationMS; got != tc.want {
				t.Fatalf("DurationMS = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCompleteErrorAttempt(t *testing.T) {
	want := &provider.Error{Kind: provider.KindRateLimited, Provider: "p", Message: "slow down"}
	p := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		return nil, want
	}}
	d := decision("m")
	_, err := newExec(p).Complete(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeError || a.ErrorKind != "rate_limited" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestCompleteCanceledAttempt(t *testing.T) {
	p := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		return nil, fmt.Errorf("x: %w", context.Canceled)
	}}
	d := decision("m")
	_, err := newExec(p).Complete(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeCanceled || a.ErrorKind != "" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestUnknownTarget(t *testing.T) {
	p := &stubProvider{complete: okComplete, stream: streamOf(&stubStream{})}
	e := newExec(p)

	_, err := e.Complete(context.Background(), decision("ghost"), provider.Request{}, nil)
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Kind != provider.KindInternal {
		t.Fatalf("Complete err = %v, want internal *provider.Error", err)
	}
	_, err = e.Stream(context.Background(), decision("ghost"), provider.Request{}, nil)
	if !errors.As(err, &pe) || pe.Kind != provider.KindInternal {
		t.Fatalf("Stream err = %v, want internal *provider.Error", err)
	}
	if p.calls != 0 {
		t.Fatalf("provider called %d times", p.calls)
	}
}

func TestStreamPrimesFirstEvent(t *testing.T) {
	ss := &stubStream{steps: []step{{ev: provider.StreamEvent{Delta: "a"}}, {ev: provider.StreamEvent{Delta: "b"}}}}
	p := &stubProvider{stream: streamOf(ss)}
	d := decision("m")
	s, err := newExec(p).Stream(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if recvs, _ := ss.counts(); recvs != 1 {
		t.Fatalf("inner Recv called %d times before return, want 1", recvs)
	}
	if p.seen[0].Model != "up-1" || s.Model != "m" {
		t.Fatalf("upstream = %q, stream model = %q", p.seen[0].Model, s.Model)
	}
	if len(d.Attempts) != 0 {
		t.Fatalf("attempt recorded before the stream ended: %+v", d.Attempts)
	}
	for _, want := range []string{"a", "b"} {
		ev, err := s.Recv()
		if err != nil || ev.Delta != want {
			t.Fatalf("Recv = %+v, %v; want %q", ev, err, want)
		}
	}
	if _, err := s.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("final Recv err = %v, want io.EOF", err)
	}
	a := oneAttempt(t, d)
	if a.Model != "m" || a.Outcome != router.OutcomeOK || a.ErrorKind != "" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestStreamOpenFailure(t *testing.T) {
	want := &provider.Error{Kind: provider.KindUnavailable, Provider: "p", Message: "down"}
	p := &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
		return nil, want
	}}
	d := decision("m")
	_, err := newExec(p).Stream(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeError || a.ErrorKind != "unavailable" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestStreamFirstRecvFailure(t *testing.T) {
	want := &provider.Error{Kind: provider.KindUnavailable, Provider: "p", Message: "down"}
	ss := &stubStream{steps: []step{{err: want}}}
	p := &stubProvider{stream: streamOf(ss)}
	d := decision("m")
	s, err := newExec(p).Stream(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) || s != nil {
		t.Fatalf("stream = %v, err = %v; want nil stream and %v", s, err, want)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeError || a.ErrorKind != "unavailable" {
		t.Fatalf("attempt = %+v", a)
	}
	if _, closes := ss.counts(); closes != 1 {
		t.Fatalf("inner Close called %d times, want 1", closes)
	}
}

func TestStreamMidFailureIsPartial(t *testing.T) {
	boom := &provider.Error{Kind: provider.KindUnavailable, Provider: "p", Message: "cut"}
	ss := &stubStream{steps: []step{{ev: provider.StreamEvent{Delta: "a"}}, {err: boom}}}
	d := decision("m")
	s, err := newExec(&stubProvider{stream: streamOf(ss)}).Stream(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ev, err := s.Recv(); err != nil || ev.Delta != "a" {
		t.Fatalf("Recv = %+v, %v", ev, err)
	}
	if _, err := s.Recv(); !errors.Is(err, boom) {
		t.Fatalf("Recv err = %v, want %v", err, boom)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomePartial || a.ErrorKind != "unavailable" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestStreamCloseEarlyIsCanceled(t *testing.T) {
	ss := &stubStream{steps: []step{{ev: provider.StreamEvent{Delta: "a"}}, {ev: provider.StreamEvent{Delta: "b"}}}}
	d := decision("m")
	s, err := newExec(&stubProvider{stream: streamOf(ss)}).Stream(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeCanceled || a.ErrorKind != "" {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestStreamEmpty(t *testing.T) {
	ss := &stubStream{}
	d := decision("m")
	s, err := newExec(&stubProvider{stream: streamOf(ss)}).Stream(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv err = %v, want io.EOF", err)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeOK {
		t.Fatalf("attempt = %+v", a)
	}
}

func TestStreamCanceledMidStream(t *testing.T) {
	ss := &stubStream{steps: []step{
		{ev: provider.StreamEvent{Delta: "a"}},
		{err: fmt.Errorf("x: %w", context.Canceled)},
	}}
	d := decision("m")
	s, err := newExec(&stubProvider{stream: streamOf(ss)}).Stream(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Recv(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv err = %v", err)
	}
	a := oneAttempt(t, d)
	if a.Outcome != router.OutcomeCanceled || a.ErrorKind != "" {
		t.Fatalf("attempt = %+v", a)
	}
}

// Close racing a draining Recv must still record exactly one attempt.
func TestStreamCloseRacesRecv(t *testing.T) {
	for i := 0; i < 200; i++ {
		steps := make([]step, 50)
		for j := range steps {
			steps[j] = step{ev: provider.StreamEvent{Delta: "x"}}
		}
		ss := &stubStream{steps: steps}
		d := decision("m")
		s, err := newExec(&stubProvider{stream: streamOf(ss)}).Stream(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			for {
				if _, err := s.Recv(); err != nil {
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			_ = s.Close()
		}()
		wg.Wait()

		a := oneAttempt(t, d)
		if a.Outcome != router.OutcomeOK && a.Outcome != router.OutcomeCanceled {
			t.Fatalf("iteration %d: attempt = %+v", i, a)
		}
	}
}
