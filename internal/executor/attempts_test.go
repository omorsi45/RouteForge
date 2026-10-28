package executor_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/health"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
)

// provErr is a classified provider failure.
func provErr(kind provider.ErrorKind, msg string) *provider.Error {
	return &provider.Error{Kind: kind, Provider: "p", Message: msg}
}

// result is one scripted outcome of a Complete call.
type result struct {
	resp *provider.Response
	err  error
}

func ok(content string) result { return result{resp: &provider.Response{Content: content}} }

func okUsage(content string, completion int) result {
	return result{resp: &provider.Response{Content: content, Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: completion}}}
}

func fail(err error) result { return result{err: err} }

// seq plays results in order; once they run out it repeats the last one.
func seq(rs ...result) func(context.Context, provider.Request) (*provider.Response, error) {
	var (
		mu sync.Mutex
		i  int
	)
	return func(context.Context, provider.Request) (*provider.Response, error) {
		mu.Lock()
		r := rs[min(i, len(rs)-1)]
		i++
		mu.Unlock()
		return r.resp, r.err
	}
}

// streamResult is one scripted outcome of a Stream call.
type streamResult struct {
	s   *stubStream
	err error
}

// streams plays stream results in order; once they run out it repeats the last.
func streams(rs ...streamResult) func(context.Context, provider.Request) (provider.Stream, error) {
	var (
		mu sync.Mutex
		i  int
	)
	return func(context.Context, provider.Request) (provider.Stream, error) {
		mu.Lock()
		r := rs[min(i, len(rs)-1)]
		i++
		mu.Unlock()
		if r.err != nil {
			return nil, r.err
		}
		return r.s, nil
	}
}

func delta(s string) step { return step{ev: provider.StreamEvent{Delta: s}} }

func calls(p *stubProvider) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// sleeper records the delays the executor asks for and never waits.
type sleeper struct {
	mu  sync.Mutex
	got []time.Duration
	err error // returned instead of ctx.Err when set
}

func (s *sleeper) sleep(ctx context.Context, d time.Duration) error {
	s.mu.Lock()
	s.got = append(s.got, d)
	err := s.err
	s.mu.Unlock()
	if err != nil {
		return err
	}
	return ctx.Err()
}

func (s *sleeper) delays() []time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.got)
}

// half draws the middle of every jitter range: the delay before retry n is half
// of min(2s, 100ms * 2^(n-1)), so 50ms, then 100ms, then 200ms.
var half = reliability.Backoff{
	Base: 100 * time.Millisecond,
	Cap:  2 * time.Second,
	Rand: func() float64 { return 0.5 },
}

// rig is an executor over stub providers, one per model, whose sleeps are
// recorded. Tests script the providers before calling exec.
type rig struct {
	p  map[string]*stubProvider
	sl *sleeper
}

func newRig(models ...string) *rig {
	r := &rig{p: map[string]*stubProvider{}, sl: &sleeper{}}
	for _, m := range models {
		r.p[m] = &stubProvider{}
	}
	return r
}

// exec builds the executor: three attempts, half jitter, recorded sleeps. A
// tweak may replace any option.
func (r *rig) exec(tweaks ...func(*executor.Options)) *executor.Executor {
	targets := make(map[string]executor.Target, len(r.p))
	for name, p := range r.p {
		targets[name] = executor.Target{Model: name, Upstream: "up-" + name, Provider: p}
	}
	o := executor.Options{
		Targets:     targets,
		Backoff:     half,
		MaxAttempts: 3,
		Sleep:       r.sl.sleep,
	}
	for _, tw := range tweaks {
		tw(&o)
	}
	return executor.NewWithOptions(o)
}

// route is a decision for chosen with the given fallbacks.
func route(chosen string, fallbacks ...string) *router.Decision {
	return &router.Decision{Chosen: chosen, Fallbacks: append([]string{}, fallbacks...), Attempts: []router.Attempt{}}
}

// attemptsOf renders the recorded attempts as "model:outcome[:error_kind]".
func attemptsOf(d *router.Decision) []string {
	out := make([]string, len(d.Attempts))
	for i, a := range d.Attempts {
		out[i] = a.Model + ":" + a.Outcome
		if a.ErrorKind != "" {
			out[i] += ":" + a.ErrorKind
		}
	}
	return out
}

func wantAttempts(t *testing.T, d *router.Decision, want ...string) {
	t.Helper()
	if got := attemptsOf(d); !slices.Equal(got, want) {
		t.Fatalf("attempts = %v, want %v", got, want)
	}
}

func wantDelays(t *testing.T, s *sleeper, want ...time.Duration) {
	t.Helper()
	if got := s.delays(); !slices.Equal(got, want) {
		t.Fatalf("sleeps = %v, want %v", got, want)
	}
}

// drain reads s to the end and returns the deltas and the final error.
func drain(s *executor.Stream) (deltas []string, err error) {
	for {
		ev, err := s.Recv()
		if err != nil {
			return deltas, err
		}
		deltas = append(deltas, ev.Delta)
	}
}

func TestRetryThenSucceedOnFallback(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	r.p["b"].complete = seq(ok("from b"))
	d := route("a", "b")

	res, err := r.exec().Complete(context.Background(), d, provider.Request{Model: "client-name"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != "b" || res.Response.Content != "from b" {
		t.Fatalf("result = %+v", res)
	}
	wantAttempts(t, d, "a:error:unavailable", "b:ok")
	wantDelays(t, r.sl, 50*time.Millisecond)
	if got := r.p["a"].seen[0].Model; got != "up-a" {
		t.Errorf("a saw model %q, want up-a", got)
	}
	if got := r.p["b"].seen[0].Model; got != "up-b" {
		t.Errorf("b saw model %q, want up-b", got)
	}
}

func TestNoRetryOnNonRetryableKinds(t *testing.T) {
	for _, kind := range []provider.ErrorKind{
		provider.KindBadRequest, provider.KindAuth, provider.KindContextLength, provider.KindInternal,
	} {
		t.Run(string(kind), func(t *testing.T) {
			r := newRig("a", "b")
			want := provErr(kind, "no")
			r.p["a"].complete = seq(fail(want))
			r.p["b"].complete = seq(ok("never"))
			d := route("a", "b")

			_, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
			if !errors.Is(err, want) {
				t.Fatalf("err = %v, want %v", err, want)
			}
			wantAttempts(t, d, "a:error:"+string(kind))
			wantDelays(t, r.sl)
			if n := calls(r.p["b"]); n != 0 {
				t.Errorf("fallback called %d times, want 0", n)
			}
		})
	}
}

func TestNoRetryOnBadRequest(t *testing.T) {
	r := newRig("a")
	want := provErr(provider.KindBadRequest, "temperature out of range")
	r.p["a"].complete = seq(fail(want))
	d := route("a")

	_, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	wantAttempts(t, d, "a:error:bad_request")
	if n := calls(r.p["a"]); n != 1 {
		t.Fatalf("provider called %d times, want 1", n)
	}
}

func TestPinnedRetriesSameModel(t *testing.T) {
	r := newRig("a")
	down := fail(provErr(provider.KindUnavailable, "down"))
	r.p["a"].complete = seq(down, down, ok("third time"))
	d := route("a")

	res, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != "a" || res.Response.Content != "third time" {
		t.Fatalf("result = %+v", res)
	}
	wantAttempts(t, d, "a:error:unavailable", "a:error:unavailable", "a:ok")
	wantDelays(t, r.sl, 50*time.Millisecond, 100*time.Millisecond)
}

func TestMaxAttemptsThree(t *testing.T) {
	r := newRig("a", "b", "c", "d")
	for name, p := range r.p {
		p.complete = seq(fail(provErr(provider.KindUnavailable, "down from "+name)))
	}
	d := route("a", "b", "c", "d")

	_, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Message != "down from c" {
		t.Fatalf("err = %v, want the error of the third attempt (model c)", err)
	}
	wantAttempts(t, d, "a:error:unavailable", "b:error:unavailable", "c:error:unavailable")
	if n := calls(r.p["d"]); n != 0 {
		t.Errorf("fourth model called %d times, want 0", n)
	}
	wantDelays(t, r.sl, 50*time.Millisecond, 100*time.Millisecond)
}

func TestMaxAttemptsZeroMeansOne(t *testing.T) {
	r := newRig("a")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	d := route("a")

	_, err := r.exec(func(o *executor.Options) { o.MaxAttempts = 0 }).Complete(context.Background(), d, provider.Request{}, nil)
	if err == nil {
		t.Fatal("want an error")
	}
	wantAttempts(t, d, "a:error:unavailable")
	wantDelays(t, r.sl)
}

// Attempt i uses targets[min(i, len-1)]: failover walks the list, then the
// last model is tried again.
func TestAttemptsWalkTargetsThenRepeatTheLast(t *testing.T) {
	down := fail(provErr(provider.KindUnavailable, "down"))

	t.Run("three targets, three attempts", func(t *testing.T) {
		r := newRig("a", "b", "c")
		r.p["a"].complete = seq(down)
		r.p["b"].complete = seq(down)
		r.p["c"].complete = seq(ok("c"))
		d := route("a", "b", "c")
		if _, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		wantAttempts(t, d, "a:error:unavailable", "b:error:unavailable", "c:ok")
	})

	t.Run("two targets, the last one is repeated", func(t *testing.T) {
		r := newRig("a", "b")
		r.p["a"].complete = seq(down)
		r.p["b"].complete = seq(down, ok("b again"))
		d := route("a", "b")
		res, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if res.Response.Content != "b again" {
			t.Fatalf("result = %+v", res)
		}
		wantAttempts(t, d, "a:error:unavailable", "b:error:unavailable", "b:ok")
	})
}

func TestRetryAfterHonored(t *testing.T) {
	limited := fail(&provider.Error{Kind: provider.KindRateLimited, Provider: "p", Message: "slow down", RetryAfter: 300 * time.Millisecond})

	t.Run("raises a shorter backoff", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(limited, ok("hi"))
		zero := func(o *executor.Options) { o.Backoff.Rand = func() float64 { return 0 } }
		if _, err := r.exec(zero).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		wantDelays(t, r.sl, 300*time.Millisecond)
	})

	t.Run("keeps a longer backoff", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(limited, ok("hi"))
		long := func(o *executor.Options) {
			o.Backoff = reliability.Backoff{Base: time.Second, Cap: 2 * time.Second, Rand: func() float64 { return 0.5 }}
		}
		if _, err := r.exec(long).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		wantDelays(t, r.sl, 500*time.Millisecond)
	})
}

func TestRetryAfterBeyondDeadlineStops(t *testing.T) {
	r := newRig("a", "b")
	want := &provider.Error{Kind: provider.KindRateLimited, Provider: "p", Message: "slow down", RetryAfter: 5 * time.Second}
	r.p["a"].complete = seq(fail(want))
	r.p["b"].complete = seq(ok("never"))
	d := route("a", "b")
	budget := reliability.NewRetryBudget(0.2, 10*time.Second, 1, nil)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, err := r.exec(func(o *executor.Options) { o.RetryBudget = budget }).Complete(ctx, d, provider.Request{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want the 429", err)
	}
	wantAttempts(t, d, "a:error:rate_limited")
	wantDelays(t, r.sl)
	if n := calls(r.p["b"]); n != 0 {
		t.Errorf("fallback called %d times, want 0", n)
	}
	// A retry that never happened must not cost budget.
	if !budget.TryRetry() {
		t.Error("the retry budget was spent on a retry that was not made")
	}
}

func TestRetryBudgetExhaustedStops(t *testing.T) {
	down := fail(provErr(provider.KindUnavailable, "down"))
	cases := []struct {
		name     string
		floor    int
		attempts []string
	}{
		{"no retries allowed", 0, []string{"a:error:unavailable"}},
		{"one retry allowed", 1, []string{"a:error:unavailable", "b:error:unavailable"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig("a", "b", "c")
			for _, p := range r.p {
				p.complete = seq(down)
			}
			d := route("a", "b", "c")
			// 20% of one request is 0 retries, so only the floor allows any.
			budget := reliability.NewRetryBudget(0.2, 10*time.Second, tc.floor, nil)

			_, err := r.exec(func(o *executor.Options) { o.RetryBudget = budget }).Complete(context.Background(), d, provider.Request{}, nil)
			if err == nil {
				t.Fatal("want the last upstream error")
			}
			wantAttempts(t, d, tc.attempts...)
		})
	}
}

// A request is counted once however many attempts it takes: with a ratio of 1
// and no floor one request earns exactly one retry.
func TestRetryBudgetCountsRequestsNotAttempts(t *testing.T) {
	r := newRig("a")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	d := route("a")
	budget := reliability.NewRetryBudget(1, 10*time.Second, 0, nil)

	if _, err := r.exec(func(o *executor.Options) { o.RetryBudget = budget }).Complete(context.Background(), d, provider.Request{}, nil); err == nil {
		t.Fatal("want an error")
	}
	wantAttempts(t, d, "a:error:unavailable", "a:error:unavailable")
}

func TestNoRetryAfterTheClientLeaves(t *testing.T) {
	r := newRig("a", "b")
	ctx, cancel := context.WithCancel(context.Background())
	r.p["a"].complete = func(context.Context, provider.Request) (*provider.Response, error) {
		cancel() // the client goes away while the model is failing
		return nil, provErr(provider.KindUnavailable, "down")
	}
	r.p["b"].complete = seq(ok("never"))
	d := route("a", "b")

	_, err := r.exec().Complete(ctx, d, provider.Request{}, nil)
	if provider.KindOf(err) != provider.KindUnavailable {
		t.Fatalf("err = %v, want the upstream error", err)
	}
	wantAttempts(t, d, "a:error:unavailable")
	wantDelays(t, r.sl)
	if n := calls(r.p["b"]); n != 0 {
		t.Errorf("fallback called %d times, want 0", n)
	}
}

func TestSleepFailureStopsRetrying(t *testing.T) {
	r := newRig("a", "b")
	want := provErr(provider.KindUnavailable, "down")
	r.p["a"].complete = seq(fail(want))
	r.p["b"].complete = seq(ok("never"))
	r.sl.err = context.Canceled // the context ended during the backoff
	d := route("a", "b")

	_, err := r.exec().Complete(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) {
		t.Fatalf("err = %v, want the previous upstream error", err)
	}
	wantAttempts(t, d, "a:error:unavailable")
	if n := calls(r.p["b"]); n != 0 {
		t.Errorf("fallback called %d times, want 0", n)
	}
}

// Without an injected Sleep the executor waits on the clock and gives up when
// the context ends.
func TestDefaultSleepWaitsAndHonorsContext(t *testing.T) {
	t.Run("waits the backoff", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a")
			r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")), ok("hi"))
			ex := r.exec(func(o *executor.Options) { o.Sleep = nil })

			start := time.Now()
			if _, err := ex.Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
				t.Fatal(err)
			}
			if got := time.Since(start); got != 50*time.Millisecond {
				t.Fatalf("waited %v, want 50ms", got)
			}
		})
	})

	t.Run("does not start a retry that cannot finish", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a")
			want := provErr(provider.KindUnavailable, "down")
			r.p["a"].complete = seq(fail(want))
			ex := r.exec(func(o *executor.Options) { o.Sleep = nil })

			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			d := route("a")
			// The 50ms backoff does not fit in the 20ms left.
			if _, err := ex.Complete(ctx, d, provider.Request{}, nil); !errors.Is(err, want) {
				t.Fatalf("err = %v, want the upstream error", err)
			}
			wantAttempts(t, d, "a:error:unavailable")
		})
	})

	t.Run("stops when the context ends during the wait", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a")
			want := provErr(provider.KindUnavailable, "down")
			r.p["a"].complete = seq(fail(want), ok("never"))
			ex := r.exec(func(o *executor.Options) { o.Sleep = nil })

			ctx, cancel := context.WithCancel(context.Background())
			go func() {
				time.Sleep(10 * time.Millisecond) // inside the 50ms backoff
				cancel()
			}()
			d := route("a")
			if _, err := ex.Complete(ctx, d, provider.Request{}, nil); !errors.Is(err, want) {
				t.Fatalf("err = %v, want the upstream error", err)
			}
			wantAttempts(t, d, "a:error:unavailable")
			if n := calls(r.p["a"]); n != 1 {
				t.Fatalf("provider called %d times, want 1", n)
			}
		})
	})
}

func breakers(consecutive int) *reliability.Breakers {
	return reliability.NewBreakers(reliability.BreakerConfig{
		ConsecutiveFailures: consecutive,
		ErrorRate:           1,
		MinRequests:         1000,
		Window:              time.Minute,
		OpenFor:             time.Hour,
		HalfOpenProbes:      1,
	}, nil)
}

// trip opens the breaker of model by failing it enough times.
func trip(t *testing.T, b *reliability.Breakers, model string) {
	t.Helper()
	for i := 0; i < 20 && !b.IsOpen(model); i++ {
		if ticket, ok := b.Allow(model); ok {
			ticket.Done(false)
		}
	}
	if !b.IsOpen(model) {
		t.Fatalf("breaker of %s did not open", model)
	}
}

func TestBreakerOpenSkipsTarget(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(ok("never"))
	r.p["b"].complete = seq(ok("from b"))
	br := breakers(1)
	trip(t, br, "a")
	d := route("a", "b")

	// One attempt is allowed and a skipped target is not an attempt, so the
	// request still reaches b.
	res, err := r.exec(func(o *executor.Options) { o.Breakers = br; o.MaxAttempts = 1 }).Complete(context.Background(), d, provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != "b" {
		t.Fatalf("served by %q, want b", res.Model)
	}
	if n := calls(r.p["a"]); n != 0 {
		t.Errorf("open model called %d times, want 0", n)
	}
	wantAttempts(t, d, "b:ok")
	wantDelays(t, r.sl)
}

// manualClock is a clock the test advances by hand.
type manualClock struct {
	mu sync.Mutex
	t  time.Time
}

func newManualClock() *manualClock {
	return &manualClock{t: time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)}
}

func (c *manualClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *manualClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// halfOpen returns breakers whose model a is half-open with room for one probe.
func halfOpen(t *testing.T) *reliability.Breakers {
	t.Helper()
	clk := newManualClock()
	br := reliability.NewBreakers(reliability.BreakerConfig{
		ConsecutiveFailures: 1, ErrorRate: 1, MinRequests: 1000, Window: time.Minute, OpenFor: time.Minute, HalfOpenProbes: 1,
	}, clk.now)
	ticket, ok := br.Allow("a")
	if !ok {
		t.Fatal("a new breaker refused a request")
	}
	ticket.Done(false)
	clk.advance(time.Minute)
	if got := br.State("a"); got != reliability.StateHalfOpen {
		t.Fatalf("state = %v, want half-open", got)
	}
	return br
}

// wantProbeFree checks that a's half-open breaker is still half-open and has
// its one probe slot free: the attempt gave the ticket back without a verdict.
func wantProbeFree(t *testing.T, br *reliability.Breakers) {
	t.Helper()
	if got := br.State("a"); got != reliability.StateHalfOpen {
		t.Fatalf("state = %v, want half-open: the attempt reached no verdict", got)
	}
	ticket, ok := br.Allow("a")
	if !ok {
		t.Fatal("the probe slot was not given back")
	}
	ticket.Cancel()
}

func TestBreakerLearnsFromAttempts(t *testing.T) {
	down := provErr(provider.KindUnavailable, "down")
	once := func(o *executor.Options) { o.MaxAttempts = 1 }
	call := func(t *testing.T, r *rig, br *reliability.Breakers) {
		t.Helper()
		_, _ = r.exec(once, func(o *executor.Options) { o.Breakers = br }).Complete(context.Background(), route("a"), provider.Request{}, nil)
	}

	t.Run("failures open it", func(t *testing.T) {
		r, br := newRig("a"), breakers(2)
		r.p["a"].complete = seq(fail(down))
		call(t, r, br)
		if br.IsOpen("a") {
			t.Fatal("open after one failure of two")
		}
		call(t, r, br)
		if !br.IsOpen("a") {
			t.Fatal("not open after two consecutive failures")
		}
	})

	t.Run("a success resets the streak", func(t *testing.T) {
		r, br := newRig("a"), breakers(2)
		r.p["a"].complete = seq(fail(down), ok("hi"), fail(down))
		call(t, r, br)
		call(t, r, br)
		call(t, r, br)
		if br.IsOpen("a") {
			t.Fatal("open although a success came between the failures")
		}
	})

	t.Run("a request the model rejected on its merits is not a failure", func(t *testing.T) {
		r, br := newRig("a"), breakers(2)
		r.p["a"].complete = seq(fail(provErr(provider.KindBadRequest, "no")))
		for range 5 {
			call(t, r, br)
		}
		if br.IsOpen("a") {
			t.Fatal("open after bad requests")
		}
	})

	t.Run("a stream cut mid-way is a failure", func(t *testing.T) {
		r, br := newRig("a"), breakers(1)
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), {err: down}}}})
		s, err := r.exec(once, func(o *executor.Options) { o.Breakers = br }).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if br.IsOpen("a") {
			t.Fatal("open before the stream failed")
		}
		_, _ = drain(s)
		if !br.IsOpen("a") {
			t.Fatal("not open after a mid-stream failure")
		}
	})

	t.Run("a probe that succeeds closes it", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].complete = seq(ok("hi"))
		call(t, r, br)
		if got := br.State("a"); got != reliability.StateClosed {
			t.Fatalf("state = %v, want closed", got)
		}
	})

	t.Run("a probe that fails reopens it", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].complete = seq(fail(down))
		call(t, r, br)
		if !br.IsOpen("a") {
			t.Fatal("not open after a failed probe")
		}
	})
}

// An attempt that ends without a verdict on the model, because the caller left
// or the call never started, must give a half-open probe back, not report it.
func TestAttemptsWithoutAVerdictReturnTheProbe(t *testing.T) {
	t.Run("canceled call", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].complete = seq(fail(fmt.Errorf("p: %w", context.Canceled)))
		_, _ = r.exec(func(o *executor.Options) { o.Breakers = br }).Complete(context.Background(), route("a"), provider.Request{}, nil)
		wantProbeFree(t, br)
	})

	t.Run("stream closed early", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), delta("b")}}})
		s, err := r.exec(func(o *executor.Options) { o.Breakers = br }).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		wantProbeFree(t, br)
	})

	t.Run("stream canceled mid-stream", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), {err: fmt.Errorf("p: %w", context.Canceled)}}}})
		s, err := r.exec(func(o *executor.Options) { o.Breakers = br }).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = drain(s)
		wantProbeFree(t, br)
	})

	t.Run("skipped by a full queue", func(t *testing.T) {
		r, br := newRig("a", "b"), halfOpen(t)
		r.p["b"].complete = seq(ok("from b"))
		q := reliability.NewQueues(map[string]reliability.QueueLimit{
			"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second},
			"b": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second},
		})
		held, err := q.Acquire(context.Background(), "a")
		if err != nil {
			t.Fatal(err)
		}
		defer held()
		_, err = r.exec(func(o *executor.Options) { o.Breakers = br; o.Queues = q }).Complete(context.Background(), route("a", "b"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wantProbeFree(t, br)
	})

	t.Run("refused by the meter", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		r.p["a"].complete = seq(ok("never"))
		m := &recMeter{beginErr: func(int) error { return errors.New("budget exceeded") }}
		_, _ = r.exec(func(o *executor.Options) { o.Breakers = br }).Complete(context.Background(), route("a"), provider.Request{}, m)
		wantProbeFree(t, br)
	})

	t.Run("context ended in the queue", func(t *testing.T) {
		r, br := newRig("a"), halfOpen(t)
		q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 1, Timeout: time.Second}})
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // already over: Acquire fails at once
		_, err := r.exec(func(o *executor.Options) { o.Breakers = br; o.Queues = q }).Complete(ctx, route("a"), provider.Request{}, nil)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		wantProbeFree(t, br)
	})
}

func TestAllTargetsSkipped(t *testing.T) {
	t.Run("breakers open", func(t *testing.T) {
		r := newRig("a", "b")
		r.p["a"].complete = seq(ok("never"))
		r.p["b"].complete = seq(ok("never"))
		br := breakers(1)
		trip(t, br, "a")
		trip(t, br, "b")
		d := route("a", "b")

		_, err := r.exec(func(o *executor.Options) { o.Breakers = br }).Complete(context.Background(), d, provider.Request{}, nil)
		if !errors.Is(err, executor.ErrNoTargetAccepted) {
			t.Fatalf("err = %v, want ErrNoTargetAccepted", err)
		}
		if calls(r.p["a"])+calls(r.p["b"]) != 0 {
			t.Error("an upstream call was made")
		}
		wantAttempts(t, d)
	})

	t.Run("queue full", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(ok("never"))
		q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second}})
		held, err := q.Acquire(context.Background(), "a")
		if err != nil {
			t.Fatal(err)
		}
		defer held()
		d := route("a")

		_, err = r.exec(func(o *executor.Options) { o.Queues = q }).Complete(context.Background(), d, provider.Request{}, nil)
		if !errors.Is(err, executor.ErrNoTargetAccepted) {
			t.Fatalf("err = %v, want ErrNoTargetAccepted", err)
		}
		if n := calls(r.p["a"]); n != 0 {
			t.Errorf("upstream called %d times, want 0", n)
		}
	})

	t.Run("queue timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a")
			r.p["a"].complete = seq(ok("never"))
			q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 1, Timeout: 50 * time.Millisecond}})
			held, err := q.Acquire(context.Background(), "a")
			if err != nil {
				t.Fatal(err)
			}
			defer held()

			start := time.Now()
			_, err = r.exec(func(o *executor.Options) { o.Queues = q }).Complete(context.Background(), route("a"), provider.Request{}, nil)
			if !errors.Is(err, executor.ErrNoTargetAccepted) {
				t.Fatalf("err = %v, want ErrNoTargetAccepted", err)
			}
			if got := time.Since(start); got != 50*time.Millisecond {
				t.Errorf("waited %v in the queue, want its 50ms timeout", got)
			}
		})
	})

	t.Run("streams too", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].stream = streams(streamResult{s: &stubStream{}})
		br := breakers(1)
		trip(t, br, "a")
		s, err := r.exec(func(o *executor.Options) { o.Breakers = br }).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if !errors.Is(err, executor.ErrNoTargetAccepted) || s != nil {
			t.Fatalf("stream = %v, err = %v; want nil and ErrNoTargetAccepted", s, err)
		}
	})
}

func TestSkippedTargetsAfterUpstreamCallsReturnTheLastError(t *testing.T) {
	r := newRig("a", "b")
	want := provErr(provider.KindUnavailable, "down")
	r.p["a"].complete = seq(fail(want))
	r.p["b"].complete = seq(ok("never"))
	br := breakers(1)
	trip(t, br, "b")
	d := route("a", "b")

	_, err := r.exec(func(o *executor.Options) { o.Breakers = br }).Complete(context.Background(), d, provider.Request{}, nil)
	if !errors.Is(err, want) || errors.Is(err, executor.ErrNoTargetAccepted) {
		t.Fatalf("err = %v, want the upstream error", err)
	}
	wantAttempts(t, d, "a:error:unavailable")
}

func TestQueueFullTargetFallsThrough(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(ok("never"))
	r.p["b"].complete = seq(ok("from b"))
	q := reliability.NewQueues(map[string]reliability.QueueLimit{
		"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second},
		"b": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second},
	})
	held, err := q.Acquire(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	defer held()

	res, err := r.exec(func(o *executor.Options) { o.Queues = q; o.MaxAttempts = 1 }).Complete(context.Background(), route("a", "b"), provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Model != "b" {
		t.Fatalf("served by %q, want b", res.Model)
	}
}

// The wait for a slot spends the request's deadline: a context that ends in the
// line ends the request, classified like any other expiry.
func TestQueueWaitCountsAgainstTheDeadline(t *testing.T) {
	// line returns queues whose only slot is taken, with room for one waiter.
	line := func(t *testing.T) (*reliability.Queues, func()) {
		t.Helper()
		q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 1, Timeout: 10 * time.Second}})
		held, err := q.Acquire(context.Background(), "a")
		if err != nil {
			t.Fatal(err)
		}
		return q, held
	}

	t.Run("deadline", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a", "b")
			r.p["a"].complete = seq(ok("never"))
			r.p["b"].complete = seq(ok("never"))
			q, held := line(t)
			defer held()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			d := route("a", "b")

			_, err := r.exec(func(o *executor.Options) { o.Queues = q }).Complete(ctx, d, provider.Request{}, nil)
			if !errors.Is(err, context.DeadlineExceeded) || provider.KindOf(err) != provider.KindTimeout {
				t.Fatalf("err = %v (kind %q), want the deadline, kind timeout", err, provider.KindOf(err))
			}
			if calls(r.p["a"])+calls(r.p["b"]) != 0 {
				t.Error("an upstream call was made")
			}
			wantAttempts(t, d)
		})
	})

	t.Run("client gone", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			r := newRig("a")
			r.p["a"].complete = seq(ok("never"))
			q, held := line(t)
			defer held()
			ctx, cancel := context.WithCancel(context.Background())
			go func() {
				time.Sleep(time.Second)
				cancel()
			}()

			_, err := r.exec(func(o *executor.Options) { o.Queues = q }).Complete(ctx, route("a"), provider.Request{}, nil)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want context.Canceled", err)
			}
		})
	})
}

func TestQueueReleasedAfterAttempt(t *testing.T) {
	q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second}})
	r := newRig("a")
	var heldDuringCall bool
	r.p["a"].complete = func(context.Context, provider.Request) (*provider.Response, error) {
		heldDuringCall = q.Full("a")
		return &provider.Response{Content: "hi"}, nil
	}
	ex := r.exec(func(o *executor.Options) { o.Queues = q })

	if _, err := ex.Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	if !heldDuringCall {
		t.Error("the slot was not held while the call ran")
	}
	if q.Full("a") {
		t.Fatal("the slot was not released after a success")
	}

	r.p["a"].complete = seq(fail(provErr(provider.KindBadRequest, "no")))
	if _, err := ex.Complete(context.Background(), route("a"), provider.Request{}, nil); err == nil {
		t.Fatal("want an error")
	}
	if q.Full("a") {
		t.Fatal("the slot was not released after a failure")
	}
}

func TestStreamReleasesQueueAtEnd(t *testing.T) {
	newQueues := func() *reliability.Queues {
		return reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second}})
	}
	boom := provErr(provider.KindUnavailable, "cut")
	cases := []struct {
		name  string
		steps []step
		end   func(t *testing.T, s *executor.Stream)
	}{
		{"EOF", []step{delta("a")}, func(t *testing.T, s *executor.Stream) {
			if _, err := drain(s); !errors.Is(err, io.EOF) {
				t.Fatalf("drain err = %v", err)
			}
		}},
		{"mid-stream failure", []step{delta("a"), {err: boom}}, func(t *testing.T, s *executor.Stream) {
			if _, err := drain(s); !errors.Is(err, boom) {
				t.Fatalf("drain err = %v", err)
			}
		}},
		{"close before the end", []step{delta("a"), delta("b")}, func(t *testing.T, s *executor.Stream) {
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			q := newQueues()
			r := newRig("a")
			r.p["a"].stream = streams(streamResult{s: &stubStream{steps: tc.steps}})
			s, err := r.exec(func(o *executor.Options) { o.Queues = q }).Stream(context.Background(), route("a"), provider.Request{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !q.Full("a") {
				t.Fatal("the slot was released before the stream ended")
			}
			tc.end(t, s)
			if q.Full("a") {
				t.Fatal("the slot was not released when the stream ended")
			}
		})
	}

	t.Run("failed open", func(t *testing.T) {
		q := newQueues()
		r := newRig("a")
		r.p["a"].stream = streams(streamResult{err: provErr(provider.KindBadRequest, "no")})
		if _, err := r.exec(func(o *executor.Options) { o.Queues = q }).Stream(context.Background(), route("a"), provider.Request{}, nil); err == nil {
			t.Fatal("want an error")
		}
		if q.Full("a") {
			t.Fatal("the slot was not released after a failed open")
		}
	})
}

func TestStreamRetriesBeforeFirstEventOnly(t *testing.T) {
	down := provErr(provider.KindUnavailable, "down")

	t.Run("failed open is retried", func(t *testing.T) {
		r := newRig("a", "b")
		r.p["a"].stream = streams(streamResult{err: down})
		r.p["b"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("x"), delta("y")}}})
		d := route("a", "b")

		s, err := r.exec().Stream(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if s.Model != "b" {
			t.Fatalf("stream model = %q, want b", s.Model)
		}
		if got, err := drain(s); !slices.Equal(got, []string{"x", "y"}) || !errors.Is(err, io.EOF) {
			t.Fatalf("drain = %v, %v", got, err)
		}
		wantAttempts(t, d, "a:error:unavailable", "b:ok")
		wantDelays(t, r.sl, 50*time.Millisecond)
	})

	t.Run("failure on the first event is retried", func(t *testing.T) {
		r := newRig("a", "b")
		first := &stubStream{steps: []step{{err: down}}}
		r.p["a"].stream = streams(streamResult{s: first})
		r.p["b"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("x")}}})
		d := route("a", "b")

		s, err := r.exec().Stream(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := drain(s); !errors.Is(err, io.EOF) {
			t.Fatalf("drain err = %v", err)
		}
		if _, closes := first.counts(); closes != 1 {
			t.Errorf("the failed stream was closed %d times, want 1", closes)
		}
		wantAttempts(t, d, "a:error:unavailable", "b:ok")
	})

	t.Run("failure after the first event is not retried", func(t *testing.T) {
		r := newRig("a", "b")
		cut := provErr(provider.KindUnavailable, "cut")
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), {err: cut}}}})
		r.p["b"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("never")}}})
		d := route("a", "b")

		s, err := r.exec().Stream(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := drain(s)
		if !slices.Equal(got, []string{"a"}) || !errors.Is(err, cut) {
			t.Fatalf("drain = %v, %v", got, err)
		}
		wantAttempts(t, d, "a:partial:unavailable")
		wantDelays(t, r.sl)
		if n := calls(r.p["b"]); n != 0 {
			t.Errorf("fallback called %d times after the first byte, want 0", n)
		}
	})

	t.Run("a stream that ends at once is a valid empty stream", func(t *testing.T) {
		r := newRig("a", "b")
		r.p["a"].stream = streams(streamResult{s: &stubStream{}})
		d := route("a", "b")
		s, err := r.exec().Stream(context.Background(), d, provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv err = %v, want io.EOF", err)
		}
		wantAttempts(t, d, "a:ok")
	})
}

// recMeter records every Begin and End.
type recMeter struct {
	mu       sync.Mutex
	begun    []string
	ends     []executor.Settlement
	beginErr func(call int) error // nil: Begin always succeeds
}

func (m *recMeter) Begin(_ context.Context, model string) (executor.Charge, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	call := len(m.begun)
	m.begun = append(m.begun, model)
	if m.beginErr != nil {
		if err := m.beginErr(call); err != nil {
			return nil, err
		}
	}
	return &recCharge{m: m}, nil
}

func (m *recMeter) settlements() []executor.Settlement {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.ends)
}

type recCharge struct{ m *recMeter }

func (c *recCharge) End(s executor.Settlement) {
	c.m.mu.Lock()
	defer c.m.mu.Unlock()
	c.m.ends = append(c.m.ends, s)
}

// wantOneSettlement checks that End ran exactly once and returns what it got.
func wantOneSettlement(t *testing.T, m *recMeter) executor.Settlement {
	t.Helper()
	ends := m.settlements()
	if len(ends) != 1 {
		t.Fatalf("End called %d times, want 1: %+v", len(ends), ends)
	}
	return ends[0]
}

func TestMeterEndCalledOnEveryOutcome(t *testing.T) {
	usage := &provider.Usage{PromptTokens: 7, CompletionTokens: 3}

	t.Run("ok", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(result{resp: &provider.Response{Content: "hello", Usage: usage}})
		m := &recMeter{}
		if _, err := r.exec().Complete(context.Background(), route("a"), provider.Request{}, m); err != nil {
			t.Fatal(err)
		}
		s := wantOneSettlement(t, m)
		if s.Outcome != router.OutcomeOK || s.Usage == nil || *s.Usage != *usage || s.DeliveredBytes != len("hello") {
			t.Fatalf("settlement = %+v", s)
		}
		if !slices.Equal(m.begun, []string{"a"}) {
			t.Errorf("Begin models = %v, want [a]", m.begun)
		}
	})

	t.Run("ok without usage", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(ok("hello"))
		m := &recMeter{}
		if _, err := r.exec().Complete(context.Background(), route("a"), provider.Request{}, m); err != nil {
			t.Fatal(err)
		}
		if s := wantOneSettlement(t, m); s.Usage != nil || s.DeliveredBytes != 5 {
			t.Fatalf("settlement = %+v, want nil usage and 5 bytes", s)
		}
	})

	t.Run("error", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(fail(provErr(provider.KindBadRequest, "no")))
		m := &recMeter{}
		if _, err := r.exec().Complete(context.Background(), route("a"), provider.Request{}, m); err == nil {
			t.Fatal("want an error")
		}
		if s := wantOneSettlement(t, m); s.Outcome != router.OutcomeError || s.Usage != nil || s.DeliveredBytes != 0 {
			t.Fatalf("settlement = %+v", s)
		}
	})

	t.Run("canceled", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].complete = seq(fail(fmt.Errorf("p: %w", context.Canceled)))
		m := &recMeter{}
		if _, err := r.exec().Complete(context.Background(), route("a"), provider.Request{}, m); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v", err)
		}
		if s := wantOneSettlement(t, m); s.Outcome != router.OutcomeCanceled || s.DeliveredBytes != 0 {
			t.Fatalf("settlement = %+v", s)
		}
	})

	t.Run("stream ok settles the last usage and every byte", func(t *testing.T) {
		r := newRig("a")
		early, last := &provider.Usage{PromptTokens: 1, CompletionTokens: 1}, usage
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{
			delta("ab"),
			{ev: provider.StreamEvent{Delta: "c", Usage: early}},
			{ev: provider.StreamEvent{FinishReason: "stop", Usage: last}},
		}}})
		m := &recMeter{}
		s, err := r.exec().Stream(context.Background(), route("a"), provider.Request{}, m)
		if err != nil {
			t.Fatal(err)
		}
		if got := m.settlements(); len(got) != 0 {
			t.Fatalf("End called before the stream ended: %+v", got)
		}
		if _, err := drain(s); !errors.Is(err, io.EOF) {
			t.Fatalf("drain err = %v", err)
		}
		got := wantOneSettlement(t, m)
		if got.Outcome != router.OutcomeOK || got.Usage == nil || *got.Usage != *last || got.DeliveredBytes != 3 {
			t.Fatalf("settlement = %+v", got)
		}
	})

	t.Run("partial stream", func(t *testing.T) {
		r := newRig("a")
		cut := provErr(provider.KindUnavailable, "cut")
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("ab"), delta("cde"), {err: cut}}}})
		m := &recMeter{}
		s, err := r.exec().Stream(context.Background(), route("a"), provider.Request{}, m)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := drain(s); !errors.Is(err, cut) {
			t.Fatalf("drain err = %v", err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if got := wantOneSettlement(t, m); got.Outcome != router.OutcomePartial || got.Usage != nil || got.DeliveredBytes != 5 {
			t.Fatalf("settlement = %+v", got)
		}
	})

	t.Run("stream closed early", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("ab"), delta("cd")}}})
		m := &recMeter{}
		s, err := r.exec().Stream(context.Background(), route("a"), provider.Request{}, m)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Recv(); err != nil { // "ab" is delivered, "cd" is not
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if got := wantOneSettlement(t, m); got.Outcome != router.OutcomeCanceled || got.DeliveredBytes != 2 {
			t.Fatalf("settlement = %+v", got)
		}
	})

	t.Run("stream canceled mid-stream", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("ab"), {err: fmt.Errorf("p: %w", context.Canceled)}}}})
		m := &recMeter{}
		s, err := r.exec().Stream(context.Background(), route("a"), provider.Request{}, m)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := drain(s); !errors.Is(err, context.Canceled) {
			t.Fatalf("drain err = %v", err)
		}
		if got := wantOneSettlement(t, m); got.Outcome != router.OutcomeCanceled || got.DeliveredBytes != 2 {
			t.Fatalf("settlement = %+v", got)
		}
	})

	t.Run("stream open failure", func(t *testing.T) {
		r := newRig("a")
		r.p["a"].stream = streams(streamResult{err: provErr(provider.KindBadRequest, "no")})
		m := &recMeter{}
		if _, err := r.exec().Stream(context.Background(), route("a"), provider.Request{}, m); err == nil {
			t.Fatal("want an error")
		}
		if got := wantOneSettlement(t, m); got.Outcome != router.OutcomeError || got.DeliveredBytes != 0 {
			t.Fatalf("settlement = %+v", got)
		}
	})
}

// A provider that panics must not leak: the charge ends once as an error, the
// queue slot and the breaker ticket come back, and the panic goes on.
func TestMeterEndCalledWhenProviderPanics(t *testing.T) {
	mustPanic := func(t *testing.T, f func()) (v any) {
		t.Helper()
		defer func() { v = recover() }()
		f()
		return nil
	}
	for _, tc := range []struct {
		name string
		arm  func(p *stubProvider)
		call func(ex *executor.Executor, m executor.Meter) error
	}{
		{
			"complete",
			func(p *stubProvider) {
				p.complete = func(context.Context, provider.Request) (*provider.Response, error) { panic("provider exploded") }
			},
			func(ex *executor.Executor, m executor.Meter) error {
				_, err := ex.Complete(context.Background(), route("a"), provider.Request{}, m)
				return err
			},
		},
		{
			"stream open",
			func(p *stubProvider) {
				p.stream = func(context.Context, provider.Request) (provider.Stream, error) { panic("provider exploded") }
			},
			func(ex *executor.Executor, m executor.Meter) error {
				_, err := ex.Stream(context.Background(), route("a"), provider.Request{}, m)
				return err
			},
		},
		{
			"stream first event",
			func(p *stubProvider) {
				p.stream = func(context.Context, provider.Request) (provider.Stream, error) { return panicStream{}, nil }
			},
			func(ex *executor.Executor, m executor.Meter) error {
				_, err := ex.Stream(context.Background(), route("a"), provider.Request{}, m)
				return err
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := newRig("a")
			tc.arm(r.p["a"])
			q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second}})
			br := breakers(1)
			m := &recMeter{}
			ex := r.exec(func(o *executor.Options) { o.Queues = q; o.Breakers = br })

			var err error
			v := mustPanic(t, func() { err = tc.call(ex, m) })
			if v != "provider exploded" {
				t.Fatalf("recovered %v (call returned %v), want the provider's panic", v, err)
			}
			if s := wantOneSettlement(t, m); s.Outcome != router.OutcomeError || s.Usage != nil || s.DeliveredBytes != 0 {
				t.Fatalf("settlement = %+v", s)
			}
			if q.Full("a") {
				t.Error("the queue slot was not released")
			}
			if !br.IsOpen("a") {
				t.Error("the breaker ticket was not reported as a failure")
			}
		})
	}
}

// panicStream panics on the first Recv.
type panicStream struct{}

func (panicStream) Recv() (provider.StreamEvent, error) { panic("provider exploded") }
func (panicStream) Close() error                        { return nil }

func TestMeterBeginPerAttempt(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	r.p["b"].complete = seq(ok("hi"))
	m := &recMeter{}
	if _, err := r.exec().Complete(context.Background(), route("a", "b"), provider.Request{}, m); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(m.begun, []string{"a", "b"}) {
		t.Fatalf("Begin models = %v, want [a b]: each attempt is charged on its own model", m.begun)
	}
	ends := m.settlements()
	if len(ends) != 2 || ends[0].Outcome != router.OutcomeError || ends[1].Outcome != router.OutcomeOK {
		t.Fatalf("settlements = %+v, want an error then ok", ends)
	}
}

func TestMeterBeginFailure(t *testing.T) {
	budget := errors.New("budget exceeded")

	t.Run("first attempt returns the error", func(t *testing.T) {
		r := newRig("a", "b")
		r.p["a"].complete = seq(ok("never"))
		q := reliability.NewQueues(map[string]reliability.QueueLimit{"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second}})
		m := &recMeter{beginErr: func(int) error { return budget }}
		d := route("a", "b")

		_, err := r.exec(func(o *executor.Options) { o.Queues = q }).Complete(context.Background(), d, provider.Request{}, m)
		if !errors.Is(err, budget) {
			t.Fatalf("err = %v, want the meter's error", err)
		}
		if calls(r.p["a"])+calls(r.p["b"]) != 0 {
			t.Error("an upstream call was made")
		}
		if len(m.settlements()) != 0 {
			t.Error("End called for a charge that never began")
		}
		if q.Full("a") {
			t.Error("the queue slot was not released")
		}
		wantAttempts(t, d)
	})

	t.Run("later attempt returns the last upstream error", func(t *testing.T) {
		r := newRig("a", "b")
		down := provErr(provider.KindUnavailable, "down")
		r.p["a"].complete = seq(fail(down))
		r.p["b"].complete = seq(ok("never"))
		m := &recMeter{beginErr: func(call int) error {
			if call == 1 {
				return budget
			}
			return nil
		}}
		d := route("a", "b")

		_, err := r.exec().Complete(context.Background(), d, provider.Request{}, m)
		if !errors.Is(err, down) || errors.Is(err, budget) {
			t.Fatalf("err = %v, want the upstream error of the first attempt", err)
		}
		if n := calls(r.p["b"]); n != 0 {
			t.Errorf("second model called %d times, want 0", n)
		}
		if ends := m.settlements(); len(ends) != 1 || ends[0].Outcome != router.OutcomeError {
			t.Fatalf("settlements = %+v, want the first attempt's only", ends)
		}
		wantAttempts(t, d, "a:error:unavailable")
	})
}

func TestHealthObservedOnSuccessOnly(t *testing.T) {
	const tick = 40 * time.Millisecond
	priors := map[string]health.Prior{"a": {TTFTMS: 500, DecodeTokensPerS: 50, Jitter: 0.1}}
	predict := func(h *health.Registry) float64 { return h.PredictedP95MS("a", 100) }
	fresh := predict(health.NewRegistry(priors, health.DefaultAlpha))
	// expected is what a registry predicts after seeing exactly obs.
	expected := func(obs ...health.Observation) float64 {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		for _, o := range obs {
			h.Observe("a", o)
		}
		return predict(h)
	}
	// The stepping clock advances one step per reading, and the executor reads
	// it at the start of an attempt, at the first event of a stream and at the
	// end.
	setup := func(h *health.Registry) func(*executor.Options) {
		return func(o *executor.Options) { o.Health = h; o.Now = steppingClock(tick); o.MaxAttempts = 1 }
	}
	newHealth := func() *health.Registry { return health.NewRegistry(priors, health.DefaultAlpha) }

	t.Run("a successful call", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].complete = seq(okUsage("hi", 10))
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		want := expected(health.Observation{Total: tick, Tokens: 10})
		if got := predict(h); got != want || got == fresh {
			t.Fatalf("prediction = %v, want %v (fresh %v)", got, want, fresh)
		}
	})

	t.Run("a failed call", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err == nil {
			t.Fatal("want an error")
		}
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v after a failure, want the prior %v", got, fresh)
		}
	})

	t.Run("a canceled call", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].complete = seq(fail(fmt.Errorf("p: %w", context.Canceled)))
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err == nil {
			t.Fatal("want an error")
		}
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v after a cancellation, want the prior %v", got, fresh)
		}
	})

	t.Run("a stream is observed at its end", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		last := provider.StreamEvent{FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: 10}}
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), delta("b"), {ev: last}}}})
		s, err := r.exec(setup(h)).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Recv(); err != nil {
			t.Fatal(err)
		}
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v before the stream ended, want the prior %v", got, fresh)
		}
		if _, err := drain(s); !errors.Is(err, io.EOF) {
			t.Fatal(err)
		}
		// TTFT is the time to the first event; the total runs to the end.
		want := expected(health.Observation{TTFT: tick, Total: 2 * tick, Tokens: 10, Streamed: true})
		if got := predict(h); got != want || got == fresh {
			t.Fatalf("prediction = %v, want %v (fresh %v)", got, want, fresh)
		}
	})

	t.Run("a partial stream", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), {err: provErr(provider.KindUnavailable, "cut")}}}})
		s, err := r.exec(setup(h)).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = drain(s)
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v after a partial stream, want the prior %v", got, fresh)
		}
	})

	t.Run("a stream closed early", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("a"), delta("b")}}})
		s, err := r.exec(setup(h)).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v after an early close, want the prior %v", got, fresh)
		}
	})

	t.Run("an empty stream has no first event to time", func(t *testing.T) {
		r, h := newRig("a"), newHealth()
		r.p["a"].stream = streams(streamResult{s: &stubStream{}})
		s, err := r.exec(setup(h)).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = drain(s)
		if got := predict(h); got != fresh {
			t.Fatalf("prediction = %v after an empty stream, want the prior %v", got, fresh)
		}
	})

	t.Run("a retried call is observed on the model that served it", func(t *testing.T) {
		priors := map[string]health.Prior{
			"a": {TTFTMS: 500, DecodeTokensPerS: 50, Jitter: 0.1},
			"b": {TTFTMS: 500, DecodeTokensPerS: 50, Jitter: 0.1},
		}
		h := health.NewRegistry(priors, health.DefaultAlpha)
		r := newRig("a", "b")
		r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
		r.p["b"].complete = seq(okUsage("hi", 10))
		if _, err := r.exec(func(o *executor.Options) { o.Health = h; o.Now = steppingClock(tick) }).Complete(context.Background(), route("a", "b"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		if got := h.PredictedP95MS("a", 100); got != fresh {
			t.Errorf("failed model moved to %v, want the prior %v", got, fresh)
		}
		if got := h.PredictedP95MS("b", 100); got == fresh {
			t.Error("serving model was not observed")
		}
	})
}

// A provider that sends no usage still decoded something, so the estimators
// are told the spec 11 estimate of the output, not zero tokens: with 0 the whole
// decode time would be charged to one token and inflate the per-token latency.
func TestHealthObservedWithEstimatedTokensWhenUsageMissing(t *testing.T) {
	// Longer than the TTFT prior, so a response's decode time is not zero and the
	// token count shows in the prediction.
	const tick = 2 * time.Second
	priors := map[string]health.Prior{"a": {TTFTMS: 500, DecodeTokensPerS: 50, Jitter: 0.1}}
	predict := func(h *health.Registry) float64 { return h.PredictedP95MS("a", 100) }
	expected := func(o health.Observation) float64 {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		h.Observe("a", o)
		return predict(h)
	}
	setup := func(h *health.Registry) func(*executor.Options) {
		return func(o *executor.Options) { o.Health = h; o.Now = steppingClock(tick); o.MaxAttempts = 1 }
	}

	t.Run("a response without usage", func(t *testing.T) {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		r := newRig("a")
		r.p["a"].complete = seq(ok(strings.Repeat("x", 41))) // 41 bytes: 11 tokens
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		want := expected(health.Observation{Total: tick, Tokens: 11})
		if got := predict(h); got != want {
			t.Fatalf("prediction = %v, want %v (11 estimated tokens)", got, want)
		}
		if zero := expected(health.Observation{Total: tick, Tokens: 0}); want == zero {
			t.Fatalf("the estimate changes nothing: %v equals the zero-token prediction", want)
		}
	})

	t.Run("a stream without usage", func(t *testing.T) {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		r := newRig("a")
		// 7 + 6 delivered bytes: 4 tokens.
		r.p["a"].stream = streams(streamResult{s: &stubStream{steps: []step{delta("xxxxxxx"), delta("xxxxxx")}}})
		s, err := r.exec(setup(h)).Stream(context.Background(), route("a"), provider.Request{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := drain(s); !errors.Is(err, io.EOF) {
			t.Fatal(err)
		}
		want := expected(health.Observation{TTFT: tick, Total: 2 * tick, Tokens: 4, Streamed: true})
		if got := predict(h); got != want {
			t.Fatalf("prediction = %v, want %v (4 estimated tokens)", got, want)
		}
		if zero := expected(health.Observation{TTFT: tick, Total: 2 * tick, Tokens: 0, Streamed: true}); want == zero {
			t.Fatalf("the estimate changes nothing: %v equals the zero-token prediction", want)
		}
	})

	t.Run("reported usage still wins over the estimate", func(t *testing.T) {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		r := newRig("a")
		r.p["a"].complete = seq(okUsage(strings.Repeat("x", 400), 3))
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		if got, want := predict(h), expected(health.Observation{Total: tick, Tokens: 3}); got != want {
			t.Fatalf("prediction = %v, want %v (3 reported tokens)", got, want)
		}
	})

	t.Run("an empty response has no tokens to estimate", func(t *testing.T) {
		h := health.NewRegistry(priors, health.DefaultAlpha)
		r := newRig("a")
		r.p["a"].complete = seq(ok(""))
		if _, err := r.exec(setup(h)).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
			t.Fatal(err)
		}
		if got, want := predict(h), expected(health.Observation{Total: tick, Tokens: 0}); got != want {
			t.Fatalf("prediction = %v, want %v", got, want)
		}
	})
}

func TestUnknownFallbackTarget(t *testing.T) {
	r := newRig("a")
	r.p["a"].complete = seq(ok("never"))
	_, err := r.exec().Complete(context.Background(), route("a", "ghost"), provider.Request{}, nil)
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Kind != provider.KindInternal {
		t.Fatalf("err = %v, want an internal *provider.Error", err)
	}
	if n := calls(r.p["a"]); n != 0 {
		t.Errorf("provider called %d times, want 0", n)
	}
}

// New keeps its signature and behavior: one attempt, no reliability features.
func TestNewIsASingleAttemptExecutor(t *testing.T) {
	p := &stubProvider{complete: seq(fail(provErr(provider.KindUnavailable, "down")))}
	ex := executor.New(map[string]executor.Target{"a": {Model: "a", Upstream: "up", Provider: p}}, nil)
	d := route("a")
	if _, err := ex.Complete(context.Background(), d, provider.Request{}, nil); err == nil {
		t.Fatal("want an error")
	}
	wantAttempts(t, d, "a:error:unavailable")
	if n := calls(p); n != 1 {
		t.Fatalf("provider called %d times, want 1", n)
	}
}
