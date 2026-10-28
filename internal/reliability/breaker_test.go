package reliability

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

// fakeClock is a manually advanced clock, safe for concurrent use.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock {
	return &fakeClock{t: time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// specBreaker is the default breaker of spec 10.
func specBreaker() BreakerConfig {
	return BreakerConfig{
		ConsecutiveFailures: 5,
		ErrorRate:           0.5,
		MinRequests:         10,
		Window:              30 * time.Second,
		OpenFor:             15 * time.Second,
		HalfOpenProbes:      2,
	}
}

const model = "m"

// newBreakers returns breakers over the spec defaults and the clock they read.
func newBreakers() (*Breakers, *fakeClock) {
	clk := newFakeClock()
	return NewBreakers(specBreaker(), clk.Now), clk
}

// run sends one request through the breaker of model and reports its outcome.
func run(t *testing.T, b *Breakers, model string, success bool) {
	t.Helper()
	ticket, ok := b.Allow(model)
	if !ok {
		t.Fatalf("Allow(%q) rejected while %v", model, b.State(model))
	}
	ticket.Done(success)
}

// runAll sends one request per outcome; true is a success.
func runAll(t *testing.T, b *Breakers, model string, outcomes ...bool) {
	t.Helper()
	for _, success := range outcomes {
		run(t, b, model, success)
	}
}

func wantState(t *testing.T, b *Breakers, model string, want State) {
	t.Helper()
	if got := b.State(model); got != want {
		t.Fatalf("State(%q) = %v, want %v", model, got, want)
	}
}

// tripped returns breakers whose model is open, with the clock at the moment
// of opening.
func tripped(t *testing.T) (*Breakers, *fakeClock) {
	t.Helper()
	b, clk := newBreakers()
	runAll(t, b, model, false, false, false, false, false)
	wantState(t, b, model, StateOpen)
	return b, clk
}

func TestStateValues(t *testing.T) {
	// The breaker_state metric exposes these numbers.
	if StateClosed != 0 || StateHalfOpen != 1 || StateOpen != 2 {
		t.Fatalf("states = %d, %d, %d; want 0, 1, 2", StateClosed, StateHalfOpen, StateOpen)
	}
}

func TestBreakerOpensAfterConsecutiveFailures(t *testing.T) {
	b, _ := newBreakers()
	runAll(t, b, model, false, false, false, false)
	wantState(t, b, model, StateClosed)
	if b.IsOpen(model) {
		t.Fatal("IsOpen after 4 failures")
	}

	run(t, b, model, false)
	wantState(t, b, model, StateOpen)
	if !b.IsOpen(model) {
		t.Fatal("IsOpen = false after 5 consecutive failures")
	}
	if _, ok := b.Allow(model); ok {
		t.Fatal("Allow admitted a request through an open breaker")
	}
}

func TestBreakerSuccessResetsConsecutiveFailures(t *testing.T) {
	b, _ := newBreakers()
	// Eight failures in nine requests, but never five in a row, and fewer than
	// the ten requests the error rate needs.
	runAll(t, b, model, false, false, false, false, true, false, false, false, false)
	wantState(t, b, model, StateClosed)
}

func TestBreakerOpensOnErrorRate(t *testing.T) {
	b, _ := newBreakers()
	// Failures and successes alternate, so five in a row is never reached.
	runAll(t, b, model, false, true, false, true, false, true, false, true, false)
	wantState(t, b, model, StateClosed) // nine requests: below the minimum

	run(t, b, model, true) // the tenth request: 5 of 10 failed
	wantState(t, b, model, StateOpen)
}

func TestBreakerIgnoresBelowMinRequests(t *testing.T) {
	b, _ := newBreakers()
	runAll(t, b, model, false, true, false, true, false, true, false, true, false)
	wantState(t, b, model, StateClosed) // 5 of 9 failed, not consecutive

	// Below the rate the breaker stays closed with plenty of requests.
	b2, _ := newBreakers()
	runAll(t, b2, model, true, true, true, true, true, true, true, true, false, false, false, false)
	wantState(t, b2, model, StateClosed) // 4 of 12 failed
}

func TestBreakerHalfOpenAfterOpenFor(t *testing.T) {
	b, clk := tripped(t)

	clk.Advance(15*time.Second - time.Nanosecond)
	wantState(t, b, model, StateOpen)
	if _, ok := b.Allow(model); ok {
		t.Fatal("Allow admitted a request just before OpenFor elapsed")
	}

	clk.Advance(time.Nanosecond)
	wantState(t, b, model, StateHalfOpen)
	if b.IsOpen(model) {
		t.Fatal("IsOpen = true for a half-open breaker: a probe could never be admitted")
	}
	if _, ok := b.Allow(model); !ok {
		t.Fatal("Allow rejected the first probe")
	}
}

func TestBreakerHalfOpenProbeLimit(t *testing.T) {
	b, clk := tripped(t)
	clk.Advance(15 * time.Second)

	for i := 1; i <= 2; i++ {
		if _, ok := b.Allow(model); !ok {
			t.Fatalf("probe %d rejected", i)
		}
	}
	if _, ok := b.Allow(model); ok {
		t.Fatal("third concurrent probe admitted")
	}
	wantState(t, b, model, StateHalfOpen)
}

func TestBreakerProbeSuccessCloses(t *testing.T) {
	b, clk := tripped(t)
	clk.Advance(15 * time.Second)

	first, _ := b.Allow(model)
	second, ok := b.Allow(model)
	if !ok {
		t.Fatal("second probe rejected")
	}
	first.Done(true)
	wantState(t, b, model, StateClosed)

	// The other probe was issued before the breaker closed; its late failure
	// says nothing about the closed breaker.
	second.Done(false)
	wantState(t, b, model, StateClosed)

	// Closing reset the counters: the failures that opened the breaker are gone.
	runAll(t, b, model, false, false, false, false)
	wantState(t, b, model, StateClosed)
	run(t, b, model, false)
	wantState(t, b, model, StateOpen)
}

func TestBreakerProbeFailureReopens(t *testing.T) {
	b, clk := tripped(t)
	clk.Advance(15 * time.Second)

	probe, ok := b.Allow(model)
	if !ok {
		t.Fatal("probe rejected")
	}
	probe.Done(false)
	wantState(t, b, model, StateOpen)
	if _, ok := b.Allow(model); ok {
		t.Fatal("Allow admitted a request right after a failed probe")
	}

	// OpenFor counts from the failed probe, not from the first opening.
	clk.Advance(14 * time.Second)
	wantState(t, b, model, StateOpen)
	clk.Advance(time.Second)
	wantState(t, b, model, StateHalfOpen)
}

func TestBreakerStaleTicketDoesNotCount(t *testing.T) {
	b, _ := newBreakers()
	stale, ok := b.Allow(model)
	if !ok {
		t.Fatal("Allow rejected")
	}
	runAll(t, b, model, false, false, false, false, false)
	wantState(t, b, model, StateOpen)

	// Admitted while closed, finished while open: the outcome is not recorded.
	stale.Done(true)
	wantState(t, b, model, StateOpen)
}

func TestBreakerWindowSlides(t *testing.T) {
	// Five failures among nine requests, none consecutive and too few requests
	// to open the breaker.
	old := []bool{false, true, false, true, false, true, false, true, false}

	t.Run("failures inside the window still count", func(t *testing.T) {
		b, clk := newBreakers()
		runAll(t, b, model, old...)
		clk.Advance(29 * time.Second)
		run(t, b, model, true) // the tenth request in the window
		wantState(t, b, model, StateOpen)
	})

	t.Run("failures older than the window are forgotten", func(t *testing.T) {
		b, clk := newBreakers()
		runAll(t, b, model, old...)
		clk.Advance(30 * time.Second)
		run(t, b, model, true) // the same request, but now the only one in the window
		wantState(t, b, model, StateClosed)
	})

	t.Run("a stale streak of failures is forgotten too", func(t *testing.T) {
		b, clk := newBreakers()
		runAll(t, b, model, false, false, false, false)
		clk.Advance(time.Minute)
		run(t, b, model, false) // the fifth failure, but a minute after the fourth
		wantState(t, b, model, StateClosed)
	})
}

func TestBreakersAreIndependentPerModel(t *testing.T) {
	b, _ := newBreakers()
	runAll(t, b, "a", false, false, false, false, false)
	wantState(t, b, "a", StateOpen)
	wantState(t, b, "b", StateClosed)
	run(t, b, "b", true)
	if b.IsOpen("b") || !b.IsOpen("a") {
		t.Fatalf("IsOpen(a), IsOpen(b) = %v, %v; want true, false", b.IsOpen("a"), b.IsOpen("b"))
	}
	if got := b.State("never-used"); got != StateClosed || b.IsOpen("never-used") {
		t.Fatalf("State of an unknown model = %v, want closed", got)
	}
}

func TestTicketDoneOnRejectedTicketIsNoop(t *testing.T) {
	b, _ := tripped(t)
	ticket, ok := b.Allow(model)
	if ok {
		t.Fatal("Allow admitted a request through an open breaker")
	}
	ticket.Done(false) // must not panic or change anything
	ticket.Done(true)
	wantState(t, b, model, StateOpen)
}

func TestNewBreakersDefaultsClock(t *testing.T) {
	b := NewBreakers(specBreaker(), nil)
	run(t, b, model, true)
	wantState(t, b, model, StateClosed)
}

func TestCountsAsFailure(t *testing.T) {
	kinds := func(k provider.ErrorKind) error { return &provider.Error{Kind: k, Provider: "p", Message: "x"} }
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"canceled", context.Canceled, false},
		{"wrapped canceled", fmt.Errorf("p: %w", context.Canceled), false},
		{"unavailable", kinds(provider.KindUnavailable), true},
		{"timeout", kinds(provider.KindTimeout), true},
		{"rate limited", kinds(provider.KindRateLimited), true},
		{"internal", kinds(provider.KindInternal), true},
		{"bad request", kinds(provider.KindBadRequest), false},
		{"auth", kinds(provider.KindAuth), false},
		{"context length", kinds(provider.KindContextLength), false},
		{"wrapped unavailable", fmt.Errorf("attempt 2: %w", kinds(provider.KindUnavailable)), true},
		{"wrapped bad request", fmt.Errorf("attempt 2: %w", kinds(provider.KindBadRequest)), false},
		{"deadline exceeded", context.DeadlineExceeded, true},
		{"network error", &net.OpError{Op: "dial", Err: errors.New("refused")}, true},
		{"unclassified", errors.New("boom"), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CountsAsFailure(tc.err); got != tc.want {
				t.Errorf("CountsAsFailure(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestBreakerConcurrent(t *testing.T) {
	t.Run("half-open admits exactly the probe limit", func(t *testing.T) {
		b, clk := tripped(t)
		clk.Advance(15 * time.Second)

		const callers = 64
		var (
			granted atomic.Int32
			start   = make(chan struct{})
			wg      sync.WaitGroup
		)
		for range callers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				if _, ok := b.Allow(model); ok {
					granted.Add(1)
				}
			}()
		}
		close(start)
		wg.Wait()
		if got := granted.Load(); got != 2 {
			t.Fatalf("%d probes admitted, want 2", got)
		}
	})

	t.Run("failing callers open the breaker", func(t *testing.T) {
		b, _ := newBreakers()
		const callers = 8
		var (
			granted atomic.Int32
			wg      sync.WaitGroup
		)
		for range callers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range 100 {
					if ticket, ok := b.Allow(model); ok {
						granted.Add(1)
						ticket.Done(false)
					}
					b.State(model)
					b.IsOpen(model)
				}
			}()
		}
		wg.Wait()
		wantState(t, b, model, StateOpen)
		// Five failures open the breaker; at most one request per other caller
		// was already admitted when it did.
		if got := granted.Load(); got < 5 || got > 5+callers-1 {
			t.Fatalf("%d requests admitted, want between 5 and %d", got, 5+callers-1)
		}
	})

	t.Run("many models at once", func(t *testing.T) {
		b, clk := newBreakers()
		var wg sync.WaitGroup
		for i := range 16 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				name := fmt.Sprintf("model-%d", i%4)
				for j := range 200 {
					if ticket, ok := b.Allow(name); ok {
						ticket.Done(j%3 != 0)
					}
					if j%50 == 0 {
						clk.Advance(time.Second)
					}
					b.State(name)
				}
			}()
		}
		wg.Wait()
		for i := range 4 {
			if s := b.State(fmt.Sprintf("model-%d", i)); s < StateClosed || s > StateOpen {
				t.Errorf("model-%d in unknown state %d", i, int(s))
			}
		}
	})
}
