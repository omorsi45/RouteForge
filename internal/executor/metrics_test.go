package executor_test

import (
	"context"
	"io"
	"math"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

func withMetrics(m *telemetry.Metrics) func(*executor.Options) {
	return func(o *executor.Options) { o.Metrics = m }
}

// steppedExec is an executor over p whose clock advances step on every read, so
// the timing of an attempt is a known number of steps.
func steppedExec(p provider.Provider, step time.Duration, m *telemetry.Metrics) *executor.Executor {
	return executor.NewWithOptions(executor.Options{
		Targets: map[string]executor.Target{"m": {Model: "m", Upstream: "up-m", Provider: p}},
		Now:     steppingClock(step),
		Metrics: m,
	})
}

func approx(got, want float64) bool { return math.Abs(got-want) < 1e-9 }

func TestMetricsRetryThenFailover(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	r.p["b"].complete = seq(ok("from b"))
	m := telemetry.NewMetrics()

	if _, err := r.exec(withMetrics(m)).Complete(context.Background(), route("a", "b"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}

	// The retry is counted against the model that failed, by the kind of the
	// failure; going on to another model is also a failover.
	if v := telemetrytest.Value(t, m, "routeforge_retries_total", labels{"model": "a", "kind": "unavailable"}); v != 1 {
		t.Errorf("retries_total{a,unavailable} = %g, want 1", v)
	}
	if v := telemetrytest.Value(t, m, "routeforge_failovers_total", labels{}); v != 1 {
		t.Errorf("failovers_total = %g, want 1", v)
	}
	for _, l := range []labels{{"model": "a", "outcome": "error"}, {"model": "b", "outcome": "ok"}} {
		if h := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", l); h.Count != 1 {
			t.Errorf("upstream_duration_seconds%v count = %d, want 1", l, h.Count)
		}
	}
}

func TestMetricsRetryOnTheSameModelIsNoFailover(t *testing.T) {
	r := newRig("a")
	r.p["a"].complete = seq(fail(provErr(provider.KindRateLimited, "slow down")), ok("fine"))
	m := telemetry.NewMetrics()

	if _, err := r.exec(withMetrics(m)).Complete(context.Background(), route("a"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	if v := telemetrytest.Value(t, m, "routeforge_retries_total", labels{"model": "a", "kind": "rate_limited"}); v != 1 {
		t.Errorf("retries_total{a,rate_limited} = %g, want 1", v)
	}
	if v := telemetrytest.Value(t, m, "routeforge_failovers_total", labels{}); v != 0 {
		t.Errorf("failovers_total = %g, want 0 for a retry on the same model", v)
	}
}

func TestMetricsNoRetryForNonRetryableFailure(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindBadRequest, "no")))
	m := telemetry.NewMetrics()

	if _, err := r.exec(withMetrics(m)).Complete(context.Background(), route("a", "b"), provider.Request{}, nil); err == nil {
		t.Fatal("Complete succeeded, want the bad request")
	}
	telemetrytest.Absent(t, m, "routeforge_retries_total")
	if v := telemetrytest.Value(t, m, "routeforge_failovers_total", labels{}); v != 0 {
		t.Errorf("failovers_total = %g, want 0", v)
	}
	if h := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", labels{"model": "a", "outcome": "error"}); h.Count != 1 {
		t.Errorf("upstream_duration_seconds{a,error} count = %d, want 1", h.Count)
	}
}

// The move from one model to the next is the failover. The third attempt
// stays on the last model, where it is a retry and nothing more.
func TestMetricsFailoverCountsOnlyTheMove(t *testing.T) {
	r := newRig("a", "b")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	r.p["b"].complete = seq(fail(provErr(provider.KindTimeout, "slow")), ok("fine"))
	m := telemetry.NewMetrics()
	d := route("a", "b")

	if _, err := r.exec(withMetrics(m)).Complete(context.Background(), d, provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	wantAttempts(t, d, "a:error:unavailable", "b:error:timeout", "b:ok")
	if v := telemetrytest.Value(t, m, "routeforge_failovers_total", labels{}); v != 1 {
		t.Errorf("failovers_total = %g, want 1 (a to b only)", v)
	}
	for _, l := range []labels{{"model": "a", "kind": "unavailable"}, {"model": "b", "kind": "timeout"}} {
		if v := telemetrytest.Value(t, m, "routeforge_retries_total", l); v != 1 {
			t.Errorf("retries_total%v = %g, want 1", l, v)
		}
	}
}

func TestMetricsRetryDeniedByBudgetIsNotCounted(t *testing.T) {
	r := newRig("a")
	r.p["a"].complete = seq(fail(provErr(provider.KindUnavailable, "down")))
	m := telemetry.NewMetrics()
	// A budget of zero retries: the request fails after one call.
	budget := reliability.NewRetryBudget(0, time.Second, 0, nil)

	_, err := r.exec(withMetrics(m), func(o *executor.Options) { o.RetryBudget = budget }).
		Complete(context.Background(), route("a"), provider.Request{}, nil)
	if err == nil {
		t.Fatal("Complete succeeded, want the unavailable error")
	}
	telemetrytest.Absent(t, m, "routeforge_retries_total")
}

func TestMetricsSkippedChosenTargetCountsAsFailover(t *testing.T) {
	r := newRig("a", "b")
	r.p["b"].complete = seq(ok("from b"))
	m := telemetry.NewMetrics()
	breakers := reliability.NewBreakers(reliability.BreakerConfig{
		ConsecutiveFailures: 1, ErrorRate: 1, MinRequests: 100, Window: time.Minute, OpenFor: time.Hour, HalfOpenProbes: 1,
	}, nil)
	ticket, _ := breakers.Allow("a")
	ticket.Done(false) // opens a's breaker

	if _, err := r.exec(withMetrics(m), func(o *executor.Options) { o.Breakers = breakers }).
		Complete(context.Background(), route("a", "b"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	if calls(r.p["a"]) != 0 {
		t.Fatal("the open model was called")
	}
	// The request left its chosen model without any attempt failing: that is a
	// failover, not a retry.
	if v := telemetrytest.Value(t, m, "routeforge_failovers_total", labels{}); v != 1 {
		t.Errorf("failovers_total = %g, want 1", v)
	}
	telemetrytest.Absent(t, m, "routeforge_retries_total")
}

func TestMetricsAttemptDurationUsesTheExecutorClock(t *testing.T) {
	m := telemetry.NewMetrics()
	p := &stubProvider{complete: okComplete}

	if _, err := steppedExec(p, 10*time.Millisecond, m).Complete(context.Background(), decision("m"), provider.Request{}, nil); err != nil {
		t.Fatal(err)
	}
	h := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", labels{"model": "m", "outcome": "ok"})
	if h.Count != 1 || !approx(h.Sum, 0.010) {
		t.Errorf("upstream_duration_seconds{m,ok}: count = %d, sum = %g; want 1, 0.010", h.Count, h.Sum)
	}
	// A non-streaming call has no first event to time.
	telemetrytest.Absent(t, m, "routeforge_upstream_ttft_seconds")
}

func TestMetricsStreamTTFTAndDuration(t *testing.T) {
	m := telemetry.NewMetrics()
	st := &stubStream{steps: []step{delta("a"), delta("b")}}
	p := &stubProvider{stream: streamOf(st)}

	s, err := steppedExec(p, 10*time.Millisecond, m).Stream(context.Background(), decision("m"), provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if got := s.TTFT(); got != 10*time.Millisecond {
		t.Errorf("TTFT() = %v, want 10ms", got)
	}
	h := telemetrytest.Hist(t, m, "routeforge_upstream_ttft_seconds", labels{"model": "m"})
	if h.Count != 1 || !approx(h.Sum, 0.010) {
		t.Errorf("upstream_ttft_seconds{m}: count = %d, sum = %g; want 1, 0.010", h.Count, h.Sum)
	}
	// The attempt is still open, so its duration is not known yet.
	telemetrytest.Absent(t, m, "routeforge_upstream_duration_seconds")

	if _, err := drain(s); err != io.EOF {
		t.Fatalf("drain error = %v, want io.EOF", err)
	}
	d := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", labels{"model": "m", "outcome": "ok"})
	if d.Count != 1 || !approx(d.Sum, 0.020) {
		t.Errorf("upstream_duration_seconds{m,ok}: count = %d, sum = %g; want 1, 0.020", d.Count, d.Sum)
	}
}

func TestMetricsStreamWithoutFirstEventHasNoTTFT(t *testing.T) {
	m := telemetry.NewMetrics()
	p := &stubProvider{stream: streamOf(&stubStream{})} // ends at once

	s, err := steppedExec(p, 10*time.Millisecond, m).Stream(context.Background(), decision("m"), provider.Request{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.TTFT(); got != 0 {
		t.Errorf("TTFT() = %v, want 0 for a stream with no event", got)
	}
	if _, err := drain(s); err != io.EOF {
		t.Fatalf("drain error = %v, want io.EOF", err)
	}
	telemetrytest.Absent(t, m, "routeforge_upstream_ttft_seconds")
}

func TestMetricsStreamOutcomes(t *testing.T) {
	boom := provErr(provider.KindUnavailable, "cut off")
	tests := []struct {
		name    string
		steps   []step
		consume func(t *testing.T, s *executor.Stream)
		outcome string
	}{
		{
			name:  "failure after output is partial",
			steps: []step{delta("a"), {err: boom}},
			consume: func(t *testing.T, s *executor.Stream) {
				if _, err := drain(s); err == nil || err == io.EOF {
					t.Fatalf("drain error = %v, want the stream failure", err)
				}
			},
			outcome: "partial",
		},
		{
			name:  "close before the end is canceled",
			steps: []step{delta("a"), delta("b")},
			consume: func(t *testing.T, s *executor.Stream) {
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			},
			outcome: "canceled",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := telemetry.NewMetrics()
			p := &stubProvider{stream: streamOf(&stubStream{steps: tc.steps})}
			s, err := steppedExec(p, time.Millisecond, m).Stream(context.Background(), decision("m"), provider.Request{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			tc.consume(t, s)
			h := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", labels{"model": "m", "outcome": tc.outcome})
			if h.Count != 1 {
				t.Errorf("upstream_duration_seconds{m,%s} count = %d, want 1", tc.outcome, h.Count)
			}
		})
	}
}
