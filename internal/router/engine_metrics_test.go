package router_test

import (
	"math"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

// metricEngine is an engine over the world that reports to m.
func (w *world) metricEngine(t *testing.T, reg *policy.Registry, m *telemetry.Metrics) *router.Engine {
	t.Helper()
	return router.NewEngine(router.EngineOptions{
		Models:   w.models,
		Prices:   w.prices(t),
		Policies: reg,
		Signals:  w.signals,
		Metrics:  m,
		NewID:    func() string { return "dec_test" },
		Now:      func() time.Time { return w.now },
	})
}

func TestEngineMetricsForPolicyDecision(t *testing.T) {
	w := newWorld()
	m := telemetry.NewMetrics()
	e := w.metricEngine(t, w.registry(t, "pv-1"), m)

	d := mustDecide(t, e, autoReq("quality", "balanced"))

	if v := telemetrytest.Value(t, m, "routeforge_route_decisions_total",
		labels{"model": d.Chosen, "reason": d.Reason, "strategy": "quality", "policy_version": "pv-1"}); v != 1 {
		t.Errorf("route_decisions_total = %g, want 1", v)
	}
	if h := telemetrytest.Hist(t, m, "routeforge_route_decision_seconds", labels{}); h.Count != 1 || h.Sum <= 0 {
		t.Errorf("route_decision_seconds: count = %d, sum = %g; want one positive observation", h.Count, h.Sum)
	}
	// The chosen model's calibrated p is what the histogram sees.
	h := telemetrytest.Hist(t, m, "routeforge_predicted_success", labels{"model": d.Chosen})
	want := *candidate(t, d, d.Chosen).P
	if h.Count != 1 || math.Abs(h.Sum-want) > 1e-12 {
		t.Errorf("predicted_success{%s}: count = %d, sum = %g; want 1, %g", d.Chosen, h.Count, h.Sum, want)
	}
}

func TestEngineMetricsForPinnedAndNoPolicy(t *testing.T) {
	w := newWorld()

	t.Run("pinned request", func(t *testing.T) {
		m := telemetry.NewMetrics()
		req := autoReq("", "balanced")
		req.Model = sonnet
		mustDecide(t, w.metricEngine(t, w.registry(t, "pv-1"), m), req)

		want := labels{"model": sonnet, "reason": "pinned", "strategy": "pinned", "policy_version": "none"}
		if v := telemetrytest.Value(t, m, "routeforge_route_decisions_total", want); v != 1 {
			t.Errorf("route_decisions_total%v = %g, want 1", want, v)
		}
		// A pinned decision is not scored, so it has no predicted success.
		telemetrytest.Absent(t, m, "routeforge_predicted_success")
	})

	t.Run("no active policy", func(t *testing.T) {
		m := telemetry.NewMetrics()
		d := mustDecide(t, w.metricEngine(t, policy.NewRegistry(poolNames, nil), m), autoReq("", "balanced"))

		want := labels{"model": d.Chosen, "reason": "strongest", "strategy": "strongest", "policy_version": "none"}
		if v := telemetrytest.Value(t, m, "routeforge_route_decisions_total", want); v != 1 {
			t.Errorf("route_decisions_total%v = %g, want 1", want, v)
		}
		telemetrytest.Absent(t, m, "routeforge_predicted_success")
	})
}

func TestEngineMetricsWhenNothingIsEligible(t *testing.T) {
	w := newWorld()
	w.signals.BreakerOpen = func(string) bool { return true }
	m := telemetry.NewMetrics()

	d, err := decideWith(w.metricEngine(t, w.registry(t, "pv-1"), m), autoReq("quality", "balanced"))
	if err == nil || d == nil {
		t.Fatalf("Decide() = %v, %v; want a decision with an error", d, err)
	}
	// The time to decide is still worth knowing; nothing was decided.
	if h := telemetrytest.Hist(t, m, "routeforge_route_decision_seconds", labels{}); h.Count != 1 {
		t.Errorf("route_decision_seconds count = %d, want 1", h.Count)
	}
	telemetrytest.Absent(t, m, "routeforge_route_decisions_total")
	telemetrytest.Absent(t, m, "routeforge_predicted_success")
}

func TestEngineMetricsSkipFailedRouting(t *testing.T) {
	w := newWorld()
	m := telemetry.NewMetrics()
	req := autoReq("", "balanced")
	req.Model = "no-such-model"

	if d, err := decideWith(w.metricEngine(t, w.registry(t, "pv-1"), m), req); err == nil || d != nil {
		t.Fatalf("Decide() = %v, %v; want an error and no decision", d, err)
	}
	telemetrytest.Absent(t, m, "routeforge_route_decisions_total")
	if h := telemetrytest.Hist(t, m, "routeforge_route_decision_seconds", labels{}); h.Count != 0 {
		t.Errorf("route_decision_seconds count = %d, want 0 for a request with no decision", h.Count)
	}
}
