package router_test

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel/trace"

	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

// tracedEngine is an engine over the world that traces into spans.
func (w *world) tracedEngine(t *testing.T, reg *policy.Registry, spans *telemetrytest.Spans) *router.Engine {
	t.Helper()
	return router.NewEngine(router.EngineOptions{
		Models:   w.models,
		Prices:   w.prices(t),
		Policies: reg,
		Signals:  w.signals,
		Tracer:   spans.Tracer(),
		NewID:    func() string { return "dec_test" },
	})
}

func TestEngineDecideSpan(t *testing.T) {
	w := newWorld()
	spans := telemetrytest.NewSpans(t)
	e := w.tracedEngine(t, w.registry(t, "pv-1"), spans)

	// The span is a child of the span in the context.
	ctx, parent := spans.Tracer().Start(context.Background(), "parent")
	d, err := e.Decide(ctx, autoReq("quality", "balanced"))
	parent.End()
	if err != nil {
		t.Fatal(err)
	}

	decide := spans.One(t, "route.decide")
	if decide.ParentID != spans.One(t, "parent").SpanID {
		t.Errorf("route.decide parent = %s, want the span of the context", decide.ParentID)
	}
	want := map[string]any{
		"routeforge.strategy":       "quality",
		"routeforge.profile":        "balanced",
		"routeforge.policy_version": "pv-1",
		"routeforge.decision_id":    "dec_test",
		"routeforge.model":          d.Chosen,
		"routeforge.reason":         d.Reason,
	}
	for k, v := range want {
		if got := decide.Attrs[k]; got != v {
			t.Errorf("attribute %s = %v, want %v", k, got, v)
		}
	}
	if len(decide.Attrs) != len(want) || decide.Error {
		t.Errorf("attributes = %v (Error %v), want exactly %v and no error", decide.Attrs, decide.Error, want)
	}
}

// What the engine asks of its signals runs inside the span, so anything a
// signal traces nests under route.decide.
func TestEngineSignalsRunInsideTheDecideSpan(t *testing.T) {
	w := newWorld()
	spans := telemetrytest.NewSpans(t)
	var seen context.Context
	w.signals.BudgetRemainingNanos = func(ctx context.Context, _ string) (int64, bool, error) {
		seen = ctx
		return 0, false, nil
	}

	if _, err := w.tracedEngine(t, w.registry(t, "pv-1"), spans).Decide(context.Background(), autoReq("quality", "balanced")); err != nil {
		t.Fatal(err)
	}
	if seen == nil {
		t.Fatal("the budget signal was not asked")
	}
	if got := trace.SpanFromContext(seen).SpanContext().SpanID().String(); got != spans.One(t, "route.decide").SpanID {
		t.Errorf("the signal saw span %s, want the route.decide span", got)
	}
}

func TestEngineDecideSpanWithoutAPolicyOrAParent(t *testing.T) {
	w := newWorld()
	spans := telemetrytest.NewSpans(t)
	req := autoReq("", "balanced")
	req.Model = sonnet

	if _, err := w.tracedEngine(t, policy.NewRegistry(poolNames, nil), spans).Decide(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	decide := spans.One(t, "route.decide")
	if decide.ParentID != "" {
		t.Errorf("parent = %s, want a root span when the context has none", decide.ParentID)
	}
	if decide.Attrs["routeforge.policy_version"] != "none" || decide.Attrs["routeforge.strategy"] != "pinned" || decide.Attrs["routeforge.reason"] != "pinned" {
		t.Errorf("attributes = %v, want policy_version none, strategy and reason pinned", decide.Attrs)
	}
}

func TestEngineDecideSpanErrors(t *testing.T) {
	t.Run("nothing eligible keeps the decision and names the kind", func(t *testing.T) {
		w := newWorld()
		w.signals.BreakerOpen = func(string) bool { return true }
		spans := telemetrytest.NewSpans(t)

		d, err := w.tracedEngine(t, w.registry(t, "pv-1"), spans).Decide(context.Background(), autoReq("quality", "balanced"))
		if err == nil || d == nil {
			t.Fatalf("Decide() = %v, %v; want a decision with an error", d, err)
		}
		decide := spans.One(t, "route.decide")
		if !decide.Error || decide.Status != "no_eligible_model" {
			t.Errorf("status: error %v, description %q; want Error with %q", decide.Error, decide.Status, "no_eligible_model")
		}
		// Nothing was chosen, but the rest of the decision is on the span.
		if decide.Attrs["routeforge.decision_id"] != "dec_test" || decide.Attrs["routeforge.strategy"] != "quality" ||
			decide.Attrs["routeforge.policy_version"] != "pv-1" {
			t.Errorf("attributes = %v, want the decision's id, strategy and policy version", decide.Attrs)
		}
		for _, k := range []string{"routeforge.model", "routeforge.reason"} {
			if v, set := decide.Attrs[k]; set {
				t.Errorf("%s = %v, want it left out when nothing was chosen", k, v)
			}
		}
	})

	kinds := []struct {
		name string
		edit func(req *router.Request)
		kind string
	}{
		{"unknown model", func(req *router.Request) { req.Model = "no-such-model" }, "unknown_model"},
		{"model not allowed", func(req *router.Request) {
			req.Model = sonnet
			req.Tenant.AllowedModels = []string{gpt5}
		}, "model_not_allowed"},
		{"strategy unavailable", func(req *router.Request) { req.Strategy = router.StrategyLatency }, "strategy_unavailable"},
	}
	for _, tc := range kinds {
		t.Run(tc.name, func(t *testing.T) {
			w := newWorld()
			spans := telemetrytest.NewSpans(t)
			req := autoReq("quality", "balanced")
			tc.edit(req)

			if d, err := w.tracedEngine(t, w.registry(t, "pv-1"), spans).Decide(context.Background(), req); err == nil || d != nil {
				t.Fatalf("Decide() = %v, %v; want an error and no decision", d, err)
			}
			decide := spans.One(t, "route.decide")
			if !decide.Error || decide.Status != tc.kind {
				t.Errorf("status: error %v, description %q; want Error with %q", decide.Error, decide.Status, tc.kind)
			}
			// There is no decision, so the span has no decision id.
			if _, set := decide.Attrs["routeforge.decision_id"]; set {
				t.Errorf("attributes = %v: a request with no decision has no decision id", decide.Attrs)
			}
			if decide.Attrs["routeforge.profile"] != "balanced" {
				t.Errorf("profile = %v, want the request's profile", decide.Attrs["routeforge.profile"])
			}
		})
	}
}
