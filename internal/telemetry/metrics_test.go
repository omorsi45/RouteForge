package telemetry_test

import (
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

// wantLabelNames are the exact families of spec 14 and 11 with their sorted
// label names.
var wantLabelNames = map[string][]string{
	"routeforge_requests_total":             {"model", "policy_version", "profile", "status", "strategy"},
	"routeforge_route_decisions_total":      {"model", "policy_version", "reason", "strategy"},
	"routeforge_route_decision_seconds":     nil,
	"routeforge_gateway_overhead_seconds":   {"stream"},
	"routeforge_upstream_ttft_seconds":      {"model"},
	"routeforge_upstream_duration_seconds":  {"model", "outcome"},
	"routeforge_retries_total":              {"kind", "model"},
	"routeforge_failovers_total":            nil,
	"routeforge_breaker_state":              {"model"},
	"routeforge_queue_depth":                {"model"},
	"routeforge_inflight":                   nil,
	"routeforge_rate_limited_total":         {"kind"},
	"routeforge_budget_rejections_total":    nil,
	"routeforge_cost_usd_total":             {"cost_source", "model"},
	"routeforge_tokens_total":               {"direction", "model", "usage_source"},
	"routeforge_budget_overshoot_usd_total": {"tenant"},
	"routeforge_predicted_success":          {"model"},
	"routeforge_policy_info":                {"version"},
}

// feedEveryEvent records one event of every kind.
func feedEveryEvent(m *telemetry.Metrics) {
	m.Request("m1", "quality", "balanced", "pv-1", 200)
	m.Decision("m1", "cheapest_passing", "quality", "pv-1")
	m.DecisionTime(2 * time.Millisecond)
	m.GatewayOverhead(false, 3*time.Millisecond)
	m.GatewayOverhead(true, 4*time.Millisecond)
	m.UpstreamTTFT("m1", 50*time.Millisecond)
	m.UpstreamDuration("m1", "ok", 120*time.Millisecond)
	m.Retry("m1", "unavailable")
	m.Failover()
	m.RateLimited("rpm")
	m.BudgetRejected()
	m.Cost("m1", "reported", 0.25)
	m.Tokens("m1", "reported", 10, 20)
	m.BudgetOvershoot("tenant-a", 0.5)
	m.PredictedSuccess("m1", 0.9)
	m.PolicyActivated("pv-1", "")
	m.SetStateSources(telemetry.StateSources{
		Models:       []string{"m1"},
		BreakerState: func(string) int { return 2 },
		QueueDepth:   func(string) int { return 3 },
		InFlight:     func() int { return 4 },
	})
}

func TestMetricNamesAndLabels(t *testing.T) {
	m := telemetry.NewMetrics()
	feedEveryEvent(m)

	got := telemetrytest.Gather(t, m)
	for name, want := range wantLabelNames {
		fam, ok := got[name]
		if !ok {
			t.Errorf("family %s is missing", name)
			continue
		}
		if names := fam.LabelNames(t); !slices.Equal(names, want) {
			t.Errorf("%s label names = %v, want %v", name, names, want)
		}
	}
	// Nothing else carries the project prefix.
	for name := range got {
		if strings.HasPrefix(name, "routeforge_") {
			if _, ok := wantLabelNames[name]; !ok {
				t.Errorf("unexpected family %s", name)
			}
		}
	}
	// The Go and process collectors are there too.
	if _, ok := got["go_goroutines"]; !ok {
		t.Error("go_goroutines is missing: the Go collector is not registered")
	}
}

func TestMetricKinds(t *testing.T) {
	m := telemetry.NewMetrics()
	feedEveryEvent(m)
	got := telemetrytest.Gather(t, m)

	histograms := []string{
		"routeforge_route_decision_seconds", "routeforge_gateway_overhead_seconds",
		"routeforge_upstream_ttft_seconds", "routeforge_upstream_duration_seconds",
		"routeforge_predicted_success",
	}
	gauges := []string{
		"routeforge_breaker_state", "routeforge_queue_depth", "routeforge_inflight", "routeforge_policy_info",
	}
	for name := range wantLabelNames {
		want := "counter"
		if slices.Contains(histograms, name) {
			want = "histogram"
		} else if slices.Contains(gauges, name) {
			want = "gauge"
		}
		if kind := got[name].Kind; kind != want {
			t.Errorf("%s is a %s, want a %s", name, kind, want)
		}
	}
}

func TestHistogramBuckets(t *testing.T) {
	m := telemetry.NewMetrics()
	feedEveryEvent(m)

	exponential := func(start float64, n int) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = start * math.Pow(2, float64(i))
		}
		return out
	}
	tests := []struct {
		name   string
		labels labels
		want   []float64
	}{
		{"routeforge_route_decision_seconds", labels{}, exponential(0.00001, 16)},
		{"routeforge_gateway_overhead_seconds", labels{"stream": "false"}, exponential(0.0001, 16)},
		{"routeforge_gateway_overhead_seconds", labels{"stream": "true"}, exponential(0.0001, 16)},
		{"routeforge_upstream_ttft_seconds", labels{"model": "m1"}, []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10}},
		{"routeforge_upstream_duration_seconds", labels{"model": "m1", "outcome": "ok"}, []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10}},
	}
	for _, tc := range tests {
		got := telemetrytest.Hist(t, m, tc.name, tc.labels).Bounds
		if len(got) != len(tc.want) {
			t.Errorf("%s%v has %d buckets, want %d: %v", tc.name, tc.labels, len(got), len(tc.want), got)
			continue
		}
		for i := range got {
			if math.Abs(got[i]-tc.want[i]) > 1e-12*tc.want[i] {
				t.Errorf("%s%v bucket %d = %g, want %g", tc.name, tc.labels, i, got[i], tc.want[i])
			}
		}
	}

	// Linear buckets 0.05, 0.10, ... 1.00.
	p := telemetrytest.Hist(t, m, "routeforge_predicted_success", labels{"model": "m1"}).Bounds
	if len(p) != 20 {
		t.Fatalf("predicted_success has %d buckets, want 20: %v", len(p), p)
	}
	for i, b := range p {
		if want := 0.05 * float64(i+1); math.Abs(b-want) > 1e-9 {
			t.Errorf("predicted_success bucket %d = %g, want %g", i, b, want)
		}
	}
}

func TestEventsRecordTheirValues(t *testing.T) {
	m := telemetry.NewMetrics()
	m.Request("m1", "quality", "balanced", "pv-1", 200)
	m.Request("m1", "quality", "balanced", "pv-1", 200)
	m.Request("", "", "", "", 499)
	m.Decision("m1", "pinned", "pinned", "")
	m.Retry("m1", "rate_limited")
	m.Retry("m1", "rate_limited")
	m.Failover()
	m.RateLimited("tpm")
	m.BudgetRejected()
	m.BudgetRejected()
	m.Cost("m1", "estimated", 0.25)
	m.Cost("m1", "estimated", 0.5)
	m.Tokens("m1", "reported", 10, 20)
	m.Tokens("m1", "reported", 1, 2)
	m.BudgetOvershoot("tenant-a", 0.125)
	m.GatewayOverhead(true, 5*time.Millisecond)
	m.GatewayOverhead(true, 7*time.Millisecond)
	m.DecisionTime(1500 * time.Microsecond)

	value := func(name string, l labels) float64 { return telemetrytest.Value(t, m, name, l) }
	want := []struct {
		name string
		l    labels
		v    float64
	}{
		// The status is the HTTP status as text, and no policy is "none".
		{"routeforge_requests_total", labels{"model": "m1", "strategy": "quality", "profile": "balanced", "policy_version": "pv-1", "status": "200"}, 2},
		{"routeforge_requests_total", labels{"model": "", "strategy": "", "profile": "", "policy_version": "none", "status": "499"}, 1},
		{"routeforge_route_decisions_total", labels{"model": "m1", "reason": "pinned", "strategy": "pinned", "policy_version": "none"}, 1},
		{"routeforge_retries_total", labels{"model": "m1", "kind": "rate_limited"}, 2},
		{"routeforge_failovers_total", labels{}, 1},
		{"routeforge_rate_limited_total", labels{"kind": "tpm"}, 1},
		{"routeforge_budget_rejections_total", labels{}, 2},
		{"routeforge_cost_usd_total", labels{"model": "m1", "cost_source": "estimated"}, 0.75},
		{"routeforge_tokens_total", labels{"model": "m1", "direction": "prompt", "usage_source": "reported"}, 11},
		{"routeforge_tokens_total", labels{"model": "m1", "direction": "completion", "usage_source": "reported"}, 22},
		{"routeforge_budget_overshoot_usd_total", labels{"tenant": "tenant-a"}, 0.125},
	}
	for _, tc := range want {
		if got := value(tc.name, tc.l); got != tc.v {
			t.Errorf("%s%v = %g, want %g", tc.name, tc.l, got, tc.v)
		}
	}

	// Durations are recorded in seconds.
	h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "true"})
	if h.Count != 2 || math.Abs(h.Sum-0.012) > 1e-12 {
		t.Errorf("overhead: count = %d, sum = %g; want 2, 0.012", h.Count, h.Sum)
	}
	d := telemetrytest.Hist(t, m, "routeforge_route_decision_seconds", labels{})
	if d.Count != 1 || math.Abs(d.Sum-0.0015) > 1e-12 {
		t.Errorf("decision time: count = %d, sum = %g; want 1, 0.0015", d.Count, d.Sum)
	}
}

func TestNilMetricsIsNoop(t *testing.T) {
	var m *telemetry.Metrics
	// None of these may panic.
	feedEveryEvent(m)
	m.PolicyActivated("pv-2", "pv-1")
	m.SetStateSources(telemetry.StateSources{})
	if m.Gatherer() != nil {
		t.Error("Gatherer on a nil Metrics should be nil")
	}
	h := m.Handler()
	if h == nil {
		t.Fatal("Handler on a nil Metrics returned nil")
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("nil Handler status = %d, want 404", rec.Code)
	}
}

func TestPolicyInfoFollowsActivation(t *testing.T) {
	m := telemetry.NewMetrics()
	active := func() []string {
		f := telemetrytest.Gather(t, m)["routeforge_policy_info"]
		var versions []string
		for _, s := range f.Series {
			if s.Value != 1 {
				t.Errorf("policy_info{version=%q} = %g, want 1", s.Labels["version"], s.Value)
			}
			versions = append(versions, s.Labels["version"])
		}
		slices.Sort(versions)
		return versions
	}
	check := func(step string, want ...string) {
		t.Helper()
		if got := active(); !slices.Equal(got, want) {
			t.Errorf("after %s: versions = %v, want %v", step, got, want)
		}
	}

	check("start")
	m.PolicyActivated("pv-1", "")
	check("first activation", "pv-1")
	m.PolicyActivated("pv-2", "pv-1")
	check("second activation", "pv-2")
	m.PolicyActivated("pv-1", "pv-2") // a rollback
	check("rollback", "pv-1")
	m.PolicyActivated("pv-1", "pv-1") // the same version again changes nothing
	check("repeat", "pv-1")
}

func TestStateSourcesAreReadAtScrape(t *testing.T) {
	m := telemetry.NewMetrics()

	// Before any source is set the in-flight gauge reads zero and the
	// per-model gauges have no series.
	if v := telemetrytest.Value(t, m, "routeforge_inflight", labels{}); v != 0 {
		t.Errorf("inflight without a source = %g, want 0", v)
	}
	telemetrytest.Absent(t, m, "routeforge_breaker_state")
	telemetrytest.Absent(t, m, "routeforge_queue_depth")

	states := map[string]int{"m1": 0, "m2": 2}
	depths := map[string]int{"m1": 5, "m2": 0}
	inflight := 7
	models := []string{"m1", "m2"}
	m.SetStateSources(telemetry.StateSources{
		Models:       models,
		BreakerState: func(model string) int { return states[model] },
		QueueDepth:   func(model string) int { return depths[model] },
		InFlight:     func() int { return inflight },
	})
	models[0] = "changed" // the sources keep their own copy of the model list

	check := func(wantStates, wantDepths map[string]int, wantInflight int) {
		t.Helper()
		for model, want := range wantStates {
			if got := telemetrytest.Value(t, m, "routeforge_breaker_state", labels{"model": model}); got != float64(want) {
				t.Errorf("breaker_state{%s} = %g, want %d", model, got, want)
			}
		}
		for model, want := range wantDepths {
			if got := telemetrytest.Value(t, m, "routeforge_queue_depth", labels{"model": model}); got != float64(want) {
				t.Errorf("queue_depth{%s} = %g, want %d", model, got, want)
			}
		}
		if got := telemetrytest.Value(t, m, "routeforge_inflight", labels{}); got != float64(wantInflight) {
			t.Errorf("inflight = %g, want %d", got, wantInflight)
		}
	}
	check(states, depths, 7)

	// The values are computed at each scrape, not stored.
	states["m1"], depths["m2"], inflight = 1, 9, 0
	check(map[string]int{"m1": 1, "m2": 2}, map[string]int{"m1": 5, "m2": 9}, 0)

	if n := len(telemetrytest.Gather(t, m)["routeforge_breaker_state"].Series); n != 2 {
		t.Errorf("breaker_state has %d series, want 2", n)
	}
}

func TestHandlerServesTextFormat(t *testing.T) {
	m := telemetry.NewMetrics()
	m.Request("m1", "quality", "balanced", "pv-1", 200)

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}
	body, err := io.ReadAll(rec.Body)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"# TYPE routeforge_requests_total counter",
		`routeforge_requests_total{model="m1",policy_version="pv-1",profile="balanced",status="200",strategy="quality"} 1`,
		"routeforge_inflight 0",
		"go_goroutines",
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("scrape lacks %q", want)
		}
	}
}

func TestRegistriesAreIndependent(t *testing.T) {
	a, b := telemetry.NewMetrics(), telemetry.NewMetrics()
	a.Failover()
	if v := telemetrytest.Value(t, a, "routeforge_failovers_total", labels{}); v != 1 {
		t.Errorf("first Metrics counted %g failovers, want 1", v)
	}
	if v := telemetrytest.Value(t, b, "routeforge_failovers_total", labels{}); v != 0 {
		t.Errorf("second Metrics saw the first one's failover: %g", v)
	}
}
