package main

import (
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

// policy_info names the policy that is active when the gateway starts and
// follows every later activation and rollback.
func TestWatchPolicyFollowsActivations(t *testing.T) {
	reg := policy.NewRegistry(nil, nil)
	for _, version := range []string{"pv-1", "pv-2"} {
		if err := reg.Add(&policy.Policy{Version: version}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := reg.Activate("pv-1", startupActor); err != nil {
		t.Fatal(err)
	}
	m := telemetry.NewMetrics()
	watchPolicy(reg, m)

	check := func(step, want string) {
		t.Helper()
		series := telemetrytest.Gather(t, m)["routeforge_policy_info"].Series
		if len(series) != 1 || series[0].Labels["version"] != want || series[0].Value != 1 {
			t.Errorf("after %s: policy_info = %+v, want only version %s set to 1", step, series, want)
		}
	}
	check("startup", "pv-1")
	if _, err := reg.Activate("pv-2", "admin"); err != nil {
		t.Fatal(err)
	}
	check("an activation", "pv-2")
	if _, err := reg.Rollback("admin"); err != nil {
		t.Fatal(err)
	}
	check("a rollback", "pv-1")
}

// A gateway without a policy has no policy_info series.
func TestWatchPolicyWithoutAPolicy(t *testing.T) {
	m := telemetry.NewMetrics()
	watchPolicy(policy.NewRegistry(nil, nil), m)
	telemetrytest.Absent(t, m, "routeforge_policy_info")
}

// The gateway reports to one set of metrics from every component, and serves
// them on /metrics: the wiring gives the API, the router, the executor, the
// accountant and the policy registry the same *Metrics.
func TestGatewayExposesMetrics(t *testing.T) {
	cfg, err := config.Load("../../deploy/configs/policy-fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for i := range cfg.Providers {
		cfg.Providers[i].Fake.TTFT = 0
		cfg.Providers[i].Fake.PerToken = 0
	}
	cfg.Tenants[0].DailyBudgetUSD = 10
	gw, err := buildGateway(cfg, func(string) string { return "" }, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(gw.Handler())
	t.Cleanup(ts.Close)

	const (
		auto       = `{"model":"routeforge-auto","messages":[{"role":"user","content":"hello"}]}`
		autoStream = `{"model":"routeforge-auto","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	)
	resp, raw := apiRequest(t, http.MethodPost, ts.URL+"/v1/chat/completions", demoKey, auto)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status %d, body %s", resp.StatusCode, raw)
	}
	model, version := resp.Header.Get("X-RouteForge-Model"), resp.Header.Get("X-RouteForge-Policy-Version")
	reason, strategy := resp.Header.Get("X-RouteForge-Reason"), "quality"
	if resp, raw = apiRequest(t, http.MethodPost, ts.URL+"/v1/chat/completions", demoKey, autoStream); resp.StatusCode != http.StatusOK {
		t.Fatalf("stream status %d, body %s", resp.StatusCode, raw)
	}
	if resp, raw = apiRequest(t, http.MethodPost, ts.URL+"/v1/chat/completions", "wrong-key", auto); resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad key status %d, body %s", resp.StatusCode, raw)
	}

	m := gw.metrics
	value := func(name string, l labels) float64 { return telemetrytest.Value(t, m, name, l) }

	// The routing engine, with the policy it served from.
	if !policyVersionRE.MatchString(version) {
		t.Fatalf("policy version header = %q, want a version", version)
	}
	decided := labels{"model": model, "reason": reason, "strategy": strategy, "policy_version": version}
	if v := value("routeforge_route_decisions_total", decided); v < 1 {
		t.Errorf("route_decisions_total%v = %g, want at least 1; have %+v", decided, v, telemetrytest.Gather(t, m)["routeforge_route_decisions_total"].Series)
	}
	if h := telemetrytest.Gather(t, m)["routeforge_route_decision_seconds"].Series; len(h) != 1 || h[0].Count != 2 {
		t.Errorf("route_decision_seconds = %+v, want one series of 2 observations", h)
	}
	// The API.
	served := labels{"model": model, "strategy": strategy, "profile": "balanced", "policy_version": version, "status": "200"}
	if v := value("routeforge_requests_total", served); v < 1 {
		t.Errorf("requests_total%v = %g, want at least 1", served, v)
	}
	if v := value("routeforge_requests_total", labels{"model": "", "strategy": "", "profile": "", "policy_version": "none", "status": "401"}); v != 1 {
		t.Errorf("requests_total{401} = %g, want 1", v)
	}
	for _, stream := range []string{"false", "true"} {
		if h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": stream}); h.Count != 1 {
			t.Errorf("gateway_overhead_seconds{stream=%s} count = %d, want 1", stream, h.Count)
		}
	}
	// The executor and the accountant: both calls were served and settled.
	if h := telemetrytest.Hist(t, m, "routeforge_upstream_duration_seconds", labels{"model": model, "outcome": "ok"}); h.Count < 1 {
		t.Errorf("upstream_duration_seconds{%s,ok} count = %d, want at least 1", model, h.Count)
	}
	if h := telemetrytest.Gather(t, m)["routeforge_upstream_ttft_seconds"].Series; len(h) == 0 {
		t.Error("the streamed call recorded no upstream_ttft_seconds")
	}
	var tokens float64
	for _, s := range telemetrytest.Gather(t, m)["routeforge_tokens_total"].Series {
		tokens += s.Value
	}
	if tokens <= 0 {
		t.Errorf("tokens_total adds up to %g, want a positive number", tokens)
	}
	var cost float64
	for _, s := range telemetrytest.Gather(t, m)["routeforge_cost_usd_total"].Series {
		cost += s.Value
	}
	if cost <= 0 {
		t.Errorf("cost_usd_total adds up to %g, want a positive number", cost)
	}
	// The policy registry: the startup policy is the active version.
	if v := value("routeforge_policy_info", labels{"version": version}); v != 1 {
		t.Errorf("policy_info{%s} = %g, want 1", version, v)
	}
	if n := len(telemetrytest.Gather(t, m)["routeforge_policy_info"].Series); n != 1 {
		t.Errorf("policy_info has %d series, want 1", n)
	}

	// The scrape carries the state gauges of every configured model.
	resp, raw = apiRequest(t, http.MethodGet, ts.URL+"/metrics", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /metrics status %d, body %s", resp.StatusCode, raw)
	}
	scrape := string(raw)
	for _, mod := range cfg.Models {
		for _, series := range []string{"routeforge_breaker_state", "routeforge_queue_depth"} {
			if want := series + `{model="` + mod.Name + `"} 0`; !strings.Contains(scrape, want) {
				t.Errorf("scrape lacks %q", want)
			}
		}
	}
	if !strings.Contains(scrape, "routeforge_inflight 0") {
		t.Error(`scrape lacks "routeforge_inflight 0" once all requests are done`)
	}
}

// scrapeText returns the body of GET /metrics.
func scrapeText(t *testing.T, base string) string {
	t.Helper()
	resp, raw := apiRequest(t, http.MethodGet, base+"/metrics", "", "")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /metrics status %d, body %s", resp.StatusCode, raw)
	}
	return string(raw)
}

// A breaker that opened shows as state 2 in the scrape.
func TestGatewayBreakerStateIsScraped(t *testing.T) {
	upstream, _ := scriptedUpstream(t, http.StatusServiceUnavailable)
	cfg := upstreamConfig(t, upstream, "reliability:\n  max_attempts: 1\n  breaker:\n    consecutive_failures: 1\n")
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))

	const closed, open = `routeforge_breaker_state{model="m-up"} 0`, `routeforge_breaker_state{model="m-up"} 2`
	if body := scrapeText(t, base); !strings.Contains(body, closed) {
		t.Fatalf("before any failure the scrape lacks %q:\n%s", closed, body)
	}
	resp := post(t, base, e2eKey, chatBody("m-up", ""))
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
	body := scrapeText(t, base)
	if !strings.Contains(body, open) {
		t.Errorf("after the failure the scrape lacks %q:\n%s", open, body)
	}
	if !strings.Contains(body, `routeforge_upstream_duration_seconds_count{model="m-up",outcome="error"} 1`) {
		t.Errorf("the failed attempt is not in upstream_duration_seconds:\n%s", body)
	}
}

// The requests the gateway holds show in routeforge_inflight, and the ones
// waiting for a model slot in routeforge_queue_depth, while they run.
func TestGatewayInflightAndQueueDepthAreScraped(t *testing.T) {
	upstream, entered, release := holdingUpstream(t)
	cfg := upstreamConfig(t, upstream, "")
	cfg.Models[0].MaxConcurrency, cfg.Models[0].QueueSize, cfg.Models[0].QueueTimeout = 1, 1, 10*time.Second
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))
	t.Cleanup(release)

	status := make(chan int, 2)
	send := func() {
		resp := post(t, base, e2eKey, chatBody("m-up", ""))
		_ = resp.Body.Close()
		status <- resp.StatusCode
	}
	go send()
	select {
	case <-entered:
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("the first request never reached the upstream")
	}
	go send() // finds the only slot taken and waits in the line

	const busy = "routeforge_inflight 2"
	deadline := time.Now().Add(5 * time.Second)
	for {
		body := scrapeText(t, base)
		if strings.Contains(body, busy) && strings.Contains(body, `routeforge_queue_depth{model="m-up"} 1`) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the scrape never showed 2 requests in flight and 1 queued:\n%s", body)
		}
		time.Sleep(5 * time.Millisecond)
	}

	release()
	for range 2 {
		select {
		case got := <-status:
			if got != http.StatusOK {
				t.Fatalf("request status = %d, want 200", got)
			}
		case <-time.After(5 * time.Second): // a failure guard, not a wait
			t.Fatal("a held request never finished")
		}
	}
	body := scrapeText(t, base)
	for _, want := range []string{"routeforge_inflight 0", `routeforge_queue_depth{model="m-up"} 0`} {
		if !strings.Contains(body, want) {
			t.Errorf("after both requests finished the scrape lacks %q:\n%s", want, body)
		}
	}
}
