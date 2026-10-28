package api

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

type labels = map[string]string

// metricsServer is a server over testConfig that reports to m.
func metricsServer(t *testing.T, m *telemetry.Metrics, providers map[string]provider.Provider, opts ...func(*Options)) *Server {
	t.Helper()
	opts = append([]func(*Options){func(o *Options) { o.Metrics = m }}, opts...)
	return newServer(t, testConfig(), providers, opts...)
}

// fakeClock is a manually advanced time source.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock { return &fakeClock{t: time.Date(2026, 10, 27, 12, 0, 0, 0, time.UTC)} }

func (c *fakeClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

// advancingRouter spends time in the gateway, before the upstream is called.
type advancingRouter struct {
	router.Router
	clock *fakeClock
	by    time.Duration
}

func (r advancingRouter) Decide(ctx context.Context, req *router.Request) (*router.Decision, error) {
	r.clock.advance(r.by)
	return r.Router.Decide(ctx, req)
}

// clockedStream is a stream whose second event takes a long time.
type clockedStream struct {
	clock *fakeClock
	late  time.Duration
	sent  int
}

func (s *clockedStream) Recv() (provider.StreamEvent, error) {
	s.sent++
	switch s.sent {
	case 1:
		return provider.StreamEvent{Delta: "a"}, nil
	case 2:
		s.clock.advance(s.late)
		return provider.StreamEvent{Delta: "b"}, nil
	}
	return provider.StreamEvent{}, io.EOF
}

func (s *clockedStream) Close() error { return nil }

func scrape(t *testing.T, srv *Server) string {
	t.Helper()
	rec := do(t, srv.Handler(), http.MethodGet, "/metrics", "", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /metrics status = %d, want 200", rec.Code)
	}
	return rec.Body.String()
}

func TestRequestsTotalCountsStatus(t *testing.T) {
	m := telemetry.NewMetrics()
	cfg := testConfig()
	strong := &stubProvider{complete: replyWith("hello")}
	srv := newServer(t, cfg, map[string]provider.Provider{"p-strong": strong, "p-fast": &stubProvider{
		complete: func(context.Context, provider.Request) (*provider.Response, error) {
			return nil, &provider.Error{Kind: provider.KindRateLimited, Provider: "p-fast", Message: "slow down"}
		},
	}}, func(o *Options) { o.Metrics = m })
	h := srv.Handler()
	profile := cfg.Tenants[0].Profile

	if rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}
	fastBody := strings.Replace(okBody, "m-strong", "m-fast", 1)
	if rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, fastBody, nil); rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: %s", rec.Code, rec.Body)
	}
	if rec := do(t, h, http.MethodPost, "/v1/chat/completions", "wrong-key", okBody, nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401: %s", rec.Code, rec.Body)
	}

	for _, want := range []labels{
		{"model": "m-strong", "strategy": "pinned", "profile": profile, "policy_version": "none", "status": "200"},
		{"model": "m-fast", "strategy": "pinned", "profile": profile, "policy_version": "none", "status": "429"},
		// A request that never reached routing still counts, with empty labels.
		{"model": "", "strategy": "", "profile": "", "policy_version": "none", "status": "401"},
	} {
		if v := telemetrytest.Value(t, m, "routeforge_requests_total", want); v != 1 {
			t.Errorf("requests_total%v = %g, want 1", want, v)
		}
	}
	// Only chat requests are counted: nothing else is a request of the gateway.
	do(t, h, http.MethodGet, "/healthz", "", "", nil)
	do(t, h, http.MethodGet, "/v1/models", keyA, "", nil)
	do(t, h, http.MethodGet, "/metrics", "", "", nil)
	if n := len(telemetrytest.Gather(t, m)["routeforge_requests_total"].Series); n != 3 {
		t.Errorf("requests_total has %d series, want 3", n)
	}
}

func TestRequestsTotalLabelsAnAutoRoutedDecision(t *testing.T) {
	m := telemetry.NewMetrics()
	srv := newServer(t, testConfig(), map[string]provider.Provider{
		"p-strong": &stubProvider{complete: replyWith("hello")},
		"p-fast":   &stubProvider{complete: replyWith("hello")},
	}, func(o *Options) { o.Metrics = m })

	body := `{"model":"routeforge-auto","messages":[{"role":"user","content":"hi"}]}`
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body, map[string]string{"X-RouteForge-Profile": "strict"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}
	// The basic router chooses the strongest model; the labels are the
	// decision's own.
	want := labels{"model": "m-strong", "strategy": "strongest", "profile": "strict", "policy_version": "none", "status": "200"}
	if v := telemetrytest.Value(t, m, "routeforge_requests_total", want); v != 1 {
		t.Errorf("requests_total%v = %g, want 1; have %+v", want, v, telemetrytest.Gather(t, m)["routeforge_requests_total"].Series)
	}
}

func TestExplainIsNotARequest(t *testing.T) {
	m := telemetry.NewMetrics()
	srv := metricsServer(t, m, map[string]provider.Provider{"p-strong": &stubProvider{}})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/routing/explain", keyA, okBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}
	telemetrytest.Absent(t, m, "routeforge_requests_total")
	telemetrytest.Absent(t, m, "routeforge_gateway_overhead_seconds")
}

func TestClientGoneCounted499(t *testing.T) {
	m := telemetry.NewMetrics()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srv := metricsServer(t, m, map[string]provider.Provider{"p-strong": &stubProvider{
		complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
			cancel() // the client leaves during the upstream call
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}}, func(o *Options) { o.Logger = slog.New(slog.DiscardHandler) })

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(okBody)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyA)
	req.Header.Set("Content-Type", "application/json")
	srv.Handler().ServeHTTP(httptest.NewRecorder(), req)

	want := labels{"model": "m-strong", "strategy": "pinned", "profile": testConfig().Tenants[0].Profile, "policy_version": "none", "status": "499"}
	if v := telemetrytest.Value(t, m, "routeforge_requests_total", want); v != 1 {
		t.Errorf("requests_total%v = %g, want 1; have %+v", want, v, telemetrytest.Gather(t, m)["routeforge_requests_total"].Series)
	}
	// What the handler wrote (nothing) is not what was counted.
	if n := len(telemetrytest.Gather(t, m)["routeforge_requests_total"].Series); n != 1 {
		t.Errorf("requests_total has %d series, want only the 499", n)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	m := telemetry.NewMetrics()
	srv := metricsServer(t, m, nil)

	// No key is needed: the proxy restricts the endpoint.
	rec := do(t, srv.Handler(), http.MethodGet, "/metrics", "", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}
	if !strings.Contains(rec.Body.String(), "routeforge_inflight") {
		t.Errorf("scrape lacks routeforge_inflight:\n%s", rec.Body)
	}
	if rec := do(t, srv.Handler(), http.MethodPost, "/metrics", "", "", nil); rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST /metrics status = %d, want 405", rec.Code)
	}
}

func TestMetricsEndpointNeedsMetrics(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	if rec := do(t, srv.Handler(), http.MethodGet, "/metrics", "", "", nil); rec.Code != http.StatusNotFound {
		t.Errorf("GET /metrics without metrics: status = %d, want 404", rec.Code)
	}
}

func TestNoPromptTextInMetrics(t *testing.T) {
	const marker = "secret-prompt-marker"
	m := telemetry.NewMetrics()
	srv := metricsServer(t, m, map[string]provider.Provider{
		"p-strong": &stubProvider{
			complete: replyWith("reply with " + marker),
			stream: streamingProvider(&stubStream{events: []provider.StreamEvent{
				{Delta: marker}, {Delta: "done", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 3, CompletionTokens: 4}},
			}}).stream,
		},
	})
	h := srv.Handler()

	prompt := func(model string, stream bool) string {
		s := `{"model":"` + model + `","messages":[{"role":"user","content":"tell me about ` + marker + `"}]`
		if stream {
			s += `,"stream":true`
		}
		return s + "}"
	}
	// A pinned call, a streamed one, an auto-routed one, an explain request and
	// a request the gateway rejects, all carrying the marker.
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, prompt("m-strong", false), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, prompt("m-strong", true), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, prompt("routeforge-auto", false), nil)
	do(t, h, http.MethodPost, "/v1/routing/explain", keyA, prompt("routeforge-auto", false), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", "wrong-key", prompt("m-strong", false), nil)

	body := scrape(t, srv)
	if !strings.Contains(body, "routeforge_requests_total") {
		t.Fatal("the scrape has no request counter, so the test did not exercise the metrics")
	}
	if strings.Contains(body, marker) {
		t.Errorf("the scrape contains prompt or completion text:\n%s", body)
	}
}

func TestGatewayOverheadNonStreaming(t *testing.T) {
	clock := newFakeClock()
	m := telemetry.NewMetrics()
	cfg := testConfig()
	prov := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		clock.advance(30 * time.Millisecond) // time spent upstream
		return &provider.Response{Content: "hi", FinishReason: "stop"}, nil
	}}
	exec := executor.New(map[string]executor.Target{
		"m-strong": {Model: "m-strong", Upstream: "up-strong", Provider: prov},
	}, clock.now)
	srv := newServer(t, cfg, nil, func(o *Options) {
		o.Metrics, o.Now, o.Executor = m, clock.now, exec
		o.Router = advancingRouter{Router: o.Router, clock: clock, by: 5 * time.Millisecond} // time spent in the gateway
	})

	if rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}

	h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "false"})
	if h.Count != 1 || !closeTo(h.Sum, 0.005) {
		t.Errorf("overhead{stream=false}: count = %d, sum = %g; want 1, 0.005 (35ms in the handler minus 30ms upstream)", h.Count, h.Sum)
	}
	if _, ok := telemetrytest.Find(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "true"}); ok {
		t.Error("a non-streaming request recorded a stream=true overhead")
	}
}

func TestGatewayOverheadCountsAFailedRequestOnce(t *testing.T) {
	clock := newFakeClock()
	m := telemetry.NewMetrics()
	prov := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		clock.advance(30 * time.Millisecond)
		return nil, &provider.Error{Kind: provider.KindBadRequest, Provider: "p-strong", Message: "no"}
	}}
	exec := executor.New(map[string]executor.Target{
		"m-strong": {Model: "m-strong", Upstream: "up-strong", Provider: prov},
	}, clock.now)
	srv := newServer(t, testConfig(), nil, func(o *Options) {
		o.Metrics, o.Now, o.Executor = m, clock.now, exec
		o.Router = advancingRouter{Router: o.Router, clock: clock, by: 5 * time.Millisecond}
	})

	if rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body)
	}
	h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "false"})
	if h.Count != 1 || !closeTo(h.Sum, 0.005) {
		t.Errorf("overhead{stream=false}: count = %d, sum = %g; want 1, 0.005", h.Count, h.Sum)
	}
}

func TestGatewayOverheadStreamingStopsAtTheFirstByte(t *testing.T) {
	clock := newFakeClock()
	m := telemetry.NewMetrics()
	prov := &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
		clock.advance(40 * time.Millisecond) // the wait for the first upstream event
		// The second event takes 100ms, which is after the first byte.
		return &clockedStream{clock: clock, late: 100 * time.Millisecond}, nil
	}}
	exec := executor.New(map[string]executor.Target{
		"m-strong": {Model: "m-strong", Upstream: "up-strong", Provider: prov},
	}, clock.now)
	srv := newServer(t, testConfig(), nil, func(o *Options) {
		o.Metrics, o.Now, o.Executor = m, clock.now, exec
		o.Router = advancingRouter{Router: o.Router, clock: clock, by: 5 * time.Millisecond}
	})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("status = %d, body %q; want a complete stream", rec.Code, rec.Body)
	}

	h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "true"})
	if h.Count != 1 || !closeTo(h.Sum, 0.005) {
		t.Errorf("overhead{stream=true}: count = %d, sum = %g; want 1, 0.005 (45ms to the first byte minus the 40ms first-event wait)", h.Count, h.Sum)
	}
	if _, ok := telemetrytest.Find(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "false"}); ok {
		t.Error("a streaming request recorded a stream=false overhead")
	}
}

func TestGatewayOverheadStreamingExcludesEarlierAttempts(t *testing.T) {
	clock := newFakeClock()
	m := telemetry.NewMetrics()
	var calls int
	prov := &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
		calls++
		if calls == 1 {
			clock.advance(20 * time.Millisecond) // a failed first attempt
			return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "p-strong", Message: "down"}
		}
		clock.advance(40 * time.Millisecond)
		return &clockedStream{clock: clock, late: 100 * time.Millisecond}, nil
	}}
	exec := executor.NewWithOptions(executor.Options{
		Targets: map[string]executor.Target{
			"m-strong": {Model: "m-strong", Upstream: "up-strong", Provider: prov},
		},
		Now:         clock.now,
		MaxAttempts: 2,
		Sleep:       func(context.Context, time.Duration) error { return nil },
	})
	srv := newServer(t, testConfig(), nil, func(o *Options) {
		o.Metrics, o.Now, o.Executor = m, clock.now, exec
		o.Router = advancingRouter{Router: o.Router, clock: clock, by: 5 * time.Millisecond}
	})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)
	if rec.Code != http.StatusOK || calls != 2 {
		t.Fatalf("status = %d after %d upstream calls; want 200 after 2", rec.Code, calls)
	}
	h := telemetrytest.Hist(t, m, "routeforge_gateway_overhead_seconds", labels{"stream": "true"})
	if h.Count != 1 || !closeTo(h.Sum, 0.005) {
		t.Errorf("overhead{stream=true}: count = %d, sum = %g; want 1, 0.005 (65ms to the first byte minus 20ms and 40ms upstream)", h.Count, h.Sum)
	}
}

func TestGatewayOverheadNotRecordedBeforeExecution(t *testing.T) {
	m := telemetry.NewMetrics()
	srv := metricsServer(t, m, map[string]provider.Provider{"p-strong": &stubProvider{}})
	h := srv.Handler()

	do(t, h, http.MethodPost, "/v1/chat/completions", "wrong-key", okBody, nil)                             // 401
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, `{"model":"no-such-model","messages":[]}`, nil) // 400
	telemetrytest.Absent(t, m, "routeforge_gateway_overhead_seconds")
}

func closeTo(got, want float64) bool {
	d := got - want
	return d < 1e-9 && d > -1e-9
}
