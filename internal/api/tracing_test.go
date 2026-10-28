package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry/telemetrytest"
)

// chainRouter always chooses chosen and offers fallbacks.
type chainRouter struct {
	chosen    string
	fallbacks []string
}

func (r chainRouter) Decide(_ context.Context, req *router.Request) (*router.Decision, error) {
	return &router.Decision{
		ID:         "dec_failover",
		Tenant:     req.Tenant.Name,
		Strategy:   router.StrategyPinned,
		Profile:    req.Profile,
		Chosen:     r.chosen,
		Reason:     router.ReasonPinned,
		Fallbacks:  r.fallbacks,
		Attempts:   []router.Attempt{},
		Candidates: []router.Candidate{},
	}, nil
}

func targetsFor(cfg *config.Config, providers map[string]provider.Provider) map[string]executor.Target {
	targets := make(map[string]executor.Target, len(cfg.Models))
	for _, m := range cfg.Models {
		if p, ok := providers[m.Provider]; ok {
			targets[m.Name] = executor.Target{Model: m.Name, Upstream: m.UpstreamModel, Provider: p}
		}
	}
	return targets
}

// tracedServer is a server whose engine and executor trace into spans. Tweaks
// run last and may replace any option.
func tracedServer(t *testing.T, spans *telemetrytest.Spans, providers map[string]provider.Provider, tweaks ...func(*Options)) *Server {
	t.Helper()
	cfg := testConfig()
	tweaks = append([]func(*Options){func(o *Options) {
		o.Tracer = spans.Tracer()
		o.Router = router.NewEngine(router.EngineOptions{Models: cfg.Models, Tracer: spans.Tracer()})
		o.Executor = executor.NewWithOptions(executor.Options{Targets: targetsFor(cfg, providers), Tracer: spans.Tracer()})
	}}, tweaks...)
	return newServer(t, cfg, providers, tweaks...)
}

func TestSpansForRoutedRequest(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	cfg := testConfig()
	srv := tracedServer(t, spans, map[string]provider.Provider{"p-strong": &stubProvider{complete: replyWith("hello")}})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}
	reqID, decisionID := rec.Header().Get("X-Request-Id"), rec.Header().Get("X-RouteForge-Decision-Id")

	root := spans.One(t, "http.request")
	decide := spans.One(t, "route.decide")
	attempt := spans.One(t, "provider.attempt")
	if len(spans.Finished()) != 3 {
		t.Fatalf("finished spans = %d, want exactly the three named in the contract", len(spans.Finished()))
	}

	// One trace: the request span is the root, the other two its children.
	if root.ParentID != "" {
		t.Errorf("http.request has a parent %s, want a root span", root.ParentID)
	}
	for _, child := range []telemetrytest.Span{decide, attempt} {
		if child.ParentID != root.SpanID || child.TraceID != root.TraceID {
			t.Errorf("%s: parent %s in trace %s; want parent %s in trace %s", child.Name, child.ParentID, child.TraceID, root.SpanID, root.TraceID)
		}
	}

	wantAttrs := map[string]map[string]any{
		"http.request": {
			"http.request.method":       "POST",
			"url.path":                  "/v1/chat/completions",
			"http.response.status_code": int64(200),
			"routeforge.request_id":     reqID,
		},
		"route.decide": {
			"routeforge.strategy":       "pinned",
			"routeforge.profile":        cfg.Tenants[0].Profile,
			"routeforge.policy_version": "none",
			"routeforge.decision_id":    decisionID,
			"routeforge.model":          "m-strong",
			"routeforge.reason":         "pinned",
		},
		"provider.attempt": {
			"routeforge.model":          "m-strong",
			"routeforge.attempt":        int64(1),
			"routeforge.decision_id":    decisionID,
			"routeforge.policy_version": "none",
			"routeforge.outcome":        "ok",
		},
	}
	for _, sp := range []telemetrytest.Span{root, decide, attempt} {
		want := wantAttrs[sp.Name]
		for k, v := range want {
			if got, ok := sp.Attrs[k]; !ok || got != v {
				t.Errorf("%s: attribute %s = %v (present %v), want %v", sp.Name, k, got, ok, v)
			}
		}
		if len(sp.Attrs) != len(want) {
			t.Errorf("%s: attributes = %v, want exactly %v", sp.Name, sp.Attrs, want)
		}
		if sp.Error {
			t.Errorf("%s: status is Error for a successful request", sp.Name)
		}
	}
	if reqID == "" || decisionID == "" {
		t.Errorf("request id %q and decision id %q must both be set", reqID, decisionID)
	}
}

func TestAttemptSpansOnFailover(t *testing.T) {
	spans := telemetrytest.NewSpans(t)
	cfg := testConfig()
	providers := map[string]provider.Provider{
		"p-fast": &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
			return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "p-fast", Message: "down"}
		}},
		"p-strong": &stubProvider{complete: replyWith("from strong")},
	}
	srv := tracedServer(t, spans, providers, func(o *Options) {
		o.Router = chainRouter{chosen: "m-fast", fallbacks: []string{"m-strong"}}
		o.Executor = executor.NewWithOptions(executor.Options{
			Targets:     targetsFor(cfg, providers),
			MaxAttempts: 2,
			Sleep:       func(context.Context, time.Duration) error { return nil },
			Tracer:      spans.Tracer(),
		})
	})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body)
	}

	root := spans.One(t, "http.request")
	attempts := spans.Named("provider.attempt")
	if len(attempts) != 2 {
		t.Fatalf("provider.attempt spans = %d, want 2", len(attempts))
	}
	first, second := attempts[0], attempts[1]
	for _, sp := range attempts {
		if sp.ParentID != root.SpanID {
			t.Errorf("attempt %v: parent %s, want the request span %s", sp.Attrs["routeforge.attempt"], sp.ParentID, root.SpanID)
		}
		if sp.Attrs["routeforge.decision_id"] != "dec_failover" {
			t.Errorf("decision id = %v, want dec_failover", sp.Attrs["routeforge.decision_id"])
		}
	}

	if first.Attrs["routeforge.model"] != "m-fast" || first.Attrs["routeforge.attempt"] != int64(1) ||
		first.Attrs["routeforge.outcome"] != "error" || first.Attrs["routeforge.error_kind"] != "unavailable" {
		t.Errorf("first attempt = %v, want m-fast #1, outcome error, kind unavailable", first.Attrs)
	}
	// The status carries the error kind and nothing else.
	if !first.Error || first.Status != "unavailable" {
		t.Errorf("first attempt status: error %v, description %q; want Error with %q", first.Error, first.Status, "unavailable")
	}
	if second.Attrs["routeforge.model"] != "m-strong" || second.Attrs["routeforge.attempt"] != int64(2) ||
		second.Attrs["routeforge.outcome"] != "ok" {
		t.Errorf("second attempt = %v, want m-strong #2, outcome ok", second.Attrs)
	}
	if _, set := second.Attrs["routeforge.error_kind"]; set || second.Error {
		t.Errorf("second attempt carries an error: %v (Error %v)", second.Attrs, second.Error)
	}
}

func TestStreamAttemptSpans(t *testing.T) {
	t.Run("a complete stream", func(t *testing.T) {
		spans := telemetrytest.NewSpans(t)
		srv := tracedServer(t, spans, map[string]provider.Provider{"p-strong": streamingProvider(&stubStream{
			events: []provider.StreamEvent{{Delta: "a"}, {Delta: "b", FinishReason: "stop"}},
		})})

		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)
		if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "[DONE]") {
			t.Fatalf("status = %d, body %q; want a complete stream", rec.Code, rec.Body)
		}
		attempt := spans.One(t, "provider.attempt")
		if attempt.Attrs["routeforge.outcome"] != "ok" || attempt.Error {
			t.Errorf("attempt = %v (Error %v), want outcome ok", attempt.Attrs, attempt.Error)
		}
		// The attempt span lasts as long as the stream and has ended by the time
		// the request span does.
		finished := spans.Finished()
		if finished[len(finished)-1].Name != "http.request" {
			t.Errorf("spans ended in the order %v, want http.request last", names(finished))
		}
	})

	t.Run("a stream cut off after the first byte", func(t *testing.T) {
		spans := telemetrytest.NewSpans(t)
		srv := tracedServer(t, spans, map[string]provider.Provider{"p-strong": streamingProvider(&stubStream{
			events: []provider.StreamEvent{{Delta: "a"}},
			err:    &provider.Error{Kind: provider.KindUnavailable, Provider: "p-strong", Message: "connection lost"},
		})})

		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (the headers were out): %s", rec.Code, rec.Body)
		}
		attempt := spans.One(t, "provider.attempt")
		if attempt.Attrs["routeforge.outcome"] != "partial" || attempt.Attrs["routeforge.error_kind"] != "unavailable" ||
			!attempt.Error || attempt.Status != "unavailable" {
			t.Errorf("attempt = %v (Error %v, %q), want a partial outcome with kind unavailable", attempt.Attrs, attempt.Error, attempt.Status)
		}
	})
}

func names(spans []telemetrytest.Span) []string {
	out := make([]string, len(spans))
	for i, sp := range spans {
		out[i] = sp.Name
	}
	return out
}

func TestRequestSpanRecordsTheFinalStatus(t *testing.T) {
	type result struct {
		status int64
		isErr  bool
		spans  []string
	}
	run := func(t *testing.T, key string, p *stubProvider, ctx context.Context) result {
		t.Helper()
		spans := telemetrytest.NewSpans(t)
		srv := tracedServer(t, spans, map[string]provider.Provider{"p-strong": p})
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(okBody)).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		srv.Handler().ServeHTTP(httptest.NewRecorder(), req)
		root := spans.One(t, "http.request")
		return result{status: root.Attrs["http.response.status_code"].(int64), isErr: root.Error, spans: names(spans.Finished())}
	}

	t.Run("a client error is not a span error", func(t *testing.T) {
		got := run(t, "wrong-key", &stubProvider{}, context.Background())
		if got.status != 401 || got.isErr {
			t.Errorf("status %d (Error %v), want 401 without an error status", got.status, got.isErr)
		}
		if len(got.spans) != 1 {
			t.Errorf("spans = %v, want only http.request for a request refused before routing", got.spans)
		}
	})

	t.Run("a server error is", func(t *testing.T) {
		got := run(t, keyA, &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
			return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "p-strong", Message: "down"}
		}}, context.Background())
		if got.status != 502 || !got.isErr {
			t.Errorf("status %d (Error %v), want 502 with an error status", got.status, got.isErr)
		}
	})

	t.Run("a client that left is 499 and not an error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		got := run(t, keyA, &stubProvider{complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
			cancel()
			<-ctx.Done()
			return nil, ctx.Err()
		}}, ctx)
		if got.status != 499 || got.isErr {
			t.Errorf("status %d (Error %v), want 499 without an error status", got.status, got.isErr)
		}
	})
}

func TestNoPromptTextInSpans(t *testing.T) {
	const marker = "secret-prompt-marker"
	spans := telemetrytest.NewSpans(t)
	leaky := &provider.Error{Kind: provider.KindUnavailable, Provider: "p-fast", Message: "echoed " + marker}
	srv := tracedServer(t, spans, map[string]provider.Provider{
		"p-strong": &stubProvider{
			complete: replyWith("a reply with " + marker),
			stream: streamingProvider(&stubStream{
				events: []provider.StreamEvent{{Delta: marker}},
				err:    &provider.Error{Kind: provider.KindUnavailable, Provider: "p-strong", Message: "cut " + marker},
			}).stream,
		},
		"p-fast": &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) { return nil, leaky }},
	})
	h := srv.Handler()
	body := func(model string, stream bool) string {
		s := `{"model":"` + model + `","messages":[{"role":"user","content":"tell me about ` + marker + `"}]`
		if stream {
			s += `,"stream":true`
		}
		return s + "}"
	}
	// A completion, a stream that fails after output, a failing upstream whose
	// error text echoes the prompt, an explain request and a refused key.
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, body("m-strong", false), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, body("m-strong", true), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", keyA, body("m-fast", false), nil)
	do(t, h, http.MethodPost, "/v1/routing/explain", keyA, body("routeforge-auto", false), nil)
	do(t, h, http.MethodPost, "/v1/chat/completions", "wrong-key", body("m-strong", false), nil)

	finished := spans.Finished()
	if len(spans.Named("provider.attempt")) < 3 || len(spans.Named("route.decide")) < 3 {
		t.Fatalf("spans = %v: the test did not exercise decisions and attempts", names(finished))
	}
	for _, sp := range finished {
		for _, text := range sp.Texts() {
			if strings.Contains(text, marker) {
				t.Errorf("span %s carries prompt or completion text: %q", sp.Name, text)
			}
		}
	}
}
