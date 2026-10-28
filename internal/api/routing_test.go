package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/store"
)

// poolModels are the policy models in pool order.
var poolModels = []string{
	"gpt-5",
	"claude-sonnet-4",
	"gemini-2.5-pro",
	"gemini-2.5-flash",
	"deepseek-v3.1-terminus",
	"qwen3-235b-a22b-2507",
}

const (
	autoBody       = `{"model":"routeforge-auto","messages":[{"role":"user","content":"hello world"}]}`
	autoStreamBody = `{"model":"routeforge-auto","stream":true,"messages":[{"role":"user","content":"hello world"}]}`
)

// fixedScore scores every prompt the same.
type fixedScore float64

func (f fixedScore) Raw(*policy.Features, *[8]float64) float64 { return float64(f) }

// poolPolicy returns a synthetic policy over the pool models. Every model has
// one calibration block with p 0.9 and lcb 0.8, and every profile asks for 0.5,
// so the cheapest model passes.
func poolPolicy(version string) *policy.Policy {
	p := &policy.Policy{
		Version:       version,
		PredictorType: "stub",
		Strongest:     "gemini-2.5-pro",
		Cheapest:      "qwen3-235b-a22b-2507",
		Profiles:      map[string]policy.Profile{},
		Rules: policy.StaticRules{
			FamilyKeywords: map[string][]string{"qa": {"who"}},
			FamilyModel:    map[string]string{"qa": "qwen3-235b-a22b-2507"},
			FamilyOrder:    []string{"qa"},
			DefaultFamily:  "qa",
		},
	}
	for i := range p.DenseStd {
		p.DenseStd[i] = 1
	}
	for _, name := range config.Profiles {
		p.Profiles[name] = policy.Profile{Name: name, Lambda: 0.5}
	}
	for _, name := range poolModels {
		p.Models = append(p.Models, policy.Model{
			Name:      name,
			Predictor: fixedScore(0.5),
			Calibration: policy.Calibration{Blocks: []policy.Block{
				{Lo: 0, Hi: 1, N: 100, K: 90, P: 0.9, LCB: 0.8},
			}},
			ExpectedOutputTokens: 100,
		})
	}
	return p
}

// poolConfig is a two-tenant configuration with one model per policy model,
// all served by provider p-pool.
func poolConfig() *config.Config {
	c := &config.Config{
		Tenants: []config.Tenant{
			{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey(keyA)}},
			{Name: "tenant-b", APIKeySHA256: []string{auth.HashKey(keyB)}, AllowStrategyOverride: true},
		},
		Providers: []config.Provider{{Name: "p-pool", Type: "fake"}},
	}
	for _, name := range poolModels {
		c.Models = append(c.Models, config.Model{Name: name, Provider: "p-pool", Strongest: name == "gemini-2.5-pro"})
	}
	c.ApplyDefaults()
	if err := c.Validate(); err != nil {
		panic("api pool config is invalid: " + err.Error())
	}
	return c
}

// poolPrices prices the pool models at distinct whole-dollar rates.
func poolPrices(t *testing.T) map[string]accounting.Price {
	t.Helper()
	prices := make(map[string]accounting.Price, len(poolModels))
	for i, name := range poolModels {
		p, err := accounting.NewPrice(name, float64(10-i), float64(2*(10-i)))
		if err != nil {
			t.Fatalf("NewPrice(%s): %v", name, err)
		}
		prices[name] = p
	}
	return prices
}

// activate adds p to reg and makes it the active policy.
func activate(t *testing.T, reg *policy.Registry, p *policy.Policy) {
	t.Helper()
	if err := reg.Add(p); err != nil {
		t.Fatalf("Add(%s): %v", p.Version, err)
	}
	if _, err := reg.Activate(p.Version, "test"); err != nil {
		t.Fatalf("Activate(%s): %v", p.Version, err)
	}
}

// poolEnv is a Server over the quality engine with policy "pv-1" active.
// Tweaks run before the engine and the server are built.
type poolEnv struct {
	cfg  *config.Config
	sig  router.Signals
	reg  *policy.Registry
	prov *stubProvider
	opts []func(*Options)
	srv  *Server
}

func newPoolEnv(t *testing.T, tweaks ...func(*poolEnv)) *poolEnv {
	t.Helper()
	e := &poolEnv{
		cfg:  poolConfig(),
		reg:  policy.NewRegistry(poolModels, nil),
		prov: &stubProvider{complete: replyWith("ok")},
	}
	activate(t, e.reg, poolPolicy("pv-1"))
	for _, tweak := range tweaks {
		tweak(e)
	}
	engine := router.NewEngine(router.EngineOptions{
		Models:   e.cfg.Models,
		Prices:   poolPrices(t),
		Policies: e.reg,
		Signals:  e.sig,
	})
	opts := append([]func(*Options){func(o *Options) {
		o.Router = engine
		o.Policies = e.reg
	}}, e.opts...)
	e.srv = newServer(t, e.cfg, map[string]provider.Provider{"p-pool": e.prov}, opts...)
	return e
}

func (e *poolEnv) handler() http.Handler { return e.srv.Handler() }

// post sends body to path as key.
func (e *poolEnv) post(t *testing.T, path, key, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, e.handler(), http.MethodPost, path, key, body, hdr)
}

func (e *poolEnv) get(t *testing.T, path, key string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, e.handler(), http.MethodGet, path, key, "", nil)
}

// decisionFrom decodes a decision JSON document.
func decisionFrom(t *testing.T, body []byte) *router.Decision {
	t.Helper()
	var d router.Decision
	if err := json.Unmarshal(body, &d); err != nil {
		t.Fatalf("body is not a decision: %v; body %s", err, body)
	}
	return &d
}

// fetchDecision reads a decision through the endpoint and requires a 200.
func (e *poolEnv) fetchDecision(t *testing.T, key, id string) *router.Decision {
	t.Helper()
	rec := e.get(t, "/v1/routing/decisions/"+id, key)
	if rec.Code != http.StatusOK {
		t.Fatalf("GET decision %s: status = %d, want 200; body %s", id, rec.Code, rec.Body)
	}
	return decisionFrom(t, rec.Body.Bytes())
}

// spyStore wraps a decision store and reports every Put.
type spyStore struct {
	store.DecisionStore
	puts  atomic.Int64
	onPut func(*router.Decision)
}

func (s *spyStore) Put(d *router.Decision) error {
	s.puts.Add(1)
	if s.onPut != nil {
		s.onPut(d)
	}
	return s.DecisionStore.Put(d)
}

func withDecisions(ds store.DecisionStore) func(*poolEnv) {
	return func(e *poolEnv) { e.opts = append(e.opts, func(o *Options) { o.Decisions = ds }) }
}

func TestExplainReturnsDecisionWithoutCallingProvider(t *testing.T) {
	spy := &spyStore{DecisionStore: store.NewRing(10)}
	env := newPoolEnv(t, withDecisions(spy))

	rec := env.post(t, "/v1/routing/explain", keyA, autoBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	if n := env.prov.callCount(); n != 0 {
		t.Errorf("provider was called %d times, want 0", n)
	}

	d := decisionFrom(t, rec.Body.Bytes())
	if !strings.HasPrefix(d.ID, "dec_") || d.Tenant != "tenant-a" || d.PolicyVersion != "pv-1" {
		t.Errorf("id/tenant/version = %q/%q/%q", d.ID, d.Tenant, d.PolicyVersion)
	}
	if d.Strategy != router.StrategyQuality || d.Profile != "balanced" || d.Chosen == "" || d.Reason == "" {
		t.Errorf("strategy/profile/chosen/reason = %q/%q/%q/%q", d.Strategy, d.Profile, d.Chosen, d.Reason)
	}
	if d.Lambda == nil || *d.Lambda != 0.5 || !d.GuaranteeApplies {
		t.Errorf("lambda/guarantee = %v/%v, want 0.5/true", d.Lambda, d.GuaranteeApplies)
	}
	if len(d.Candidates) != len(poolModels) {
		t.Fatalf("got %d candidates, want %d", len(d.Candidates), len(poolModels))
	}
	for i, c := range d.Candidates {
		if c.Model != poolModels[i] || !c.Eligible {
			t.Errorf("candidate %d = %s eligible %v, want %s eligible", i, c.Model, c.Eligible, poolModels[i])
		}
		if c.P == nil || c.LCB == nil || *c.P != 0.9 || *c.LCB != 0.8 {
			t.Errorf("%s: p/lcb = %v/%v, want 0.9/0.8", c.Model, c.P, c.LCB)
		}
	}
	if len(d.Attempts) != 0 {
		t.Errorf("attempts = %+v, want none", d.Attempts)
	}

	// The decision is returned, not stored.
	if n := spy.puts.Load(); n != 0 {
		t.Errorf("explain stored %d decisions, want 0", n)
	}
	wantError(t, env.get(t, "/v1/routing/decisions/"+d.ID, keyA), http.StatusNotFound, CodeNotFound)
}

func TestExplainHonorsRoutingHeaders(t *testing.T) {
	env := newPoolEnv(t)
	rec := env.post(t, "/v1/routing/explain", keyB, autoBody, map[string]string{
		"X-RouteForge-Strategy": "cheapest",
		"X-RouteForge-Profile":  "strict",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	d := decisionFrom(t, rec.Body.Bytes())
	if d.Strategy != router.StrategyCheapest || d.Profile != "strict" || d.Reason != router.ReasonCheapest {
		t.Errorf("strategy/profile/reason = %q/%q/%q", d.Strategy, d.Profile, d.Reason)
	}
	if d.Chosen != "qwen3-235b-a22b-2507" {
		t.Errorf("chosen = %q, want the cheapest model", d.Chosen)
	}
}

func TestExplainAppliesChatRules(t *testing.T) {
	cases := []struct {
		name     string
		key      string
		body     string
		hdr      map[string]string
		wantCode int
		wantErr  string
	}{
		{"no key", "", autoBody, nil, http.StatusUnauthorized, CodeInvalidAPIKey},
		{"unknown key", "test-key-unknown", autoBody, nil, http.StatusUnauthorized, CodeInvalidAPIKey},
		{"bad profile", keyA, autoBody, map[string]string{"X-RouteForge-Profile": "reckless"}, http.StatusBadRequest, CodeInvalidRequest},
		{"bad strategy", keyB, autoBody, map[string]string{"X-RouteForge-Strategy": "random"}, http.StatusBadRequest, CodeInvalidRequest},
		{"strategy override forbidden", keyA, autoBody, map[string]string{"X-RouteForge-Strategy": "cheapest"}, http.StatusForbidden, CodeStrategyOverrideForbidden},
		{"bad deadline", keyA, autoBody, map[string]string{"X-RouteForge-Deadline-Ms": "soon"}, http.StatusBadRequest, CodeInvalidRequest},
		{"malformed body", keyA, `{`, nil, http.StatusBadRequest, CodeInvalidRequest},
		{"unsupported parameter", keyA, `{"model":"routeforge-auto","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}`, nil, http.StatusBadRequest, CodeUnsupportedParameter},
		{"unknown model", keyA, `{"model":"nope","messages":[{"role":"user","content":"x"}]}`, nil, http.StatusBadRequest, CodeInvalidRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := newPoolEnv(t)
			rec := env.post(t, "/v1/routing/explain", tc.key, tc.body, tc.hdr)
			wantError(t, rec, tc.wantCode, tc.wantErr)
			if n := env.prov.callCount(); n != 0 {
				t.Errorf("provider was called %d times, want 0", n)
			}
		})
	}
}

func TestExplainNoEligibleIncludesDecision(t *testing.T) {
	env := newPoolEnv(t, func(e *poolEnv) {
		for i := range e.cfg.Models {
			e.cfg.Models[i].ContextWindow = 1 // no request fits
		}
	})
	rec := env.post(t, "/v1/routing/explain", keyA, autoBody, nil)

	wantError(t, rec, http.StatusServiceUnavailable, CodeNoEligibleModel)
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
	var body struct {
		Decision *router.Decision `json:"decision"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Decision == nil || len(body.Decision.Candidates) != len(poolModels) {
		t.Fatalf("decision = %+v, want one with %d candidates", body.Decision, len(poolModels))
	}
	for _, c := range body.Decision.Candidates {
		if c.Eligible || !reflect.DeepEqual(c.IneligibleReasons, []string{router.IneligibleContextWindow}) {
			t.Errorf("%s: eligible %v, reasons %v; want ineligible for context_window", c.Model, c.Eligible, c.IneligibleReasons)
		}
	}
	if n := env.prov.callCount(); n != 0 {
		t.Errorf("provider was called %d times, want 0", n)
	}
}

func TestExplainOverloadedIncludesDecision(t *testing.T) {
	env := newPoolEnv(t, func(e *poolEnv) {
		e.sig.QueueFull = func(string) bool { return true }
	})
	rec := env.post(t, "/v1/routing/explain", keyA, autoBody, nil)

	wantError(t, rec, http.StatusServiceUnavailable, CodeOverloaded)
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
	var body struct {
		Decision *router.Decision `json:"decision"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Decision == nil {
		t.Fatal("body has no decision")
	}
	for _, c := range body.Decision.Candidates {
		if !slices.Contains(c.IneligibleReasons, router.IneligibleQueueFull) {
			t.Errorf("%s: reasons %v, want queue_full", c.Model, c.IneligibleReasons)
		}
	}
}

func TestChatOverloadedIs503(t *testing.T) {
	env := newPoolEnv(t, func(e *poolEnv) {
		e.sig.QueueFull = func(string) bool { return true }
	})
	rec := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)

	wantError(t, rec, http.StatusServiceUnavailable, CodeOverloaded)
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
}

func TestDecisionsEndpoint(t *testing.T) {
	env := newPoolEnv(t) // the default decision store
	rec := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	id := rec.Header().Get("X-RouteForge-Decision-Id")
	if id == "" {
		t.Fatal("chat response has no decision id")
	}

	d := env.fetchDecision(t, keyA, id)
	if d.ID != id || d.Tenant != "tenant-a" || d.PolicyVersion != "pv-1" {
		t.Errorf("id/tenant/version = %q/%q/%q", d.ID, d.Tenant, d.PolicyVersion)
	}
	if d.Chosen != rec.Header().Get("X-RouteForge-Model") {
		t.Errorf("chosen = %q, served by %q", d.Chosen, rec.Header().Get("X-RouteForge-Model"))
	}
	if len(d.Attempts) != 1 || d.Attempts[0].Outcome != router.OutcomeOK || d.Attempts[0].Model != d.Chosen {
		t.Errorf("attempts = %+v, want one ok attempt on %s", d.Attempts, d.Chosen)
	}
	if len(d.Candidates) != len(poolModels) {
		t.Errorf("got %d candidates, want %d", len(d.Candidates), len(poolModels))
	}

	// Another tenant's key and an unknown id fail with the same body, so the
	// endpoint does not say whether an id exists.
	other := env.get(t, "/v1/routing/decisions/"+id, keyB)
	unknown := env.get(t, "/v1/routing/decisions/dec_unknown", keyA)
	wantError(t, other, http.StatusNotFound, CodeNotFound)
	wantError(t, unknown, http.StatusNotFound, CodeNotFound)
	if other.Body.String() != unknown.Body.String() {
		t.Errorf("other tenant body %q differs from unknown id body %q", other.Body, unknown.Body)
	}

	wantError(t, env.get(t, "/v1/routing/decisions/"+id, ""), http.StatusUnauthorized, CodeInvalidAPIKey)
}

func TestFailedChatStoresItsDecision(t *testing.T) {
	boom := &provider.Error{Kind: provider.KindUnavailable, Message: "boom"}
	cases := []struct {
		name string
		body string
		prov func(*stubProvider)
	}{
		{"non-streaming", autoBody, func(p *stubProvider) {
			p.complete = func(context.Context, provider.Request) (*provider.Response, error) { return nil, boom }
		}},
		{"streaming before the first byte", autoStreamBody, func(p *stubProvider) {
			p.stream = func(context.Context, provider.Request) (provider.Stream, error) { return nil, boom }
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := newPoolEnv(t, func(e *poolEnv) { tc.prov(e.prov) })
			rec := env.post(t, "/v1/chat/completions", keyA, tc.body, nil)
			wantError(t, rec, http.StatusBadGateway, CodeUpstreamError)

			id := rec.Header().Get("X-RouteForge-Decision-Id")
			d := env.fetchDecision(t, keyA, id)
			if len(d.Attempts) != 1 || d.Attempts[0].Outcome != router.OutcomeError || d.Attempts[0].ErrorKind != string(provider.KindUnavailable) {
				t.Errorf("attempts = %+v, want one error attempt of kind unavailable", d.Attempts)
			}
		})
	}
}

func TestChatDecisionStoredBeforeResponse(t *testing.T) {
	rec := httptest.NewRecorder()
	sent := -1
	spy := &spyStore{DecisionStore: store.NewRing(10)}
	spy.onPut = func(*router.Decision) { sent = rec.Body.Len() }
	env := newPoolEnv(t, withDecisions(spy))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(autoBody))
	req.Header.Set("Authorization", "Bearer "+keyA)
	env.handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if sent != 0 {
		t.Errorf("%d response bytes were written when the decision was stored, want 0: a client with the response must find its decision", sent)
	}
}

func TestStreamDecisionStoredAfterClose(t *testing.T) {
	cases := []struct {
		name        string
		stream      *stubStream
		wantOutcome string
		notYetSent  string // part of the response the client must not have seen when the decision is stored
	}{
		{"completed", &stubStream{events: []provider.StreamEvent{{Delta: "a"}, {FinishReason: "stop"}}}, router.OutcomeOK, "[DONE]"},
		{"interrupted", &stubStream{events: []provider.StreamEvent{{Delta: "a"}}, err: errors.New("boom")}, router.OutcomePartial, CodeUpstreamStreamInterrupted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var (
				puts     int
				closed   bool
				attempts []router.Attempt
				sent     string
			)
			rec := httptest.NewRecorder()
			spy := &spyStore{DecisionStore: store.NewRing(10)}
			spy.onPut = func(d *router.Decision) {
				puts++
				closed = tc.stream.closed.Load()
				attempts = slices.Clone(d.Attempts)
				sent = rec.Body.String()
			}
			env := newPoolEnv(t, withDecisions(spy), func(e *poolEnv) {
				e.prov.stream = func(ctx context.Context, _ provider.Request) (provider.Stream, error) {
					tc.stream.ctx = ctx
					return tc.stream, nil
				}
			})

			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(autoStreamBody))
			req.Header.Set("Authorization", "Bearer "+keyA)
			env.handler().ServeHTTP(rec, req)

			if puts != 1 {
				t.Fatalf("decision stored %d times, want 1", puts)
			}
			if !closed {
				t.Error("decision stored before the provider stream was closed")
			}
			if len(attempts) != 1 || attempts[0].Outcome != tc.wantOutcome {
				t.Errorf("attempts at Put = %+v, want one %s attempt", attempts, tc.wantOutcome)
			}
			// A client that reads the end of the stream may fetch the decision at
			// once, so it has to be stored before the last frame goes out.
			if strings.Contains(sent, tc.notYetSent) {
				t.Errorf("response already carried %q when the decision was stored: %q", tc.notYetSent, sent)
			}
			if !strings.Contains(rec.Body.String(), tc.notYetSent) {
				t.Errorf("response never carried %q: %q", tc.notYetSent, rec.Body)
			}

			d := env.fetchDecision(t, keyA, rec.Header().Get("X-RouteForge-Decision-Id"))
			if len(d.Attempts) != 1 || d.Attempts[0].Outcome != tc.wantOutcome {
				t.Errorf("attempts = %+v, want one %s attempt", d.Attempts, tc.wantOutcome)
			}
		})
	}
}

func TestPolicyVersionHeader(t *testing.T) {
	t.Run("active policy", func(t *testing.T) {
		env := newPoolEnv(t)
		rec := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)
		if got := rec.Header().Get("X-RouteForge-Policy-Version"); got != "pv-1" {
			t.Errorf("X-RouteForge-Policy-Version = %q, want pv-1", got)
		}
	})
	t.Run("no active policy", func(t *testing.T) {
		env := newPoolEnv(t, func(e *poolEnv) { e.reg = policy.NewRegistry(poolModels, nil) })
		rec := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
		if got := rec.Header().Get("X-RouteForge-Policy-Version"); got != "none" {
			t.Errorf("X-RouteForge-Policy-Version = %q, want none", got)
		}
	})
}

func TestHotSwapKeepsInFlightDecision(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int64
	env := newPoolEnv(t, func(e *poolEnv) {
		e.prov.complete = func(context.Context, provider.Request) (*provider.Response, error) {
			if calls.Add(1) == 1 { // only the first request waits
				close(started)
				<-release
			}
			return &provider.Response{Content: "ok", FinishReason: "stop"}, nil
		}
	})
	if err := env.reg.Add(poolPolicy("pv-2")); err != nil {
		t.Fatal(err)
	}

	first := make(chan *httptest.ResponseRecorder, 1)
	go func() { first <- env.post(t, "/v1/chat/completions", keyA, autoBody, nil) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("first request never reached the provider")
	}

	// The first request has been routed under pv-1 and is waiting upstream.
	if _, err := env.reg.Activate("pv-2", "test"); err != nil {
		t.Fatal(err)
	}
	close(release)

	var firstRec *httptest.ResponseRecorder
	select {
	case firstRec = <-first:
	case <-time.After(5 * time.Second):
		t.Fatal("first request did not finish")
	}
	if firstRec.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200; body %s", firstRec.Code, firstRec.Body)
	}
	if got := firstRec.Header().Get("X-RouteForge-Policy-Version"); got != "pv-1" {
		t.Errorf("in-flight request header = %q, want pv-1", got)
	}
	if d := env.fetchDecision(t, keyA, firstRec.Header().Get("X-RouteForge-Decision-Id")); d.PolicyVersion != "pv-1" {
		t.Errorf("in-flight decision policy = %q, want pv-1", d.PolicyVersion)
	}

	second := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d, want 200; body %s", second.Code, second.Body)
	}
	if d := env.fetchDecision(t, keyA, second.Header().Get("X-RouteForge-Decision-Id")); d.PolicyVersion != "pv-2" {
		t.Errorf("next decision policy = %q, want pv-2", d.PolicyVersion)
	}
}

func TestChatPassesFeatureTextToRouter(t *testing.T) {
	var capture *captureRouter
	prov := &stubProvider{complete: replyWith("ok")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": prov}, func(o *Options) {
		capture = &captureRouter{inner: o.Router}
		o.Router = capture
	})
	body := `{"model":"m-strong","messages":[{"role":"system","content":"be brief"},{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":"there"}]}]}`
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}

	got := capture.request()
	want := policy.FeatureText([]provider.Message{
		{Role: provider.RoleSystem, Content: "be brief"},
		{Role: provider.RoleUser, Content: "hello\nthere"},
	})
	if want == "" {
		t.Fatal("test text is empty")
	}
	if got.Text != want || got.PromptBytes != len(want) {
		t.Errorf("Text = %q (%d bytes), want %q (%d bytes)", got.Text, got.PromptBytes, want, len(want))
	}
}

// failingStore is a decision store whose operations fail.
type failingStore struct{ putErr, getErr error }

func (f failingStore) Put(*router.Decision) error { return f.putErr }
func (f failingStore) Get(context.Context, string, string) (*router.Decision, bool, error) {
	return nil, false, f.getErr
}

func TestDecisionStoreFailures(t *testing.T) {
	var logs bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&logs, nil))
	env := newPoolEnv(t, withDecisions(failingStore{putErr: errors.New("put down"), getErr: errors.New("get down")}),
		func(e *poolEnv) { e.opts = append(e.opts, func(o *Options) { o.Logger = logger }) })

	// A failed Put cannot change a response that is already decided: the
	// request succeeds and the failure is logged without the decision body.
	rec := env.post(t, "/v1/chat/completions", keyA, autoBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	id := rec.Header().Get("X-RouteForge-Decision-Id")
	for _, want := range []string{`"store decision failed"`, `"put down"`, id} {
		if !strings.Contains(logs.String(), want) {
			t.Errorf("log %q does not contain %s", logs.String(), want)
		}
	}
	if strings.Contains(logs.String(), "hello world") {
		t.Errorf("log carries prompt text: %q", logs.String())
	}

	// A failed Get is a server error, not a 404.
	rec = env.get(t, "/v1/routing/decisions/"+id, keyA)
	msg := wantError(t, rec, http.StatusInternalServerError, CodeInternalError)
	if strings.Contains(msg, "get down") {
		t.Errorf("error message %q leaks the store error", msg)
	}
	if !strings.Contains(logs.String(), `"get down"`) {
		t.Errorf("log %q does not contain the Get failure", logs.String())
	}
}
