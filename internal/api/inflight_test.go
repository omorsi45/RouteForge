package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
)

// gate is a provider that holds its first call until released, so a test can
// keep a request in flight for as long as it needs.
type gate struct {
	entered chan struct{} // closed when a call is inside the provider
	release chan struct{} // close to let held calls finish
	once    sync.Once
}

func newGate() *gate {
	return &gate{entered: make(chan struct{}), release: make(chan struct{})}
}

func (g *gate) provider() *stubProvider {
	return &stubProvider{complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
		g.once.Do(func() { close(g.entered) })
		select {
		case <-g.release:
			return &provider.Response{Content: "done", FinishReason: "stop"}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}}
}

// waitEntered blocks until a request is inside the provider.
func (g *gate) waitEntered(t *testing.T) {
	t.Helper()
	select {
	case <-g.entered:
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("no request reached the provider")
	}
}

// holdOne starts a chat request that blocks in the provider and returns the
// channel its response arrives on once the gate opens.
func holdOne(t *testing.T, h http.Handler, g *gate) <-chan *httptest.ResponseRecorder {
	t.Helper()
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil) }()
	g.waitEntered(t)
	return done
}

func wantOverloaded(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	wantError(t, rec, http.StatusServiceUnavailable, CodeOverloaded)
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
}

func TestChatOverloadedWhenInFlightFull(t *testing.T) {
	g := newGate()
	strong := g.provider()
	inflight := reliability.NewInFlight(1)
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong}, func(o *Options) { o.InFlight = inflight })
	h := srv.Handler()

	first := holdOne(t, h, g)
	if inflight.Count() != 1 {
		t.Fatalf("Count = %d with one request in flight, want 1", inflight.Count())
	}

	rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)
	wantOverloaded(t, rec)
	if got := strong.callCount(); got != 1 {
		t.Errorf("provider saw %d calls, want 1: the refused request must not reach it", got)
	}

	close(g.release)
	if rec := <-first; rec.Code != http.StatusOK {
		t.Fatalf("held request status = %d, want 200; body %s", rec.Code, rec.Body)
	}

	// The slot came back, so the next request is served.
	if rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusOK {
		t.Fatalf("request after the slot was freed: status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if inflight.Count() != 0 {
		t.Fatalf("Count = %d after every request finished, want 0", inflight.Count())
	}
}

func TestExplainOverloadedWhenInFlightFull(t *testing.T) {
	g := newGate()
	inflight := reliability.NewInFlight(1)
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": g.provider()}, func(o *Options) { o.InFlight = inflight })
	h := srv.Handler()

	first := holdOne(t, h, g)

	// Explain shares the limit with chat: it runs the router and parses the
	// body just the same.
	rec := do(t, h, http.MethodPost, "/v1/routing/explain", keyA, okBody, nil)
	wantOverloaded(t, rec)

	close(g.release)
	<-first
	if rec := do(t, h, http.MethodPost, "/v1/routing/explain", keyA, okBody, nil); rec.Code != http.StatusOK {
		t.Fatalf("explain after the slot was freed: status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if inflight.Count() != 0 {
		t.Fatalf("Count = %d after every request finished, want 0", inflight.Count())
	}
}

func TestInFlightSlotFreedOnEveryPath(t *testing.T) {
	inflight := reliability.NewInFlight(1)
	ok := &stubProvider{complete: replyWith("hi")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": ok}, func(o *Options) { o.InFlight = inflight })
	h := srv.Handler()

	failing := []struct {
		name string
		path string
		key  string
		body string
		hdr  map[string]string
		code int
	}{
		{"bad key", "/v1/chat/completions", "wrong", okBody, nil, http.StatusUnauthorized},
		{"bad profile", "/v1/chat/completions", keyA, okBody, map[string]string{"X-RouteForge-Profile": "fast"}, http.StatusBadRequest},
		{"malformed body", "/v1/chat/completions", keyA, `{"model":`, nil, http.StatusBadRequest},
		{"unknown model", "/v1/chat/completions", keyA, `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`, nil, http.StatusBadRequest},
		{"explain, bad key", "/v1/routing/explain", "wrong", okBody, nil, http.StatusUnauthorized},
		{"explain, unknown model", "/v1/routing/explain", keyA, `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`, nil, http.StatusBadRequest},
	}
	for _, tc := range failing {
		t.Run(tc.name, func(t *testing.T) {
			if rec := do(t, h, http.MethodPost, tc.path, tc.key, tc.body, tc.hdr); rec.Code != tc.code {
				t.Fatalf("status = %d, want %d; body %s", rec.Code, tc.code, rec.Body)
			}
			if inflight.Count() != 0 {
				t.Fatalf("Count = %d after a failed request, want 0", inflight.Count())
			}
		})
	}

	// With a limit of one, any leaked slot would turn this into a 503.
	if rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusOK {
		t.Fatalf("request after the failures: status = %d, want 200; body %s", rec.Code, rec.Body)
	}
}

func TestChatWithoutInFlightLimit(t *testing.T) {
	strong := &stubProvider{complete: replyWith("hi")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong}) // Options.InFlight is nil
	h := srv.Handler()
	for range 3 {
		if rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyA, okBody, nil); rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
	}
}

// routerFunc adapts a function to router.Router.
type routerFunc func(context.Context, *router.Request) (*router.Decision, error)

func (f routerFunc) Decide(ctx context.Context, req *router.Request) (*router.Decision, error) {
	return f(ctx, req)
}

func TestRouterOverloadedMapsTo503(t *testing.T) {
	strong := &stubProvider{complete: replyWith("hi")}
	overloaded := routerFunc(func(context.Context, *router.Request) (*router.Decision, error) {
		return nil, fmt.Errorf("%w: m-strong (queue_full)", router.ErrOverloaded)
	})
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong}, func(o *Options) { o.Router = overloaded })

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)
	msg := wantError(t, rec, http.StatusServiceUnavailable, CodeOverloaded)
	if msg == "" {
		t.Error("the error has no message")
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
	if got := strong.callCount(); got != 0 {
		t.Errorf("provider saw %d calls, want 0", got)
	}
}
