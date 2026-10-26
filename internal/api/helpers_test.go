package api

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

const (
	keyA = "test-key-tenant-a"
	keyB = "test-key-tenant-b"
)

// testConfig returns a validated two-tenant, two-model configuration.
func testConfig() *config.Config {
	c := &config.Config{
		Tenants: []config.Tenant{
			{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey(keyA)}},
			{
				Name:                  "tenant-b",
				APIKeySHA256:          []string{auth.HashKey(keyB)},
				AllowStrategyOverride: true,
				AllowedModels:         []string{"m-fast"},
				MaxOutputTokens:       100,
				MaxDeadline:           200 * time.Millisecond,
			},
		},
		Providers: []config.Provider{
			{Name: "p-fast", Type: "fake"},
			{Name: "p-strong", Type: "fake"},
		},
		Models: []config.Model{
			{Name: "m-fast", Provider: "p-fast", UpstreamModel: "up-fast"},
			{Name: "m-strong", Provider: "p-strong", UpstreamModel: "up-strong", Strongest: true},
		},
	}
	c.ApplyDefaults()
	if err := c.Validate(); err != nil {
		panic("api test config is invalid: " + err.Error())
	}
	return c
}

// newServer builds a Server over cfg. providers is keyed by provider name.
// opts run last and may replace any Options field.
func newServer(t *testing.T, cfg *config.Config, providers map[string]provider.Provider, opts ...func(*Options)) *Server {
	t.Helper()
	targets := make(map[string]executor.Target, len(cfg.Models))
	for _, m := range cfg.Models {
		p, ok := providers[m.Provider]
		if !ok {
			continue
		}
		targets[m.Name] = executor.Target{Model: m.Name, Upstream: m.UpstreamModel, Provider: p}
	}
	o := Options{
		Config:   cfg,
		Keys:     auth.NewStore(cfg.Tenants),
		Router:   router.NewBasic(cfg.Models, router.BasicOptions{}),
		Executor: executor.New(targets, nil),
	}
	for _, opt := range opts {
		opt(&o)
	}
	return New(o)
}

// stubProvider records every call and delegates to scripted functions.
type stubProvider struct {
	complete func(ctx context.Context, req provider.Request) (*provider.Response, error)
	stream   func(ctx context.Context, req provider.Request) (provider.Stream, error)

	mu    sync.Mutex
	reqs  []provider.Request
	ctxs  []context.Context
	calls int
}

func (p *stubProvider) record(ctx context.Context, req provider.Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.reqs = append(p.reqs, req)
	p.ctxs = append(p.ctxs, ctx)
}

func (p *stubProvider) Complete(ctx context.Context, req provider.Request) (*provider.Response, error) {
	p.record(ctx, req)
	if p.complete == nil {
		return nil, errors.New("stubProvider: no complete func")
	}
	return p.complete(ctx, req)
}

func (p *stubProvider) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	p.record(ctx, req)
	if p.stream == nil {
		return nil, errors.New("stubProvider: no stream func")
	}
	return p.stream(ctx, req)
}

// lastRequest returns the most recent request the provider saw.
func (p *stubProvider) lastRequest(t *testing.T) provider.Request {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.reqs) == 0 {
		t.Fatal("provider was never called")
	}
	return p.reqs[len(p.reqs)-1]
}

// lastCtx returns the context of the most recent call.
func (p *stubProvider) lastCtx(t *testing.T) context.Context {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.ctxs) == 0 {
		t.Fatal("provider was never called")
	}
	return p.ctxs[len(p.ctxs)-1]
}

func (p *stubProvider) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// stubStream replays scripted events, then ends with err (io.EOF when nil). With
// block set it instead waits for ctx to end or the stream to be closed.
type stubStream struct {
	events []provider.StreamEvent
	err    error
	block  bool
	ctx    context.Context

	closed    atomic.Bool
	closeOnce sync.Once
	done      chan struct{}
	doneOnce  sync.Once
	i         int
}

func (s *stubStream) doneCh() chan struct{} {
	s.doneOnce.Do(func() { s.done = make(chan struct{}) })
	return s.done
}

func (s *stubStream) Recv() (provider.StreamEvent, error) {
	if s.i < len(s.events) {
		ev := s.events[s.i]
		s.i++
		return ev, nil
	}
	if s.block {
		ctx := s.ctx
		if ctx == nil {
			ctx = context.Background()
		}
		select {
		case <-ctx.Done():
			return provider.StreamEvent{}, ctx.Err()
		case <-s.doneCh():
			return provider.StreamEvent{}, context.Canceled
		}
	}
	if s.err != nil {
		return provider.StreamEvent{}, s.err
	}
	return provider.StreamEvent{}, io.EOF
}

func (s *stubStream) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		close(s.doneCh())
	})
	return nil
}

// do sends one request through h and returns the recorded response. A non-empty
// key is sent as a bearer token; hdr adds or overrides headers.
func do(t *testing.T, h http.Handler, method, path, key, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rd)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestStubStream(t *testing.T) {
	t.Run("events then terminal error", func(t *testing.T) {
		boom := errors.New("boom")
		s := &stubStream{events: []provider.StreamEvent{{Delta: "a"}, {Delta: "b"}}, err: boom}
		for _, want := range []string{"a", "b"} {
			ev, err := s.Recv()
			if err != nil || ev.Delta != want {
				t.Fatalf("Recv = %+v, %v; want delta %q", ev, err, want)
			}
		}
		if _, err := s.Recv(); !errors.Is(err, boom) {
			t.Errorf("Recv after events = %v, want boom", err)
		}
	})

	t.Run("events then EOF", func(t *testing.T) {
		s := &stubStream{events: []provider.StreamEvent{{Delta: "a"}}}
		if _, err := s.Recv(); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Recv(); !errors.Is(err, io.EOF) {
			t.Errorf("Recv after events = %v, want io.EOF", err)
		}
	})

	t.Run("block ends on ctx", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		s := &stubStream{block: true, ctx: ctx}
		cancel()
		if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
			t.Errorf("Recv = %v, want context.Canceled", err)
		}
	})

	t.Run("block ends on close", func(t *testing.T) {
		s := &stubStream{block: true}
		errc := make(chan error, 1)
		go func() {
			_, err := s.Recv()
			errc <- err
		}()
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatalf("second Close = %v, want nil", err)
		}
		select {
		case err := <-errc:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("Recv = %v, want context.Canceled", err)
			}
		case <-time.After(time.Second):
			t.Fatal("Recv did not return after Close")
		}
		if !s.closed.Load() {
			t.Error("closed flag is not set")
		}
	})
}
