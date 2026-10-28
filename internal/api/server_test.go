package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

func TestModelsList(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	h := srv.Handler()

	ids := func(key string) []string {
		t.Helper()
		rec := do(t, h, http.MethodGet, "/v1/models", key, "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
		var got struct {
			Object string `json:"object"`
			Data   []struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				OwnedBy string `json:"owned_by"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("body is not JSON: %v", err)
		}
		if got.Object != "list" {
			t.Errorf("object = %q, want list", got.Object)
		}
		out := make([]string, len(got.Data))
		for i, m := range got.Data {
			out[i] = m.ID
			if m.Object != "model" || m.Created != 0 || m.OwnedBy != "routeforge" {
				t.Errorf("entry %d = %+v, want object model, created 0, owned_by routeforge", i, m)
			}
		}
		return out
	}

	if got, want := ids(keyA), []string{"routeforge-auto", "m-fast", "m-strong"}; !reflect.DeepEqual(got, want) {
		t.Errorf("tenant-a models = %v, want %v", got, want)
	}
	if got, want := ids(keyB), []string{"routeforge-auto", "m-fast"}; !reflect.DeepEqual(got, want) {
		t.Errorf("tenant-b models = %v, want %v", got, want)
	}

	rec := do(t, h, http.MethodGet, "/v1/models", "", "", nil)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no key: status = %d, want 401", rec.Code)
	}
	if code, _ := errorFields(t, rec); code != CodeInvalidAPIKey {
		t.Errorf("no key: code = %q, want %q", code, CodeInvalidAPIKey)
	}
}

func TestHealthz(t *testing.T) {
	rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/healthz", "", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got := strings.TrimSpace(rec.Body.String()); got != `{"status":"ok"}` {
		t.Errorf("body = %q, want {\"status\":\"ok\"}", got)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestReadyz(t *testing.T) {
	assertJSON := func(t *testing.T, rec *httptest.ResponseRecorder, wantCode int, want string) {
		t.Helper()
		body := rec.Body.Bytes()
		if rec.Code != wantCode {
			t.Fatalf("status = %d, want %d; body %s", rec.Code, wantCode, body)
		}
		var got, exp any
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatalf("body is not JSON: %v", err)
		}
		if err := json.Unmarshal([]byte(want), &exp); err != nil {
			t.Fatalf("bad expectation: %v", err)
		}
		if !reflect.DeepEqual(got, exp) {
			t.Errorf("body = %s, want %s", body, want)
		}
	}

	t.Run("no checks", func(t *testing.T) {
		rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusOK, `{"status":"ready"}`)
	})

	t.Run("failing check", func(t *testing.T) {
		srv := newServer(t, testConfig(), nil, func(o *Options) {
			o.ReadyChecks = []ReadyCheck{{
				Name:  "policy",
				Check: func(context.Context) error { return errors.New("not loaded") },
			}}
		})
		rec := do(t, srv.Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusServiceUnavailable, `{"status":"not_ready","failed":{"policy":"not loaded"}}`)
	})

	t.Run("draining", func(t *testing.T) {
		srv := newServer(t, testConfig(), nil)
		srv.SetDraining()
		rec := do(t, srv.Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusServiceUnavailable, `{"status":"draining"}`)
	})
}

func TestNewHTTPServerLimits(t *testing.T) {
	c := testConfig().Server
	hs := NewHTTPServer(c, http.NewServeMux())
	if hs.Addr != ":8080" {
		t.Errorf("Addr = %q, want :8080", hs.Addr)
	}
	if hs.ReadHeaderTimeout != 5*time.Second {
		t.Errorf("ReadHeaderTimeout = %v, want 5s", hs.ReadHeaderTimeout)
	}
	if hs.IdleTimeout != 120*time.Second {
		t.Errorf("IdleTimeout = %v, want 120s", hs.IdleTimeout)
	}
	if hs.MaxHeaderBytes != 1048576 {
		t.Errorf("MaxHeaderBytes = %d, want 1048576", hs.MaxHeaderBytes)
	}
	if hs.ReadTimeout != 0 || hs.WriteTimeout != 0 {
		t.Errorf("ReadTimeout = %v, WriteTimeout = %v, want both 0", hs.ReadTimeout, hs.WriteTimeout)
	}
}

// accessLog is a goroutine-safe log sink: the handler of a request whose client
// left logs after the client call has returned.
type accessLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *accessLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

// requestLines returns the "request" lines logged so far.
func (l *accessLog) requestLines(t *testing.T) []map[string]any {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	var lines []map[string]any
	for _, raw := range strings.Split(strings.TrimSpace(l.buf.String()), "\n") {
		if raw == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Fatalf("log line is not JSON: %v; line %q", err, raw)
		}
		if m["msg"] == "request" {
			lines = append(lines, m)
		}
	}
	return lines
}

// requestLine returns the only "request" line, waiting up to a second for a
// handler that is still finishing.
func (l *accessLog) requestLine(t *testing.T) map[string]any {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		lines := l.requestLines(t)
		if len(lines) == 1 {
			return lines[0]
		}
		if len(lines) > 1 || time.Now().After(deadline) {
			t.Fatalf("got %d request log lines, want 1: %v", len(lines), lines)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// cancelAtEndStream delivers one event, then cancels the request context and
// ends cleanly, as when the client leaves just as the last event goes out.
type cancelAtEndStream struct {
	cancel context.CancelFunc
	sent   bool
}

func (s *cancelAtEndStream) Recv() (provider.StreamEvent, error) {
	if !s.sent {
		s.sent = true
		return provider.StreamEvent{Delta: "only"}, nil
	}
	s.cancel()
	return provider.StreamEvent{}, io.EOF
}

func (s *cancelAtEndStream) Close() error { return nil }

func TestAccessLogClientGone499(t *testing.T) {
	// serve runs one request through a server that logs into a fresh sink and
	// returns its access log line. build receives the cancel function of the
	// request context, so a stub can end it the way a departing client does.
	serve := func(t *testing.T, body string, hdr map[string]string, build func(cancel context.CancelFunc) *stubProvider) map[string]any {
		t.Helper()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var sink accessLog
		srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": build(cancel)}, func(o *Options) {
			o.Logger = slog.New(slog.NewJSONHandler(&sink, nil))
		})
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer "+keyA)
		req.Header.Set("Content-Type", "application/json")
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		srv.Handler().ServeHTTP(httptest.NewRecorder(), req)
		return sink.requestLine(t)
	}
	want := func(t *testing.T, line map[string]any, status float64, gone bool) {
		t.Helper()
		if line["status"] != status || line["client_gone"] != gone {
			t.Errorf("status = %v, client_gone = %v; want %v, %v (line %v)", line["status"], line["client_gone"], status, gone, line)
		}
	}

	tests := []struct {
		name       string
		body       string
		hdr        map[string]string
		build      func(cancel context.CancelFunc) *stubProvider
		wantStatus float64
		wantGone   bool
	}{
		{
			name: "client leaves during a non-streaming call",
			body: okBody,
			build: func(cancel context.CancelFunc) *stubProvider {
				return &stubProvider{complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
					cancel()
					<-ctx.Done()
					return nil, ctx.Err()
				}}
			},
			wantStatus: 499,
			wantGone:   true,
		},
		{
			name: "client leaves before the first stream event",
			body: streamBody,
			build: func(cancel context.CancelFunc) *stubProvider {
				return &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
					cancel()
					return nil, context.Canceled
				}}
			},
			wantStatus: 499,
			wantGone:   true,
		},
		{
			name: "client leaves mid-stream",
			body: streamBody,
			build: func(cancel context.CancelFunc) *stubProvider {
				return &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
					return &cancelingStream{cancel: cancel}, nil
				}}
			},
			wantStatus: 499,
			wantGone:   true,
		},
		{
			name: "client leaves after a complete JSON response",
			body: okBody,
			build: func(cancel context.CancelFunc) *stubProvider {
				return &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
					cancel()
					return &provider.Response{Content: "done", FinishReason: "stop"}, nil
				}}
			},
			wantStatus: 200,
		},
		{
			name: "client leaves as the stream ends",
			body: streamBody,
			build: func(cancel context.CancelFunc) *stubProvider {
				return &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
					return &cancelAtEndStream{cancel: cancel}, nil
				}}
			},
			wantStatus: 200,
		},
		{
			name: "ordinary success",
			body: okBody,
			build: func(context.CancelFunc) *stubProvider {
				return &stubProvider{complete: replyWith("hello")}
			},
			wantStatus: 200,
		},
		{
			name: "gateway deadline during a non-streaming call",
			body: okBody,
			hdr:  map[string]string{"X-RouteForge-Deadline-Ms": "20"},
			build: func(context.CancelFunc) *stubProvider {
				return &stubProvider{complete: blockUntilDone}
			},
			wantStatus: 504,
		},
		{
			name: "gateway deadline mid-stream",
			body: streamBody,
			hdr:  map[string]string{"X-RouteForge-Deadline-Ms": "50"},
			build: func(context.CancelFunc) *stubProvider {
				return streamingProvider(&stubStream{events: []provider.StreamEvent{{Delta: "a"}}, block: true})
			},
			wantStatus: 200, // the headers were out; the deadline arrives as an error event
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			want(t, serve(t, tc.body, tc.hdr, tc.build), tc.wantStatus, tc.wantGone)
		})
	}

	// The cases above end the request context by hand. These let net/http do
	// it, as it does when a client closes its connection.
	open := func(t *testing.T, p *stubProvider) (url string, sink *accessLog) {
		t.Helper()
		sink = &accessLog{}
		srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": p}, func(o *Options) {
			o.Logger = slog.New(slog.NewJSONHandler(sink, nil))
		})
		ts := httptest.NewServer(srv.Handler())
		t.Cleanup(ts.Close)
		return ts.URL, sink
	}
	post := func(t *testing.T, ctx context.Context, url, body string) *http.Response {
		t.Helper()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+keyA)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	t.Run("a closed connection mid-stream", func(t *testing.T) {
		st := &stubStream{events: []provider.StreamEvent{{Delta: "first"}}, block: true}
		url, sink := open(t, streamingProvider(st))
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		resp := post(t, ctx, url, streamBody)
		defer resp.Body.Close()
		if line, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil || !strings.Contains(line, `"first"`) {
			t.Fatalf("first line = %q, %v; want the first content chunk", line, err)
		}
		cancel()
		want(t, sink.requestLine(t), 499, true)
	})

	t.Run("a closed connection during a non-streaming call", func(t *testing.T) {
		entered := make(chan struct{})
		url, sink := open(t, &stubProvider{complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
			close(entered)
			<-ctx.Done()
			return nil, ctx.Err()
		}})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			<-entered
			cancel()
		}()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", strings.NewReader(okBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+keyA)
		req.Header.Set("Content-Type", "application/json")
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
			t.Fatalf("status = %d, want the call to fail with the canceled context", resp.StatusCode)
		}
		want(t, sink.requestLine(t), 499, true)
	})
}

func TestMethodNotAllowed(t *testing.T) {
	rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/v1/chat/completions", keyA, "", nil)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}
