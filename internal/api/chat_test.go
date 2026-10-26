package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

const okBody = `{"model":"m-strong","messages":[{"role":"user","content":"hi"}]}`

// chatResp mirrors the non-streaming success body.
type chatResp struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	RouteForge struct {
		DecisionID string `json:"decision_id"`
		Reason     string `json:"reason"`
		Attempts   int    `json:"attempts"`
	} `json:"routeforge"`
}

// errorFields returns the code and message of an error envelope.
func errorFields(t *testing.T, rec *httptest.ResponseRecorder) (code, message string) {
	t.Helper()
	var env struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body is not an error envelope: %v; body %q", err, rec.Body)
	}
	return env.Error.Code, env.Error.Message
}

func wantError(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) string {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("status = %d, want %d; body %s", rec.Code, status, rec.Body)
	}
	got, msg := errorFields(t, rec)
	if got != code {
		t.Fatalf("code = %q, want %q; body %s", got, code, rec.Body)
	}
	return msg
}

func replyWith(content string) func(context.Context, provider.Request) (*provider.Response, error) {
	return func(context.Context, provider.Request) (*provider.Response, error) {
		return &provider.Response{Content: content, FinishReason: "stop"}, nil
	}
}

func blockUntilDone(ctx context.Context, _ provider.Request) (*provider.Response, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

// captureRouter records the router.Request it is asked to decide.
type captureRouter struct {
	inner router.Router
	mu    sync.Mutex
	last  router.Request
}

func (c *captureRouter) Decide(ctx context.Context, req *router.Request) (*router.Decision, error) {
	c.mu.Lock()
	c.last = *req
	c.mu.Unlock()
	return c.inner.Decide(ctx, req)
}

func (c *captureRouter) request() router.Request {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.last
}

func TestChatNonStreaming(t *testing.T) {
	strong := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		return &provider.Response{Content: "hello", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: 2}}, nil
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	h := rec.Header()
	for name, want := range map[string]string{
		"X-RouteForge-Model":          "^m-strong$",
		"X-RouteForge-Decision-Id":    "^dec_[0-9a-f]{32}$",
		"X-RouteForge-Policy-Version": "^none$",
		"X-RouteForge-Reason":         "^pinned$",
		"X-Request-Id":                "^req_[0-9a-f]{24}$",
	} {
		if got := h.Get(name); !regexp.MustCompile(want).MatchString(got) {
			t.Errorf("%s = %q, want match %s", name, got, want)
		}
	}
	if ct := h.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var got chatResp
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	if !regexp.MustCompile(`^chatcmpl-[0-9a-f]{24}$`).MatchString(got.ID) {
		t.Errorf("id = %q, want chatcmpl-<24 hex>", got.ID)
	}
	if got.Object != "chat.completion" || got.Model != "m-strong" {
		t.Errorf("object, model = %q, %q; want chat.completion, m-strong", got.Object, got.Model)
	}
	if got.Created <= 0 {
		t.Errorf("created = %d, want unix seconds", got.Created)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(got.Choices))
	}
	c := got.Choices[0]
	if c.Index != 0 || c.Message.Role != "assistant" || c.Message.Content != "hello" || c.FinishReason != "stop" {
		t.Errorf("choice = %+v, want assistant hello stop", c)
	}
	if got.Usage == nil || got.Usage.PromptTokens != 5 || got.Usage.CompletionTokens != 2 || got.Usage.TotalTokens != 7 {
		t.Errorf("usage = %+v, want 5/2/7", got.Usage)
	}
	if got.RouteForge.DecisionID != h.Get("X-RouteForge-Decision-Id") {
		t.Errorf("routeforge.decision_id = %q, header = %q", got.RouteForge.DecisionID, h.Get("X-RouteForge-Decision-Id"))
	}
	if got.RouteForge.Reason != "pinned" || got.RouteForge.Attempts != 1 {
		t.Errorf("routeforge = %+v, want reason pinned, attempts 1", got.RouteForge)
	}

	req := strong.lastRequest(t)
	if req.Model != "up-strong" || req.MaxTokens != 4096 {
		t.Errorf("provider saw model %q max_tokens %d, want up-strong 4096", req.Model, req.MaxTokens)
	}
}

func TestChatOmitsUsageWhenUnreported(t *testing.T) {
	strong := &stubProvider{complete: replyWith("hello")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	if _, ok := raw["usage"]; ok {
		t.Errorf("body has a usage key, want none: %s", rec.Body)
	}
}

func TestChatAutoRoutesToStrongest(t *testing.T) {
	strong := &stubProvider{complete: replyWith("hello")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	body := `{"model":"routeforge-auto","messages":[{"role":"user","content":"hi"}]}`
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	if got := rec.Header().Get("X-RouteForge-Model"); got != "m-strong" {
		t.Errorf("X-RouteForge-Model = %q, want m-strong", got)
	}
	if got := rec.Header().Get("X-RouteForge-Reason"); got != "strongest" {
		t.Errorf("X-RouteForge-Reason = %q, want strongest", got)
	}
}

func TestChatUnauthorized(t *testing.T) {
	strong := &stubProvider{complete: replyWith("hello")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	for name, hdr := range map[string]map[string]string{
		"missing":      nil,
		"wrong bearer": {"Authorization": "Bearer wrong"},
		"basic scheme": {"Authorization": "Basic x"},
	} {
		t.Run(name, func(t *testing.T) {
			rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", "", okBody, hdr)
			msg := wantError(t, rec, http.StatusUnauthorized, CodeInvalidAPIKey)
			if msg != "invalid API key" {
				t.Errorf("message = %q, want %q", msg, "invalid API key")
			}
			if strings.Contains(rec.Body.String(), "tenant-a") {
				t.Errorf("body leaks tenant detail: %s", rec.Body)
			}
		})
	}
	if n := strong.callCount(); n != 0 {
		t.Errorf("provider called %d times, want 0", n)
	}
}

func TestChatBodyTooLarge(t *testing.T) {
	big := `{"model":"m-strong","messages":[{"role":"user","content":"` + strings.Repeat("x", 1024) + `"}]}`
	newSrv := func() (*Server, *stubProvider) {
		cfg := testConfig()
		cfg.Server.MaxBodyBytes = 64
		strong := &stubProvider{complete: replyWith("hello")}
		return newServer(t, cfg, map[string]provider.Provider{"p-strong": strong}), strong
	}

	t.Run("with content length", func(t *testing.T) {
		srv, strong := newSrv()
		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, big, nil)
		wantError(t, rec, http.StatusRequestEntityTooLarge, CodeRequestTooLarge)
		if n := strong.callCount(); n != 0 {
			t.Errorf("provider called %d times, want 0", n)
		}
	})

	t.Run("chunked", func(t *testing.T) {
		srv, strong := newSrv()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", io.NopCloser(strings.NewReader(big)))
		req.ContentLength = -1
		req.Header.Set("Authorization", "Bearer "+keyA)
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		wantError(t, rec, http.StatusRequestEntityTooLarge, CodeRequestTooLarge)
		if n := strong.callCount(); n != 0 {
			t.Errorf("provider called %d times, want 0", n)
		}
	})
}

func TestChatInvalidProfile(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, map[string]string{"X-RouteForge-Profile": "fast"})
	wantError(t, rec, http.StatusBadRequest, CodeInvalidRequest)
}

func TestChatProfilePassed(t *testing.T) {
	cfg := testConfig()
	capture := &captureRouter{inner: router.NewBasic(cfg.Models, router.BasicOptions{})}
	strong := &stubProvider{complete: replyWith("hello")}
	srv := newServer(t, cfg, map[string]provider.Provider{"p-strong": strong}, func(o *Options) { o.Router = capture })

	for _, tc := range []struct {
		header, want string
	}{
		{"", "balanced"},
		{"strict", "strict"},
	} {
		hdr := map[string]string{}
		if tc.header != "" {
			hdr["X-RouteForge-Profile"] = tc.header
		}
		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, hdr)
		if rec.Code != http.StatusOK {
			t.Fatalf("header %q: status = %d, want 200; body %s", tc.header, rec.Code, rec.Body)
		}
		if got := capture.request().Profile; got != tc.want {
			t.Errorf("header %q: router saw profile %q, want %q", tc.header, got, tc.want)
		}
	}
}

func TestStrategyOverride(t *testing.T) {
	srv := newServer(t, testConfig(), map[string]provider.Provider{
		"p-fast":   &stubProvider{complete: replyWith("hello")},
		"p-strong": &stubProvider{complete: replyWith("hello")},
	})
	h := srv.Handler()
	auto := `{"model":"routeforge-auto","messages":[{"role":"user","content":"hi"}]}`
	post := func(key, strategy string) *httptest.ResponseRecorder {
		return do(t, h, http.MethodPost, "/v1/chat/completions", key, auto, map[string]string{"X-RouteForge-Strategy": strategy})
	}

	t.Run("tenant without override", func(t *testing.T) {
		wantError(t, post(keyA, "strongest"), http.StatusForbidden, CodeStrategyOverrideForbidden)
	})
	t.Run("no eligible model", func(t *testing.T) {
		rec := post(keyB, "strongest")
		wantError(t, rec, http.StatusServiceUnavailable, CodeNoEligibleModel)
		if got := rec.Header().Get("Retry-After"); got != "1" {
			t.Errorf("Retry-After = %q, want 1", got)
		}
	})
	t.Run("strategy not available", func(t *testing.T) {
		msg := wantError(t, post(keyB, "quality"), http.StatusBadRequest, CodeInvalidRequest)
		if !strings.Contains(msg, `"quality"`) {
			t.Errorf("message = %q, want it to contain %q", msg, `"quality"`)
		}
	})
	t.Run("unknown strategy", func(t *testing.T) {
		wantError(t, post(keyB, "fastest"), http.StatusBadRequest, CodeInvalidRequest)
	})
}

func TestModelAccess(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	h := srv.Handler()

	rec := do(t, h, http.MethodPost, "/v1/chat/completions", keyB, okBody, nil)
	wantError(t, rec, http.StatusForbidden, CodeModelNotAllowed)

	body := `{"model":"nope","messages":[{"role":"user","content":"hi"}]}`
	rec = do(t, h, http.MethodPost, "/v1/chat/completions", keyB, body, nil)
	wantError(t, rec, http.StatusBadRequest, CodeInvalidRequest)
}

func TestMaxTokensCapped(t *testing.T) {
	for _, tc := range []struct {
		name, extra string
		want        int
	}{
		{"above cap", `,"max_tokens":5000`, 100},
		{"absent", ``, 100},
		{"below cap", `,"max_tokens":50`, 50},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fast := &stubProvider{complete: replyWith("hello")}
			srv := newServer(t, testConfig(), map[string]provider.Provider{"p-fast": fast})
			body := `{"model":"m-fast","messages":[{"role":"user","content":"hi"}]` + tc.extra + `}`
			rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyB, body, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
			}
			if got := fast.lastRequest(t).MaxTokens; got != tc.want {
				t.Errorf("provider saw max_tokens %d, want %d", got, tc.want)
			}
		})
	}
}

func TestDeadlineHeader(t *testing.T) {
	var calledAt time.Time
	strong := &stubProvider{complete: func(ctx context.Context, req provider.Request) (*provider.Response, error) {
		calledAt = time.Now()
		return blockUntilDone(ctx, req)
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})

	before := time.Now()
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, map[string]string{"X-RouteForge-Deadline-Ms": "50"})
	if elapsed := time.Since(before); elapsed > 2*time.Second {
		t.Errorf("took %v, want under 2s", elapsed)
	}
	wantError(t, rec, http.StatusGatewayTimeout, CodeDeadlineExceeded)

	dl, ok := strong.lastCtx(t).Deadline()
	if !ok {
		t.Fatal("provider ctx has no deadline")
	}
	if ahead := dl.Sub(calledAt); ahead <= 0 || ahead > 60*time.Millisecond {
		t.Errorf("provider ctx deadline is %v ahead of the call, want within (0, 60ms]", ahead)
	}
}

func TestDeadlineCappedByTenant(t *testing.T) {
	fast := &stubProvider{complete: blockUntilDone}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-fast": fast})
	body := `{"model":"m-fast","messages":[{"role":"user","content":"hi"}]}`

	before := time.Now()
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyB, body, nil)
	if elapsed := time.Since(before); elapsed > 2*time.Second {
		t.Errorf("took %v, want under 2s", elapsed)
	}
	wantError(t, rec, http.StatusGatewayTimeout, CodeDeadlineExceeded)
}

func TestDeadlineHeaderOverflowClamps(t *testing.T) {
	var calledAt time.Time
	fast := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
		calledAt = time.Now()
		return &provider.Response{Content: "hello", FinishReason: "stop"}, nil
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-fast": fast})
	body := `{"model":"m-fast","messages":[{"role":"user","content":"hi"}]}`

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyB, body, map[string]string{"X-RouteForge-Deadline-Ms": "9223372036854775807"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	dl, ok := fast.lastCtx(t).Deadline()
	if !ok {
		t.Fatal("provider ctx has no deadline")
	}
	if ahead := dl.Sub(calledAt); ahead < 150*time.Millisecond || ahead > 200*time.Millisecond {
		t.Errorf("provider ctx deadline is %v ahead of the call, want 150ms to 200ms", ahead)
	}
}

func TestDeadlineHeaderInvalid(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	for _, v := range []string{"abc", "0", "-5", "99999999999999999999"} {
		t.Run(v, func(t *testing.T) {
			rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, map[string]string{"X-RouteForge-Deadline-Ms": v})
			wantError(t, rec, http.StatusBadRequest, CodeInvalidRequest)
		})
	}
}

// TestDeadlineOverRealServer runs against a real HTTP server because net/http
// ties the request context to the connection's read deadline; a recorder
// cannot show that race. Repeating catches a context canceled by the read
// deadline being mistaken for a departed client.
func TestDeadlineOverRealServer(t *testing.T) {
	strong := &stubProvider{complete: blockUntilDone}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	for i := range 20 {
		req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(okBody))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+keyA)
		req.Header.Set("X-RouteForge-Deadline-Ms", "100")
		resp, err := ts.Client().Do(req)
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		body, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			t.Fatalf("request %d: read body: %v", i, err)
		}
		if resp.StatusCode != http.StatusGatewayTimeout {
			t.Fatalf("request %d: status = %d, want 504; body %s", i, resp.StatusCode, body)
		}
		var env struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(body, &env); err != nil || env.Error.Code != CodeDeadlineExceeded {
			t.Fatalf("request %d: body = %s, want code %s", i, body, CodeDeadlineExceeded)
		}
	}
}

func TestClientCancelPropagates(t *testing.T) {
	started := make(chan struct{})
	observed := make(chan time.Time, 1)
	strong := &stubProvider{complete: func(ctx context.Context, _ provider.Request) (*provider.Response, error) {
		close(started)
		<-ctx.Done()
		observed <- time.Now()
		return nil, ctx.Err()
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(okBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+keyA)

	errc := make(chan error, 1)
	go func() {
		resp, err := ts.Client().Do(req)
		if resp != nil {
			_ = resp.Body.Close()
		}
		errc <- err
	}()

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("provider was never called")
	}
	time.Sleep(50 * time.Millisecond)
	canceledAt := time.Now()
	cancel()

	select {
	case at := <-observed:
		if lag := at.Sub(canceledAt); lag > time.Second {
			t.Errorf("provider saw cancellation %v after the client left, want under 1s", lag)
		}
	case <-time.After(time.Second):
		t.Fatal("provider ctx was not canceled within 1s of the client leaving")
	}
	if err := <-errc; !errors.Is(err, context.Canceled) {
		t.Errorf("client error = %v, want context.Canceled", err)
	}
}

func TestUpstreamErrorMapping(t *testing.T) {
	for _, tc := range []struct {
		name       string
		err        *provider.Error
		status     int
		code       string
		retryAfter string
	}{
		{"rate limited", &provider.Error{Kind: provider.KindRateLimited, RetryAfter: 3 * time.Second}, 429, CodeRateLimited, "3"},
		{"unavailable", &provider.Error{Kind: provider.KindUnavailable, Message: "test-key-aaaa leaked"}, 502, CodeUpstreamError, ""},
		{"auth", &provider.Error{Kind: provider.KindAuth, Message: "bad credential"}, 502, CodeUpstreamError, ""},
		{"context length", &provider.Error{Kind: provider.KindContextLength, Message: "too long"}, 400, CodeContextLengthExceeded, ""},
		{"bad request", &provider.Error{Kind: provider.KindBadRequest, Message: "nope"}, 400, CodeInvalidRequest, ""},
		{"timeout", &provider.Error{Kind: provider.KindTimeout}, 504, CodeDeadlineExceeded, ""},
		{"internal", &provider.Error{Kind: provider.KindInternal, Message: "boom"}, 502, CodeUpstreamError, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			strong := &stubProvider{complete: func(context.Context, provider.Request) (*provider.Response, error) {
				return nil, tc.err
			}}
			srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
			rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, okBody, nil)

			wantError(t, rec, tc.status, tc.code)
			if got := rec.Header().Get("Retry-After"); got != tc.retryAfter {
				t.Errorf("Retry-After = %q, want %q", got, tc.retryAfter)
			}
			if tc.status >= 500 && strings.Contains(rec.Body.String(), "test-key-aaaa") {
				t.Errorf("5xx body leaks upstream text: %s", rec.Body)
			}
			if rec.Header().Get("X-RouteForge-Decision-Id") == "" {
				t.Error("X-RouteForge-Decision-Id is missing on the error response")
			}
			if rec.Header().Get("X-Request-Id") == "" {
				t.Error("X-Request-Id is missing on the error response")
			}
		})
	}
}

func TestRequestIDEcho(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	h := srv.Handler()

	rec := do(t, h, http.MethodGet, "/healthz", "", "", map[string]string{"X-Request-Id": "abc-123"})
	if got := rec.Header().Get("X-Request-Id"); got != "abc-123" {
		t.Errorf("X-Request-Id = %q, want abc-123", got)
	}

	rec = do(t, h, http.MethodGet, "/healthz", "", "", map[string]string{"X-Request-Id": "bad id!"})
	if got := rec.Header().Get("X-Request-Id"); !regexp.MustCompile(`^req_[0-9a-f]{24}$`).MatchString(got) {
		t.Errorf("X-Request-Id = %q, want a generated req_<24 hex> id", got)
	}
}

func TestAccessLogHasNoPrompt(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, nil))
	strong := &stubProvider{complete: replyWith("secret completion text")}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong}, func(o *Options) { o.Logger = logger })

	body := `{"model":"m-strong","messages":[{"role":"user","content":"secret prompt text"}]}`
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}

	logged := buf.String()
	for _, secret := range []string{"secret prompt text", "secret completion text", keyA} {
		if strings.Contains(logged, secret) {
			t.Errorf("log contains %q: %s", secret, logged)
		}
	}
	var lines []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(logged), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("log line is not JSON: %v; line %q", err, line)
		}
		if m["msg"] == "request" {
			lines = append(lines, m)
		}
	}
	if len(lines) != 1 {
		t.Fatalf("got %d request log lines, want 1: %s", len(lines), logged)
	}
	m := lines[0]
	if m["request_id"] != rec.Header().Get("X-Request-Id") {
		t.Errorf("request_id = %v, want %q", m["request_id"], rec.Header().Get("X-Request-Id"))
	}
	if m["decision_id"] != rec.Header().Get("X-RouteForge-Decision-Id") {
		t.Errorf("decision_id = %v, want %q", m["decision_id"], rec.Header().Get("X-RouteForge-Decision-Id"))
	}
	if m["status"] != float64(200) || m["tenant"] != "tenant-a" || m["model"] != "m-strong" {
		t.Errorf("log line = %v, want status 200, tenant tenant-a, model m-strong", m)
	}
	if m["method"] != "POST" || m["path"] != "/v1/chat/completions" {
		t.Errorf("log line = %v, want method POST, path /v1/chat/completions", m)
	}
	if _, ok := m["duration_ms"]; !ok {
		t.Errorf("log line has no duration_ms: %v", m)
	}
}
