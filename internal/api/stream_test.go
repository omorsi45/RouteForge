package api

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

const streamBody = `{"model":"m-strong","stream":true,"messages":[{"role":"user","content":"hi"}]}`

const streamBodyUsage = `{"model":"m-strong","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`

// readSSE splits an SSE body into JSON chunks, the [DONE] marker and any
// error event.
func readSSE(t *testing.T, body string) (chunks []map[string]any, done bool, errEvent map[string]any) {
	t.Helper()
	for _, frame := range strings.Split(body, "\n\n") {
		if frame == "" {
			continue
		}
		payload, ok := strings.CutPrefix(frame, "data: ")
		if !ok {
			t.Fatalf("frame %q does not start with \"data: \"", frame)
		}
		if payload == "[DONE]" {
			done = true
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(payload), &m); err != nil {
			t.Fatalf("frame %q is not JSON: %v", frame, err)
		}
		if _, isErr := m["error"]; isErr {
			errEvent = m
			continue
		}
		chunks = append(chunks, m)
	}
	return chunks, done, errEvent
}

// streamingProvider returns a stub whose Stream yields st, wiring the call
// context into a blocking stream.
func streamingProvider(st *stubStream) *stubProvider {
	return &stubProvider{stream: func(ctx context.Context, _ provider.Request) (provider.Stream, error) {
		st.ctx = ctx
		return st, nil
	}}
}

// choiceOf returns choices[0] of a chunk.
func choiceOf(t *testing.T, chunk map[string]any) map[string]any {
	t.Helper()
	choices, ok := chunk["choices"].([]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("chunk %v: want exactly one choice", chunk)
	}
	return choices[0].(map[string]any)
}

func deltaOf(t *testing.T, chunk map[string]any) map[string]any {
	t.Helper()
	return choiceOf(t, chunk)["delta"].(map[string]any)
}

// decisionRouter captures the decision the wrapped router makes.
type decisionRouter struct {
	inner router.Router
	mu    sync.Mutex
	d     *router.Decision
}

func (c *decisionRouter) Decide(ctx context.Context, req *router.Request) (*router.Decision, error) {
	d, err := c.inner.Decide(ctx, req)
	c.mu.Lock()
	c.d = d
	c.mu.Unlock()
	return d, err
}

func (c *decisionRouter) decision() *router.Decision {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.d
}

func TestStreamHappyPath(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{
		{Delta: "Hel"},
		{Delta: "lo"},
		{FinishReason: "stop"},
		{Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: 2}},
	}}
	strong := streamingProvider(st)
	cfg := testConfig()
	var capture *decisionRouter
	srv := newServer(t, cfg, map[string]provider.Provider{"p-strong": strong}, func(o *Options) {
		capture = &decisionRouter{inner: o.Router}
		o.Router = capture
	})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	h := rec.Header()
	for name, want := range map[string]string{
		"Content-Type":                "text/event-stream",
		"Cache-Control":               "no-cache",
		"X-Accel-Buffering":           "no",
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

	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if errEvent != nil {
		t.Fatalf("unexpected error event %v", errEvent)
	}
	if !done {
		t.Error("stream did not end with [DONE]")
	}
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3 (no usage chunk without include_usage): %s", len(chunks), rec.Body)
	}
	if got, want := deltaOf(t, chunks[0]), map[string]any{"role": "assistant", "content": "Hel"}; !reflect.DeepEqual(got, want) {
		t.Errorf("first delta = %v, want %v", got, want)
	}
	if got, want := deltaOf(t, chunks[1]), map[string]any{"content": "lo"}; !reflect.DeepEqual(got, want) {
		t.Errorf("second delta = %v, want %v", got, want)
	}
	if got := deltaOf(t, chunks[2]); len(got) != 0 {
		t.Errorf("third delta = %v, want {}", got)
	}
	if got := choiceOf(t, chunks[2])["finish_reason"]; got != "stop" {
		t.Errorf("finish_reason = %v, want stop", got)
	}
	if got := choiceOf(t, chunks[0])["finish_reason"]; got != nil {
		t.Errorf("content chunk finish_reason = %v, want null", got)
	}
	for i, c := range chunks {
		if id, _ := c["id"].(string); !regexp.MustCompile(`^chatcmpl-[0-9a-f]{24}$`).MatchString(id) || id != chunks[0]["id"] {
			t.Errorf("chunk %d id = %v, want the shared chatcmpl id", i, c["id"])
		}
		if c["created"] != chunks[0]["created"] {
			t.Errorf("chunk %d created = %v, want %v", i, c["created"], chunks[0]["created"])
		}
		if c["model"] != "m-strong" {
			t.Errorf("chunk %d model = %v, want m-strong", i, c["model"])
		}
		if c["object"] != "chat.completion.chunk" {
			t.Errorf("chunk %d object = %v, want chat.completion.chunk", i, c["object"])
		}
	}

	// The handler has returned, so Close ran and the attempt is recorded.
	d := capture.decision()
	if d == nil || len(d.Attempts) != 1 || d.Attempts[0].Outcome != router.OutcomeOK {
		t.Errorf("attempts = %+v, want one ok attempt after draining to EOF", d)
	}
	if !st.closed.Load() {
		t.Error("provider stream was not closed")
	}
}

func TestStreamIncludeUsage(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{
		{Delta: "Hel"},
		{Delta: "lo"},
		{FinishReason: "stop"},
		{Usage: &provider.Usage{PromptTokens: 5, CompletionTokens: 2}},
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBodyUsage, nil)

	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if errEvent != nil || !done {
		t.Fatalf("done = %v, errEvent = %v; body %s", done, errEvent, rec.Body)
	}
	if len(chunks) != 4 {
		t.Fatalf("got %d chunks, want 4: %s", len(chunks), rec.Body)
	}
	last := chunks[3]
	if choices, ok := last["choices"].([]any); !ok || len(choices) != 0 {
		t.Errorf("usage chunk choices = %v, want []", last["choices"])
	}
	want := map[string]any{"prompt_tokens": float64(5), "completion_tokens": float64(2), "total_tokens": float64(7)}
	if !reflect.DeepEqual(last["usage"], want) {
		t.Errorf("usage = %v, want %v", last["usage"], want)
	}
	if !strings.Contains(rec.Body.String(), `"choices":[]`) {
		t.Errorf("usage chunk must encode choices as [] and not null: %s", rec.Body)
	}
	for i, c := range chunks[:3] {
		if _, has := c["usage"]; has {
			t.Errorf("chunk %d carries usage", i)
		}
	}
}

// A provider event may carry content, a finish reason and usage together. The
// writer emits one chunk per set field in that order.
func TestStreamCombinedEvent(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{
		{Delta: "Hi", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 3, CompletionTokens: 1}},
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})

	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBodyUsage, nil)
	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if errEvent != nil || !done {
		t.Fatalf("done = %v, errEvent = %v; body %s", done, errEvent, rec.Body)
	}
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks, want 3: %s", len(chunks), rec.Body)
	}
	if got, want := deltaOf(t, chunks[0]), map[string]any{"role": "assistant", "content": "Hi"}; !reflect.DeepEqual(got, want) {
		t.Errorf("content delta = %v, want %v", got, want)
	}
	if got := choiceOf(t, chunks[1])["finish_reason"]; got != "stop" {
		t.Errorf("second chunk finish_reason = %v, want stop", got)
	}
	if got := deltaOf(t, chunks[1]); len(got) != 0 {
		t.Errorf("finish delta = %v, want {}", got)
	}
	if choices := chunks[2]["choices"].([]any); len(choices) != 0 {
		t.Errorf("usage chunk choices = %v, want []", choices)
	}
	if u := chunks[2]["usage"].(map[string]any); u["total_tokens"] != float64(4) {
		t.Errorf("usage = %v, want total 4", u)
	}

	// Without include_usage the same event yields content and finish only.
	st2 := &stubStream{events: []provider.StreamEvent{
		{Delta: "Hi", FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 3, CompletionTokens: 1}},
	}}
	srv2 := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st2)})
	rec2 := do(t, srv2.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)
	chunks2, _, _ := readSSE(t, rec2.Body.String())
	if len(chunks2) != 2 {
		t.Errorf("got %d chunks without include_usage, want 2: %s", len(chunks2), rec2.Body)
	}
}

func TestStreamMissingUsage(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{{Delta: "ok"}, {FinishReason: "stop"}}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBodyUsage, nil)

	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if errEvent != nil || !done {
		t.Fatalf("done = %v, errEvent = %v; body %s", done, errEvent, rec.Body)
	}
	if len(chunks) != 2 {
		t.Errorf("got %d chunks, want 2 (no usage chunk): %s", len(chunks), rec.Body)
	}
}

func TestStreamMultibyteDeltas(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{{Delta: "h\u00e9llo "}, {Delta: "\u4e16\u754c"}}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)

	chunks, done, _ := readSSE(t, rec.Body.String())
	if !done {
		t.Fatalf("stream did not end with [DONE]: %s", rec.Body)
	}
	var sb strings.Builder
	for _, c := range chunks {
		if s, ok := deltaOf(t, c)["content"].(string); ok {
			sb.WriteString(s)
		}
	}
	if got, want := sb.String(), "h\u00e9llo \u4e16\u754c"; got != want {
		t.Errorf("content = %q, want %q", got, want)
	}
}

func TestStreamMidStreamFailure(t *testing.T) {
	st := &stubStream{
		events: []provider.StreamEvent{{Delta: "a"}, {Delta: "b"}},
		err:    &provider.Error{Kind: provider.KindUnavailable, Message: "secret upstream detail"},
	}
	strong := streamingProvider(st)
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": strong})
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if len(chunks) != 2 {
		t.Errorf("got %d content chunks, want 2: %s", len(chunks), rec.Body)
	}
	want := map[string]any{"error": map[string]any{
		"message": "upstream stream interrupted",
		"type":    "server_error",
		"code":    "upstream_stream_interrupted",
	}}
	if !reflect.DeepEqual(errEvent, want) {
		t.Errorf("error event = %v, want %v", errEvent, want)
	}
	if done {
		t.Error("[DONE] written after an error event")
	}
	wantTail := `data: {"error":{"message":"upstream stream interrupted","type":"server_error","code":"upstream_stream_interrupted"}}` + "\n\n"
	if !strings.HasSuffix(rec.Body.String(), wantTail) {
		t.Errorf("body does not end with the error event: %q", rec.Body)
	}
	if strings.Contains(rec.Body.String(), "secret upstream detail") {
		t.Errorf("error event leaks upstream text: %s", rec.Body)
	}
	if got := strong.callCount(); got != 1 {
		t.Errorf("provider called %d times, want 1 (no retry after the first byte)", got)
	}
}

func TestStreamDeadlineMidStream(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{{Delta: "a"}}, block: true}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})

	start := time.Now()
	rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, map[string]string{"X-RouteForge-Deadline-Ms": "100"})
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("handler took %v, want under 2s", elapsed)
	}

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers were already sent); body %s", rec.Code, rec.Body)
	}
	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if len(chunks) != 1 {
		t.Errorf("got %d chunks, want 1: %s", len(chunks), rec.Body)
	}
	if done {
		t.Error("[DONE] written after a deadline error event")
	}
	body, _ := errEvent["error"].(map[string]any)
	if body["code"] != "deadline_exceeded" || body["message"] != "deadline exceeded" || body["type"] != "server_error" {
		t.Errorf("error event = %v, want deadline_exceeded / deadline exceeded / server_error", errEvent)
	}
}

func TestStreamFailureBeforeFirstEventIsJSONError(t *testing.T) {
	t.Run("first Recv fails", func(t *testing.T) {
		st := &stubStream{err: &provider.Error{Kind: provider.KindUnavailable, Message: "secret upstream detail"}}
		srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})
		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)

		wantError(t, rec, http.StatusBadGateway, CodeUpstreamError)
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
		if strings.Contains(rec.Body.String(), "data:") {
			t.Errorf("body looks like SSE: %s", rec.Body)
		}
		if strings.Contains(rec.Body.String(), "secret upstream detail") {
			t.Errorf("body leaks upstream text: %s", rec.Body)
		}
		if got := rec.Header().Get("X-RouteForge-Model"); got != "" {
			t.Errorf("X-RouteForge-Model = %q, want unset before commit", got)
		}
		if !st.closed.Load() {
			t.Error("provider stream was not closed after the priming failure")
		}
	})

	t.Run("Stream call fails", func(t *testing.T) {
		p := &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
			return nil, &provider.Error{Kind: provider.KindRateLimited, RetryAfter: 2 * time.Second}
		}}
		srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": p})
		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, nil)

		wantError(t, rec, http.StatusTooManyRequests, CodeRateLimited)
		if got := rec.Header().Get("Retry-After"); got != "2" {
			t.Errorf("Retry-After = %q, want 2", got)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
	})

	t.Run("deadline before first event", func(t *testing.T) {
		st := &stubStream{block: true}
		srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": streamingProvider(st)})
		rec := do(t, srv.Handler(), http.MethodPost, "/v1/chat/completions", keyA, streamBody, map[string]string{"X-RouteForge-Deadline-Ms": "50"})
		wantError(t, rec, http.StatusGatewayTimeout, CodeDeadlineExceeded)
	})
}

func TestStreamClientDisconnectCancelsUpstream(t *testing.T) {
	st := &stubStream{events: []provider.StreamEvent{{Delta: "first"}}, block: true}
	p := streamingProvider(st)
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": p})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/v1/chat/completions", strings.NewReader(streamBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+keyA)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatalf("reading first chunk: %v", err)
	}
	if !strings.HasPrefix(line, "data: ") || !strings.Contains(line, `"first"`) {
		t.Fatalf("first line = %q, want the first content chunk", line)
	}

	upstream := p.lastCtx(t)
	if upstream.Err() != nil {
		t.Fatal("upstream context ended before the client left")
	}
	cancel()

	deadline := time.After(time.Second)
	for !(upstream.Err() != nil && st.closed.Load()) {
		select {
		case <-deadline:
			t.Fatalf("within 1s of disconnect: upstream ctx err = %v, stream closed = %v; want canceled and closed", upstream.Err(), st.closed.Load())
		case <-time.After(5 * time.Millisecond):
		}
	}

	// The handler goroutine must have returned: Close waits for it.
	closed := make(chan struct{})
	go func() {
		ts.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("server.Close hung: the handler did not return after the client left")
	}
}

// cancelingStream delivers one event, then cancels the request context from
// inside Recv and fails with context.Canceled, as a provider does when the
// client leaves mid-stream.
type cancelingStream struct {
	cancel context.CancelFunc
	sent   bool
}

func (s *cancelingStream) Recv() (provider.StreamEvent, error) {
	if !s.sent {
		s.sent = true
		return provider.StreamEvent{Delta: "only"}, nil
	}
	s.cancel()
	return provider.StreamEvent{}, context.Canceled
}

func (s *cancelingStream) Close() error { return nil }

// A client that leaves after the response is committed gets no error event and
// no [DONE]: nothing is written after the last delivered chunk.
func TestStreamClientGoneMidStreamWritesNothing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	st := &cancelingStream{cancel: cancel}
	p := &stubProvider{stream: func(context.Context, provider.Request) (provider.Stream, error) {
		return st, nil
	}}
	srv := newServer(t, testConfig(), map[string]provider.Provider{"p-strong": p})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamBody)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+keyA)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	chunks, done, errEvent := readSSE(t, rec.Body.String())
	if len(chunks) != 1 {
		t.Errorf("got %d content chunks, want exactly 1: %s", len(chunks), rec.Body)
	}
	if errEvent != nil {
		t.Errorf("error event written to a client that left: %v", errEvent)
	}
	if done {
		t.Error("[DONE] written to a client that left")
	}
}
