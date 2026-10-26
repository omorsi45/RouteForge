package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/openai"
)

const testKey = "test-key-aaaa"

type capturedRequest struct {
	Method string
	Path   string
	Header http.Header
	Body   map[string]any
}

type captures struct {
	mu   sync.Mutex
	reqs []capturedRequest
}

func (c *captures) add(r capturedRequest) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reqs = append(c.reqs, r)
}

func (c *captures) first(t *testing.T) capturedRequest {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.reqs) == 0 {
		t.Fatal("no request captured")
	}
	return c.reqs[0]
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// serve starts a server that records each request and replies with status,
// headers and the named fixture.
func serve(t *testing.T, status int, headers map[string]string, name string) (*httptest.Server, *captures) {
	t.Helper()
	body := fixture(t, name)
	return serveBytes(t, status, headers, body)
}

func serveBytes(t *testing.T, status int, headers map[string]string, body []byte) (*httptest.Server, *captures) {
	t.Helper()
	caps := &captures{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var decoded map[string]any
		_ = json.Unmarshal(raw, &decoded)
		caps.add(capturedRequest{Method: r.Method, Path: r.URL.Path, Header: r.Header.Clone(), Body: decoded})
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, caps
}

func newProvider(srv *httptest.Server, key, param string) *openai.Provider {
	return openai.New(openai.Options{
		Name:           "openai",
		BaseURL:        srv.URL + "/v1/",
		APIKey:         key,
		MaxTokensParam: param,
		Client:         srv.Client(),
	})
}

func baseRequest() provider.Request {
	return provider.Request{
		Model: "gpt-test",
		Messages: []provider.Message{
			{Role: provider.RoleSystem, Content: "be brief"},
			{Role: provider.RoleUser, Content: "hi"},
		},
		MaxTokens: 64,
	}
}

func asProviderError(t *testing.T, err error) *provider.Error {
	t.Helper()
	var pe *provider.Error
	if !errors.As(err, &pe) {
		t.Fatalf("error %v (%T) is not a *provider.Error", err, err)
	}
	return pe
}

func TestCompleteRequestMapping(t *testing.T) {
	temp, topP := 0.2, 0.9
	srv, caps := serve(t, 200, map[string]string{"Content-Type": "application/json"}, "chat_ok.json")
	p := newProvider(srv, testKey, "max_completion_tokens")
	req := baseRequest()
	req.Temperature = &temp
	req.TopP = &topP
	req.Stop = []string{"END"}
	req.User = "u-1"

	resp, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	got := caps.first(t)
	if got.Method != http.MethodPost || got.Path != "/v1/chat/completions" {
		t.Fatalf("request = %s %s", got.Method, got.Path)
	}
	if v := got.Header.Get("Authorization"); v != "Bearer "+testKey {
		t.Fatalf("Authorization = %q", v)
	}
	if v := got.Header.Get("Content-Type"); v != "application/json" {
		t.Fatalf("Content-Type = %q", v)
	}
	if v := got.Header.Get("Accept"); v != "application/json" {
		t.Fatalf("Accept = %q", v)
	}
	var want map[string]any
	wantJSON := `{"model":"gpt-test","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"max_completion_tokens":64,"temperature":0.2,"top_p":0.9,"stop":["END"],"user":"u-1","stream":false}`
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Body, want) {
		t.Fatalf("body = %v, want %v", got.Body, want)
	}

	wantResp := &provider.Response{
		Content:      "Hello there",
		FinishReason: "stop",
		Usage:        &provider.Usage{PromptTokens: 12, CompletionTokens: 3},
	}
	if !reflect.DeepEqual(resp, wantResp) {
		t.Fatalf("response = %+v, want %+v", resp, wantResp)
	}
}

func TestMaxTokensParamLegacy(t *testing.T) {
	srv, caps := serve(t, 200, nil, "chat_ok.json")
	p := newProvider(srv, "", "max_tokens")

	if _, err := p.Complete(context.Background(), baseRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	got := caps.first(t)
	if v, ok := got.Body["max_tokens"]; !ok || v != float64(64) {
		t.Fatalf("max_tokens = %v (present %v)", v, ok)
	}
	for _, k := range []string{"max_completion_tokens", "temperature", "top_p", "stop", "user", "stream_options"} {
		if _, ok := got.Body[k]; ok {
			t.Fatalf("body has unexpected key %q: %v", k, got.Body)
		}
	}
	if _, ok := got.Header["Authorization"]; ok {
		t.Fatalf("Authorization header sent without an API key: %v", got.Header)
	}
}

func TestMaxTokensOmittedWhenUnset(t *testing.T) {
	srv, caps := serve(t, 200, nil, "chat_ok.json")
	p := newProvider(srv, "", "max_tokens")
	req := baseRequest()
	req.MaxTokens = 0

	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if _, ok := caps.first(t).Body["max_tokens"]; ok {
		t.Fatalf("max_tokens sent for MaxTokens 0: %v", caps.first(t).Body)
	}
}

func TestCompleteFinishReasonAndNullContent(t *testing.T) {
	tests := []struct {
		name, body, wantContent, wantFinish string
	}{
		{"length kept", `{"choices":[{"message":{"content":"x"},"finish_reason":"length"}]}`, "x", "length"},
		{"content_filter kept", `{"choices":[{"message":{"content":"x"},"finish_reason":"content_filter"}]}`, "x", "content_filter"},
		{"unknown normalized", `{"choices":[{"message":{"content":"x"},"finish_reason":"tool_calls"}]}`, "x", "stop"},
		{"null reason normalized", `{"choices":[{"message":{"content":"x"},"finish_reason":null}]}`, "x", "stop"},
		{"null content", `{"choices":[{"message":{"content":null},"finish_reason":"stop"}]}`, "", "stop"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := serveBytes(t, 200, nil, []byte(tc.body))
			resp, err := newProvider(srv, "", "max_tokens").Complete(context.Background(), baseRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if resp.Content != tc.wantContent || resp.FinishReason != tc.wantFinish || resp.Usage != nil {
				t.Fatalf("response = %+v", resp)
			}
		})
	}
}

func TestCompleteBadSuccessBody(t *testing.T) {
	tests := []struct {
		name, body, msg string
	}{
		{"no choices", `{"choices":[]}`, "response has no choices"},
		{"undecodable", `not json`, ""},
		{"trailing data", `{"choices":[{"message":{"content":"x"}}]} {}`, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := serveBytes(t, 200, nil, []byte(tc.body))
			_, err := newProvider(srv, "", "max_tokens").Complete(context.Background(), baseRequest())
			pe := asProviderError(t, err)
			if pe.Kind != provider.KindInternal {
				t.Fatalf("kind = %q", pe.Kind)
			}
			if tc.msg != "" && pe.Message != tc.msg {
				t.Fatalf("message = %q", pe.Message)
			}
		})
	}
}

func recvAll(t *testing.T, s provider.Stream) ([]provider.StreamEvent, error) {
	t.Helper()
	var events []provider.StreamEvent
	for i := 0; i < 20; i++ {
		ev, err := s.Recv()
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	t.Fatal("stream did not terminate")
	return nil, nil
}

func TestStreamWithUsage(t *testing.T) {
	srv, caps := serve(t, 200, map[string]string{"Content-Type": "text/event-stream"}, "stream_usage.sse")
	p := newProvider(srv, testKey, "max_tokens")

	s, err := p.Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	got := caps.first(t)
	if got.Body["stream"] != true {
		t.Fatalf("stream = %v", got.Body["stream"])
	}
	if !reflect.DeepEqual(got.Body["stream_options"], map[string]any{"include_usage": true}) {
		t.Fatalf("stream_options = %v", got.Body["stream_options"])
	}
	if v := got.Header.Get("Accept"); v != "text/event-stream" {
		t.Fatalf("Accept = %q", v)
	}

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v, want io.EOF", err)
	}
	want := []provider.StreamEvent{
		{Delta: "Hel"},
		{Delta: "lo"},
		{FinishReason: "stop"},
		{Usage: &provider.Usage{PromptTokens: 12, CompletionTokens: 2}},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
	for i := 0; i < 2; i++ {
		if _, err := s.Recv(); err != io.EOF {
			t.Fatalf("Recv after EOF #%d = %v, want io.EOF", i, err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestStreamWithoutUsage(t *testing.T) {
	srv, _ := serve(t, 200, map[string]string{"Content-Type": "text/event-stream"}, "stream_no_usage.sse")
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v, want io.EOF", err)
	}
	want := []provider.StreamEvent{{Delta: "Hel"}, {Delta: "lo"}, {FinishReason: "stop"}}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
}

func requireStickyError(t *testing.T, s provider.Stream, first error) {
	t.Helper()
	for i := 0; i < 2; i++ {
		_, err := s.Recv()
		if err != first {
			t.Fatalf("Recv after error #%d = %v, want same error %v", i, err, first)
		}
	}
}

func TestStreamTruncated(t *testing.T) {
	srv, _ := serve(t, 200, map[string]string{"Content-Type": "text/event-stream"}, "stream_truncated.sse")
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	if err == io.EOF {
		t.Fatal("truncated stream ended with io.EOF")
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before [DONE]" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamUnterminatedEventWithoutDone(t *testing.T) {
	// SSEReader dispatches a final event that lacks its blank line; that must
	// not be mistaken for a clean end.
	body := `data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}`
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, []byte(body))
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before [DONE]" {
		t.Fatalf("error = %+v", pe)
	}
}

func TestStreamEmptyBodyWithoutDone(t *testing.T) {
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, nil)
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	_, err = s.Recv()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before [DONE]" {
		t.Fatalf("error = %+v", pe)
	}
}

func TestStreamErrorChunk(t *testing.T) {
	srv, _ := serve(t, 200, map[string]string{"Content-Type": "text/event-stream"}, "stream_error.sse")
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "server overloaded" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamErrorChunkContextLengthAndRedaction(t *testing.T) {
	body := "data: {\"error\":{\"message\":\"bad key test-key-aaaa\",\"type\":\"x\",\"code\":\"context_length_exceeded\"}}\n\n"
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, []byte(body))
	s, err := newProvider(srv, testKey, "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	_, err = s.Recv()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindContextLength {
		t.Fatalf("kind = %q", pe.Kind)
	}
	if strings.Contains(err.Error(), testKey) || !strings.Contains(err.Error(), "[REDACTED]") {
		t.Fatalf("error not redacted: %v", err)
	}
}

func TestStreamBadChunkIsInternalAndSticky(t *testing.T) {
	body := "data: {not json}\n\ndata: [DONE]\n\n"
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, []byte(body))
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	_, err = s.Recv()
	if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
		t.Fatalf("kind = %q", pe.Kind)
	}
	requireStickyError(t, s, err)
}

func TestStreamEventTooLarge(t *testing.T) {
	body := "data: " + strings.Repeat("a", provider.MaxSSEBytes+1) + "\n\n"
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, []byte(body))
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	_, err = s.Recv()
	if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
		t.Fatalf("kind = %q", pe.Kind)
	}
	requireStickyError(t, s, err)
}

func TestStreamSkipsEmptyChunks(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[],"usage":null}`,
		`data: {"choices":[{"delta":{"content":null},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"
	srv, _ := serveBytes(t, 200, map[string]string{"Content-Type": "text/event-stream"}, []byte(body))
	s, err := newProvider(srv, "", "max_tokens").Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v", err)
	}
	want := []provider.StreamEvent{{Delta: "ok", FinishReason: "stop"}}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
}

func TestErrorMapping(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		headers    map[string]string
		body       []byte
		wantKind   provider.ErrorKind
		wantRetry  time.Duration
		wantMsg    string
		msgPrefix  bool
		wantStatus int
	}{
		{
			name: "context length", status: 400, body: fixture(t, "error_context_length.json"),
			wantKind: provider.KindContextLength, wantMsg: "This model's maximum context length is 8192 tokens.", wantStatus: 400,
		},
		{
			name: "rate limited", status: 429, headers: map[string]string{"Retry-After": "7"},
			body:     fixture(t, "error_rate_limited.json"),
			wantKind: provider.KindRateLimited, wantRetry: 7 * time.Second, wantMsg: "Rate limit reached", wantStatus: 429,
		},
		{
			name: "unparseable body", status: 503, body: []byte("<html>"),
			wantKind: provider.KindUnavailable, wantMsg: "Service Unavailable", wantStatus: 503,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			check := func(t *testing.T, err error) {
				t.Helper()
				pe := asProviderError(t, err)
				if pe.Kind != tc.wantKind || pe.Status != tc.wantStatus || pe.RetryAfter != tc.wantRetry ||
					pe.Message != tc.wantMsg || pe.Provider != "openai" {
					t.Fatalf("error = %+v", pe)
				}
			}
			srv, _ := serveBytes(t, tc.status, tc.headers, tc.body)
			p := newProvider(srv, "", "max_tokens")

			_, err := p.Complete(context.Background(), baseRequest())
			check(t, err)

			s, err := p.Stream(context.Background(), baseRequest())
			if s != nil {
				t.Fatalf("Stream returned a non-nil stream on error: %v", s)
			}
			check(t, err)
		})
	}
}

func TestErrorRedactsAPIKey(t *testing.T) {
	srv, _ := serve(t, 401, nil, "error_auth.json")
	p := newProvider(srv, testKey, "max_tokens")

	_, err := p.Complete(context.Background(), baseRequest())
	checkRedacted(t, err)

	_, err = p.Stream(context.Background(), baseRequest())
	checkRedacted(t, err)
}

func checkRedacted(t *testing.T, err error) {
	t.Helper()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindAuth {
		t.Fatalf("kind = %q", pe.Kind)
	}
	if strings.Contains(err.Error(), testKey) {
		t.Fatalf("error leaks the API key: %v", err)
	}
	if !strings.Contains(err.Error(), "[REDACTED]") {
		t.Fatalf("error not redacted: %v", err)
	}
}

func blockingServer(t *testing.T) (*httptest.Server, <-chan struct{}) {
	t.Helper()
	observed := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The server only watches for a client disconnect once the request
		// body has been consumed.
		_, _ = io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
		close(observed)
	}))
	t.Cleanup(srv.Close)
	return srv, observed
}

func TestCompleteCancel(t *testing.T) {
	srv, observed := blockingServer(t)
	p := newProvider(srv, "", "max_tokens")
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := p.Complete(ctx, baseRequest())
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Complete did not return after cancel")
	}
	select {
	case <-observed:
	case <-time.After(time.Second):
		t.Fatal("handler did not observe cancellation")
	}
}

func TestCompleteDeadline(t *testing.T) {
	srv, _ := blockingServer(t)
	p := newProvider(srv, "", "max_tokens")
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		_, err := p.Complete(ctx, baseRequest())
		done <- err
	}()
	select {
	case err := <-done:
		if pe := asProviderError(t, err); pe.Kind != provider.KindTimeout {
			t.Fatalf("kind = %q", pe.Kind)
		}
	case <-time.After(time.Second):
		t.Fatal("Complete did not return after deadline")
	}
}

func TestStreamCancelMidStream(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"))
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s, err := newProvider(srv, "", "max_tokens").Stream(ctx, baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()
	if ev, err := s.Recv(); err != nil || ev.Delta != "Hel" {
		t.Fatalf("first Recv = %+v, %v", ev, err)
	}
	time.AfterFunc(50*time.Millisecond, cancel)

	done := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Recv did not return after cancel")
	}
}

func TestResponseTooLarge(t *testing.T) {
	srv, _ := serveBytes(t, 200, nil, bytes.Repeat([]byte("a"), provider.MaxResponseBytes+1))
	_, err := newProvider(srv, "", "max_tokens").Complete(context.Background(), baseRequest())
	if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
		t.Fatalf("kind = %q", pe.Kind)
	}
}

func TestConnectionRefused(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	p := newProvider(srv, "", "max_tokens")
	srv.Close()

	_, err := p.Complete(context.Background(), baseRequest())
	if pe := asProviderError(t, err); pe.Kind != provider.KindUnavailable {
		t.Fatalf("Complete kind = %q", pe.Kind)
	}
	_, err = p.Stream(context.Background(), baseRequest())
	if pe := asProviderError(t, err); pe.Kind != provider.KindUnavailable {
		t.Fatalf("Stream kind = %q", pe.Kind)
	}
}

func TestProviderImplementsInterface(t *testing.T) {
	var _ provider.Provider = (*openai.Provider)(nil)
}
