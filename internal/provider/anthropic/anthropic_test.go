package anthropic_test

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
	"github.com/omorsi45/routeforge/internal/provider/anthropic"
)

const testKey = "test-key-bbbb"

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

func (c *captures) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.reqs)
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
	return serveBytes(t, status, headers, fixture(t, name))
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

func newProvider(srv *httptest.Server, key string) *anthropic.Provider {
	return anthropic.New(anthropic.Options{
		Name:    "anthropic",
		BaseURL: srv.URL + "/",
		APIKey:  key,
		Client:  srv.Client(),
	})
}

func baseRequest() provider.Request {
	return provider.Request{
		Model:     "claude-test",
		Messages:  []provider.Message{{Role: provider.RoleUser, Content: "hi"}},
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

func sseHeaders() map[string]string { return map[string]string{"Content-Type": "text/event-stream"} }

func openStream(t *testing.T, srv *httptest.Server, key string) provider.Stream {
	t.Helper()
	s, err := newProvider(srv, key).Stream(context.Background(), baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestCompleteRequestMapping(t *testing.T) {
	temp, topP := 0.2, 0.9
	srv, caps := serve(t, 200, map[string]string{"Content-Type": "application/json"}, "message_ok.json")
	p := newProvider(srv, testKey)
	req := provider.Request{
		Model: "claude-test",
		Messages: []provider.Message{
			{Role: provider.RoleSystem, Content: "be brief"},
			{Role: provider.RoleUser, Content: "hi"},
			{Role: provider.RoleSystem, Content: "use English"},
			{Role: provider.RoleAssistant, Content: "ok"},
			{Role: provider.RoleUser, Content: "go"},
		},
		MaxTokens:   64,
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
		User:        "u-1",
	}

	resp, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	got := caps.first(t)
	if got.Method != http.MethodPost || got.Path != "/v1/messages" {
		t.Fatalf("request = %s %s", got.Method, got.Path)
	}
	if v := got.Header.Get("x-api-key"); v != testKey {
		t.Fatalf("x-api-key = %q", v)
	}
	if v := got.Header.Get("anthropic-version"); v != "2023-06-01" {
		t.Fatalf("anthropic-version = %q", v)
	}
	if v := got.Header.Get("Content-Type"); v != "application/json" {
		t.Fatalf("Content-Type = %q", v)
	}
	if v := got.Header.Get("Accept"); v != "application/json" {
		t.Fatalf("Accept = %q", v)
	}
	var want map[string]any
	wantJSON := `{"model":"claude-test","max_tokens":64,"system":"be brief\n\nuse English","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},{"role":"user","content":"go"}],"temperature":0.2,"top_p":0.9,"stop_sequences":["END"],"metadata":{"user_id":"u-1"},"stream":false}`
	if err := json.Unmarshal([]byte(wantJSON), &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Body, want) {
		t.Fatalf("body = %v, want %v", got.Body, want)
	}

	wantResp := &provider.Response{
		Content:      "Hello there",
		FinishReason: "stop",
		Usage:        &provider.Usage{PromptTokens: 25, CompletionTokens: 4},
	}
	if !reflect.DeepEqual(resp, wantResp) {
		t.Fatalf("response = %+v, want %+v", resp, wantResp)
	}
}

func TestNoSystemOmitsField(t *testing.T) {
	srv, caps := serve(t, 200, nil, "message_ok.json")

	if _, err := newProvider(srv, "").Complete(context.Background(), baseRequest()); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	got := caps.first(t)
	for _, k := range []string{"system", "metadata", "stop_sequences", "temperature", "top_p"} {
		if _, ok := got.Body[k]; ok {
			t.Fatalf("body has unexpected key %q: %v", k, got.Body)
		}
	}
	if got.Body["max_tokens"] != float64(64) {
		t.Fatalf("max_tokens = %v", got.Body["max_tokens"])
	}
}

func TestDefaultMaxTokens(t *testing.T) {
	srv, caps := serve(t, 200, nil, "message_ok.json")
	req := baseRequest()
	req.MaxTokens = 0

	if _, err := newProvider(srv, "").Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if v := caps.first(t).Body["max_tokens"]; v != float64(4096) {
		t.Fatalf("max_tokens = %v, want 4096", v)
	}
}

func TestSystemOnlyIsBadRequest(t *testing.T) {
	srv, caps := serve(t, 200, nil, "message_ok.json")
	p := newProvider(srv, testKey)
	req := baseRequest()
	req.Messages = []provider.Message{
		{Role: provider.RoleSystem, Content: "a"},
		{Role: provider.RoleSystem, Content: "b"},
	}
	const wantMsg = "anthropic: at least one user or assistant message is required"

	_, err := p.Complete(context.Background(), req)
	if pe := asProviderError(t, err); pe.Kind != provider.KindBadRequest || pe.Message != wantMsg {
		t.Fatalf("Complete error = %+v", pe)
	}
	s, err := p.Stream(context.Background(), req)
	if s != nil {
		t.Fatalf("Stream returned a non-nil stream on error: %v", s)
	}
	if pe := asProviderError(t, err); pe.Kind != provider.KindBadRequest || pe.Message != wantMsg {
		t.Fatalf("Stream error = %+v", pe)
	}
	if n := caps.count(); n != 0 {
		t.Fatalf("server received %d requests, want 0", n)
	}
}

func TestCompleteStopReasonsAndUsage(t *testing.T) {
	tests := []struct {
		name, body, wantContent, wantFinish string
		wantUsage                           *provider.Usage
	}{
		{"stop_sequence", `{"content":[{"type":"text","text":"x"}],"stop_reason":"stop_sequence"}`, "x", "stop", nil},
		{"max_tokens", `{"content":[{"type":"text","text":"x"}],"stop_reason":"max_tokens"}`, "x", "length", nil},
		{"refusal", `{"content":[{"type":"text","text":"x"}],"stop_reason":"refusal"}`, "x", "content_filter", nil},
		{"unknown", `{"content":[{"type":"text","text":"x"}],"stop_reason":"tool_use"}`, "x", "stop", nil},
		{"null reason", `{"content":[{"type":"text","text":"x"}],"stop_reason":null}`, "x", "stop", nil},
		{"non text blocks skipped", `{"content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"y"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`, "y", "stop", &provider.Usage{PromptTokens: 1, CompletionTokens: 2}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := serveBytes(t, 200, nil, []byte(tc.body))
			resp, err := newProvider(srv, "").Complete(context.Background(), baseRequest())
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			want := &provider.Response{Content: tc.wantContent, FinishReason: tc.wantFinish, Usage: tc.wantUsage}
			if !reflect.DeepEqual(resp, want) {
				t.Fatalf("response = %+v, want %+v", resp, want)
			}
		})
	}
}

func TestCompleteBadSuccessBody(t *testing.T) {
	tests := []struct{ name, body string }{
		{"undecodable", `not json`},
		{"trailing data", `{"content":[{"type":"text","text":"x"}]} {}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := serveBytes(t, 200, nil, []byte(tc.body))
			_, err := newProvider(srv, "").Complete(context.Background(), baseRequest())
			if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
				t.Fatalf("kind = %q", pe.Kind)
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

func requireStickyError(t *testing.T, s provider.Stream, first error) {
	t.Helper()
	for i := 0; i < 2; i++ {
		_, err := s.Recv()
		if err != first {
			t.Fatalf("Recv after error #%d = %v, want same error %v", i, err, first)
		}
	}
}

func TestStreamEventsAndCumulativeUsage(t *testing.T) {
	srv, caps := serve(t, 200, sseHeaders(), "stream_ok.sse")
	s := openStream(t, srv, testKey)

	got := caps.first(t)
	if got.Body["stream"] != true {
		t.Fatalf("stream = %v", got.Body["stream"])
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
		{FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 25, CompletionTokens: 15}},
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

func TestStopReasonMapping(t *testing.T) {
	for reason, want := range map[string]string{
		"max_tokens": "length",
		"refusal":    "content_filter",
	} {
		t.Run(reason, func(t *testing.T) {
			body := bytes.ReplaceAll(fixture(t, "stream_ok.sse"), []byte(`"stop_reason":"end_turn"`), []byte(`"stop_reason":"`+reason+`"`))
			srv, _ := serveBytes(t, 200, sseHeaders(), body)
			s := openStream(t, srv, "")

			events, err := recvAll(t, s)
			if err != io.EOF {
				t.Fatalf("final error = %v, want io.EOF", err)
			}
			if last := events[len(events)-1]; last.FinishReason != want {
				t.Fatalf("final event = %+v, want finish %q", last, want)
			}
		})
	}
}

func TestStreamWithoutUsageOrStopReason(t *testing.T) {
	body := "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
		"data: {\"type\":\"message_stop\"}\n\n"
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v, want io.EOF", err)
	}
	want := []provider.StreamEvent{{Delta: "ok"}, {FinishReason: "stop"}}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
}

func TestStreamMessageDeltaInputTokensReplacePrompt(t *testing.T) {
	body := strings.Join([]string{
		`data: {"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":30,"cache_read_input_tokens":5,"output_tokens":9}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v, want io.EOF", err)
	}
	want := []provider.StreamEvent{{FinishReason: "stop", Usage: &provider.Usage{PromptTokens: 35, CompletionTokens: 9}}}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
}

func TestStreamIgnoresUnknownEventsAndNonTextDeltas(t *testing.T) {
	body := strings.Join([]string{
		`event: ping` + "\n" + `data: {"type":"ping"}`,
		`event: future_event` + "\n" + `data: {"type":"future_event","payload":{"a":1}}`,
		`data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":""}}`,
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}`,
		`data: {"type":"message_stop"}`,
	}, "\n\n") + "\n\n"
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if err != io.EOF {
		t.Fatalf("final error = %v, want io.EOF", err)
	}
	want := []provider.StreamEvent{{Delta: "ok"}, {FinishReason: "stop"}}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %+v, want %+v", events, want)
	}
}

func TestStreamCRLF(t *testing.T) {
	body := strings.ReplaceAll(string(fixture(t, "stream_ok.sse")), "\n", "\r\n")
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if err != io.EOF || len(events) != 3 {
		t.Fatalf("events = %+v, err = %v", events, err)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	srv, _ := serve(t, 200, sseHeaders(), "stream_error.sse")
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "Overloaded" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamErrorEventKindsAndRedaction(t *testing.T) {
	tests := []struct {
		name, data string
		wantKind   provider.ErrorKind
	}{
		{"rate limit", `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`, provider.KindRateLimited},
		{"context length", `{"type":"error","error":{"type":"invalid_request_error","message":"Prompt is too long"}}`, provider.KindContextLength},
		{"auth with key", `{"type":"error","error":{"type":"authentication_error","message":"bad key test-key-bbbb"}}`, provider.KindAuth},
		{"unknown type", `{"type":"error","error":{"type":"mystery","message":"huh"}}`, provider.KindUnavailable},
		{"no detail", `{"type":"error"}`, provider.KindUnavailable},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := serveBytes(t, 200, sseHeaders(), []byte("event: error\ndata: "+tc.data+"\n\n"))
			s := openStream(t, srv, testKey)

			_, err := s.Recv()
			if pe := asProviderError(t, err); pe.Kind != tc.wantKind || pe.Message == "" {
				t.Fatalf("error = %+v, want kind %q", pe, tc.wantKind)
			}
			if strings.Contains(err.Error(), testKey) {
				t.Fatalf("error leaks the API key: %v", err)
			}
			requireStickyError(t, s, err)
		})
	}
	t.Run("redacted marker", func(t *testing.T) {
		body := `data: {"type":"error","error":{"type":"authentication_error","message":"bad key test-key-bbbb"}}` + "\n\n"
		srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
		s := openStream(t, srv, testKey)
		_, err := s.Recv()
		if !strings.Contains(err.Error(), "[REDACTED]") {
			t.Fatalf("error not redacted: %v", err)
		}
	})
}

func TestStreamTruncated(t *testing.T) {
	srv, _ := serve(t, 200, sseHeaders(), "stream_truncated.sse")
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	if err == io.EOF {
		t.Fatal("truncated stream ended with io.EOF")
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before message_stop" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamEmptyBodyWithoutMessageStop(t *testing.T) {
	srv, _ := serveBytes(t, 200, sseHeaders(), nil)
	s := openStream(t, srv, "")

	_, err := s.Recv()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before message_stop" {
		t.Fatalf("error = %+v", pe)
	}
}

func TestStreamUnterminatedDeltaWithoutMessageStop(t *testing.T) {
	// SSEReader dispatches a final event that lacks its blank line; that must
	// not be mistaken for a clean end.
	body := `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}`
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	events, err := recvAll(t, s)
	if !reflect.DeepEqual(events, []provider.StreamEvent{{Delta: "Hel"}}) {
		t.Fatalf("events = %+v", events)
	}
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "stream ended before message_stop" {
		t.Fatalf("error = %+v", pe)
	}
}

func TestStreamCutMidJSONIsUnavailableAndSticky(t *testing.T) {
	body := `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}` + "\n\n" +
		`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"He`
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	ev, err := s.Recv()
	if err != nil || ev.Delta != "Hel" {
		t.Fatalf("first Recv = %+v, %v", ev, err)
	}
	_, err = s.Recv()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "undecodable stream event" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamUndecodableEventIsUnavailableAndSticky(t *testing.T) {
	body := "data: {not json}\n\n" + `data: {"type":"message_stop"}` + "\n\n"
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	_, err := s.Recv()
	pe := asProviderError(t, err)
	if pe.Kind != provider.KindUnavailable || pe.Message != "undecodable stream event" {
		t.Fatalf("error = %+v", pe)
	}
	requireStickyError(t, s, err)
}

func TestStreamUndecodableDeltaIsUnavailableAndSticky(t *testing.T) {
	tests := []struct{ name, data string }{
		{"content_block_delta", `{"type":"content_block_delta","delta":"nope"}`},
		{"message_delta", `{"type":"message_delta","delta":[1]}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := "data: " + tc.data + "\n\n" + `data: {"type":"message_stop"}` + "\n\n"
			srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
			s := openStream(t, srv, "")

			_, err := s.Recv()
			pe := asProviderError(t, err)
			if pe.Kind != provider.KindUnavailable || pe.Message != "undecodable stream event" {
				t.Fatalf("error = %+v", pe)
			}
			requireStickyError(t, s, err)
		})
	}
}

func TestStreamEventTooLarge(t *testing.T) {
	body := "data: " + strings.Repeat("a", provider.MaxSSEBytes+1) + "\n\n"
	srv, _ := serveBytes(t, 200, sseHeaders(), []byte(body))
	s := openStream(t, srv, "")

	_, err := s.Recv()
	if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
		t.Fatalf("kind = %q", pe.Kind)
	}
	requireStickyError(t, s, err)
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
		wantStatus int
	}{
		{
			name: "overloaded", status: 529, body: fixture(t, "error_overloaded.json"),
			wantKind: provider.KindUnavailable, wantMsg: "Overloaded", wantStatus: 529,
		},
		{
			name: "rate limited", status: 429, headers: map[string]string{"retry-after": "4"},
			body:     fixture(t, "error_rate_limited.json"),
			wantKind: provider.KindRateLimited, wantRetry: 4 * time.Second, wantMsg: "Rate limit reached", wantStatus: 429,
		},
		{
			name: "prompt too long", status: 400, body: fixture(t, "error_prompt_too_long.json"),
			wantKind: provider.KindContextLength, wantMsg: "prompt is too long: 210000 tokens > 200000 maximum", wantStatus: 400,
		},
		{
			name: "invalid request", status: 400,
			body:     []byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad field"}}`),
			wantKind: provider.KindBadRequest, wantMsg: "bad field", wantStatus: 400,
		},
		{
			name: "permission", status: 403,
			body:     []byte(`{"type":"error","error":{"type":"permission_error","message":"no access"}}`),
			wantKind: provider.KindAuth, wantMsg: "no access", wantStatus: 403,
		},
		{
			name: "not found", status: 404,
			body:     []byte(`{"type":"error","error":{"type":"not_found_error","message":"no model"}}`),
			wantKind: provider.KindBadRequest, wantMsg: "no model", wantStatus: 404,
		},
		{
			name: "request too large", status: 413,
			body:     []byte(`{"type":"error","error":{"type":"request_too_large","message":"too big"}}`),
			wantKind: provider.KindBadRequest, wantMsg: "too big", wantStatus: 413,
		},
		{
			name: "api error", status: 500,
			body:     []byte(`{"type":"error","error":{"type":"api_error","message":"internal"}}`),
			wantKind: provider.KindUnavailable, wantMsg: "internal", wantStatus: 500,
		},
		{
			name: "timeout error", status: 504,
			body:     []byte(`{"type":"error","error":{"type":"timeout_error","message":"slow"}}`),
			wantKind: provider.KindTimeout, wantMsg: "slow", wantStatus: 504,
		},
		{
			name: "unknown type falls back to status", status: 429,
			body:     []byte(`{"type":"error","error":{"type":"mystery","message":"huh"}}`),
			wantKind: provider.KindRateLimited, wantMsg: "huh", wantStatus: 429,
		},
		{
			name: "unparseable body falls back to status", status: 500, body: []byte("oops"),
			wantKind: provider.KindUnavailable, wantMsg: "Internal Server Error", wantStatus: 500,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			check := func(t *testing.T, err error) {
				t.Helper()
				pe := asProviderError(t, err)
				if pe.Kind != tc.wantKind || pe.Status != tc.wantStatus || pe.RetryAfter != tc.wantRetry ||
					pe.Message != tc.wantMsg || pe.Provider != "anthropic" {
					t.Fatalf("error = %+v", pe)
				}
			}
			srv, _ := serveBytes(t, tc.status, tc.headers, tc.body)
			p := newProvider(srv, "")

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
	p := newProvider(srv, testKey)

	check := func(t *testing.T, err error) {
		t.Helper()
		if pe := asProviderError(t, err); pe.Kind != provider.KindAuth {
			t.Fatalf("kind = %q", pe.Kind)
		}
		if strings.Contains(err.Error(), testKey) {
			t.Fatalf("error leaks the API key: %v", err)
		}
		if !strings.Contains(err.Error(), "[REDACTED]") {
			t.Fatalf("error not redacted: %v", err)
		}
	}
	_, err := p.Complete(context.Background(), baseRequest())
	check(t, err)
	_, err = p.Stream(context.Background(), baseRequest())
	check(t, err)
}

func TestStreamCancel(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = w.Write(fixtureHead())
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
	s, err := newProvider(srv, "").Stream(ctx, baseRequest())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer s.Close()
	if ev, err := s.Recv(); err != nil || ev.Delta != "Hel" {
		t.Fatalf("first Recv = %+v, %v", ev, err)
	}
	cancel()

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

// fixtureHead is a message_start event followed by one "Hel" text delta.
func fixtureHead() []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}` + "\n\n")
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
	p := newProvider(srv, "")
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
	p := newProvider(srv, "")
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

func TestResponseTooLarge(t *testing.T) {
	srv, _ := serveBytes(t, 200, nil, bytes.Repeat([]byte("a"), provider.MaxResponseBytes+1))
	_, err := newProvider(srv, "").Complete(context.Background(), baseRequest())
	if pe := asProviderError(t, err); pe.Kind != provider.KindInternal {
		t.Fatalf("kind = %q", pe.Kind)
	}
}

func TestConnectionRefused(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	p := newProvider(srv, "")
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
	var _ provider.Provider = (*anthropic.Provider)(nil)
}
