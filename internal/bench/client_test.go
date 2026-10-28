package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestChatOutcomeFields(t *testing.T) {
	const promptText = "PROMPT-TEXT-ALPHA"
	var gotBody struct {
		Model    string `json:"model"`
		Stream   bool   `json:"stream"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	var gotHeader http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("request = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
		}
		gotHeader = r.Header.Clone()
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Errorf("request body is not JSON: %v", err)
		}
		time.Sleep(5 * time.Millisecond)
		w.Header().Set("X-RouteForge-Model", "gemini-2.5-pro")
		w.Header().Set("X-RouteForge-Reason", "cheapest_passing")
		w.Header().Set("X-RouteForge-Decision-Id", "dec_0123")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"content":"COMPLETION-TEXT"}}],`+
			`"routeforge":{"decision_id":"dec_0123","reason":"cheapest_passing","attempts":1,"cost_usd":0.00125,"cost_source":"reported"}}`)
	}))
	defer srv.Close()

	c := NewClient(srv.URL+"/", testKey) // a trailing slash is tolerated
	out, err := c.Chat(context.Background(), promptText, map[string]string{
		"X-RouteForge-Strategy": "quality",
		"X-RouteForge-Profile":  "balanced",
	})
	if err != nil {
		t.Fatal(err)
	}

	if gotHeader.Get("Authorization") != "Bearer "+testKey {
		t.Errorf("Authorization = %q", gotHeader.Get("Authorization"))
	}
	if gotHeader.Get("X-RouteForge-Strategy") != "quality" || gotHeader.Get("X-RouteForge-Profile") != "balanced" {
		t.Errorf("routing headers = %q, %q", gotHeader.Get("X-RouteForge-Strategy"), gotHeader.Get("X-RouteForge-Profile"))
	}
	if gotBody.Model != "routeforge-auto" || gotBody.Stream {
		t.Errorf("body model = %q, stream = %v, want routeforge-auto and no stream", gotBody.Model, gotBody.Stream)
	}
	if len(gotBody.Messages) != 1 || gotBody.Messages[0].Role != "user" || gotBody.Messages[0].Content != promptText {
		t.Errorf("messages = %+v, want one user message with the prompt", gotBody.Messages)
	}

	if out.Status != http.StatusOK || out.Model != "gemini-2.5-pro" || out.Reason != "cheapest_passing" || out.DecisionID != "dec_0123" {
		t.Errorf("outcome = %+v", out)
	}
	if out.CostUSD == nil || math.Abs(*out.CostUSD-0.00125) > 1e-15 {
		t.Errorf("cost = %v, want 0.00125", out.CostUSD)
	}
	if out.ErrorCode != "" {
		t.Errorf("error code = %q, want empty on success", out.ErrorCode)
	}
	if out.Duration < 5*time.Millisecond {
		t.Errorf("duration = %s, want at least the 5ms the server took", out.Duration)
	}
}

func TestChatOutcomeWithoutCost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-RouteForge-Model", "m1")
		_, _ = io.WriteString(w, `{"routeforge":{"decision_id":"dec_body","attempts":1}}`)
	}))
	defer srv.Close()

	out, err := NewClient(srv.URL, testKey).Chat(context.Background(), "hi", nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.CostUSD != nil {
		t.Errorf("cost = %v, want nil when the response carries none", *out.CostUSD)
	}
	if out.DecisionID != "dec_body" {
		t.Errorf("decision id = %q, want the one in the body when the header is absent", out.DecisionID)
	}
}

func TestChatErrorCodes(t *testing.T) {
	for _, c := range []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{"error envelope", http.StatusTooManyRequests, `{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limited"}}`, "rate_limited"},
		{"not json", http.StatusBadGateway, "<html>bad gateway</html>", "http_502"},
		{"envelope without code", http.StatusInternalServerError, `{"error":{"message":"x"}}`, "http_500"},
		{"empty body", http.StatusServiceUnavailable, "", "http_503"},
	} {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(c.status)
				_, _ = io.WriteString(w, c.body)
			}))
			defer srv.Close()

			out, err := NewClient(srv.URL, testKey).Chat(context.Background(), "hi", nil)
			if err != nil {
				t.Fatalf("an HTTP error response is an outcome, not an error: %v", err)
			}
			if out.Status != c.status || out.ErrorCode != c.want {
				t.Errorf("status = %d, code = %q, want %d, %q", out.Status, out.ErrorCode, c.status, c.want)
			}
			if out.Model != "" || out.CostUSD != nil {
				t.Errorf("a failed request has no model or cost: %+v", out)
			}
		})
	}
}

func TestChatBadSuccessBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "not json")
	}))
	defer srv.Close()

	if _, err := NewClient(srv.URL, testKey).Chat(context.Background(), "hi", nil); err == nil {
		t.Fatal("want an error for a 200 response that is not JSON")
	}
}

func TestChatTransportError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()

	_, err := NewClient(url, testKey).Chat(context.Background(), "PROMPT-TEXT-BETA", nil)
	if err == nil {
		t.Fatal("want an error when the gateway is unreachable")
	}
	if strings.Contains(err.Error(), "PROMPT-TEXT-BETA") {
		t.Errorf("error %q repeats the prompt", err)
	}
}

func TestChatContextCanceled(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := NewClient(srv.URL, testKey).Chat(ctx, "hi", nil); err == nil {
		t.Fatal("want an error when the context ends before the response")
	}
}

// sseServer answers every request with the given SSE frames, one per element.
func sseServer(t *testing.T, headers map[string]string, frames ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(http.StatusOK)
		for _, f := range frames {
			_, _ = io.WriteString(w, "data: "+f+"\n\n")
			w.(http.Flusher).Flush()
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

const contentFrame = `{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"CHUNK-TEXT"},"finish_reason":null}]}`

func TestDoRequestBody(t *testing.T) {
	var got struct {
		Model     string `json:"model"`
		Stream    bool   `json:"stream"`
		MaxTokens *int   `json:"max_tokens"`
	}
	var header http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Clone()
		body, _ := io.ReadAll(r.Body)
		got.Model, got.Stream, got.MaxTokens = "", false, nil // a field the body lacks stays unset
		if err := json.Unmarshal(body, &got); err != nil {
			t.Errorf("body is not JSON: %v", err)
		}
		_, _ = io.WriteString(w, `{"routeforge":{"decision_id":"dec_1"}}`)
	}))
	defer srv.Close()

	_, err := NewClient(srv.URL, "test-key-bench-over").Do(context.Background(), ChatRequest{
		Prompt: "hi", Model: "gemini-2.5-flash", MaxTokens: 16, Headers: map[string]string{"X-RouteForge-Deadline-Ms": "2000"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "gemini-2.5-flash" || got.Stream || got.MaxTokens == nil || *got.MaxTokens != 16 {
		t.Errorf("body = %+v, want the pinned model, no stream and max_tokens 16", got)
	}
	if header.Get("X-RouteForge-Deadline-Ms") != "2000" || header.Get("Authorization") != "Bearer test-key-bench-over" {
		t.Errorf("headers = %v", header)
	}

	// Without a model or a token cap the request asks the router and sends no cap.
	if _, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi"}); err != nil {
		t.Fatal(err)
	}
	if got.Model != "routeforge-auto" || got.MaxTokens != nil {
		t.Errorf("default body = %+v, want routeforge-auto and no max_tokens", got)
	}
}

func TestDoStream(t *testing.T) {
	var asked struct {
		Stream bool `json:"stream"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &asked)
		w.Header().Set("X-RouteForge-Model", "gemini-2.5-pro")
		w.Header().Set("X-RouteForge-Decision-Id", "dec_s1")
		w.Header().Set("Content-Type", "text/event-stream")
		for _, f := range []string{contentFrame, contentFrame, "[DONE]"} {
			_, _ = io.WriteString(w, "data: "+f+"\n\n")
			w.(http.Flusher).Flush()
		}
	}))
	defer srv.Close()

	out, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	if !asked.Stream {
		t.Error("the request did not ask for a stream")
	}
	if out.Status != 200 || !out.Streamed || out.Partial || out.ErrorCode != "" {
		t.Errorf("outcome = %+v, want a streamed 200 that finished", out)
	}
	if out.Model != "gemini-2.5-pro" || out.DecisionID != "dec_s1" {
		t.Errorf("model = %q, decision = %q", out.Model, out.DecisionID)
	}
	if out.CostUSD != nil {
		t.Error("a stream carries no cost")
	}
}

func TestDoStreamEndingWithAnErrorEvent(t *testing.T) {
	srv := sseServer(t, map[string]string{"X-RouteForge-Decision-Id": "dec_p1"}, contentFrame,
		`{"error":{"message":"upstream stream interrupted","type":"server_error","code":"upstream_stream_interrupted"}}`)

	out, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != 200 || !out.Partial || out.ErrorCode != "upstream_stream_interrupted" {
		t.Errorf("outcome = %+v, want a partial stream with the event's code", out)
	}
	if out.DecisionID != "dec_p1" {
		t.Errorf("decision = %q", out.DecisionID)
	}
}

func TestDoStreamCutWithoutAnEnd(t *testing.T) {
	srv := sseServer(t, nil, contentFrame) // no [DONE], no error event

	out, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	if out.Partial || out.ErrorCode != "stream_truncated" {
		t.Errorf("outcome = %+v, want error code stream_truncated and no error event", out)
	}
}

func TestDoStreamRefusedBeforeItStarts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"message":"m","type":"server_error","code":"no_eligible_model"}}`)
	}))
	defer srv.Close()

	out, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != 503 || out.ErrorCode != "no_eligible_model" || out.Partial || !out.Streamed {
		t.Errorf("outcome = %+v, want a refused stream request", out)
	}
}

func TestDoStreamKeepsNoText(t *testing.T) {
	srv := sseServer(t, nil, contentFrame, "[DONE]")
	out, err := NewClient(srv.URL, testKey).Do(context.Background(), ChatRequest{Prompt: "hi", Stream: true})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(fmt.Sprintf("%+v", out), "CHUNK-TEXT") {
		t.Errorf("outcome %+v holds completion text", out)
	}
}

func TestDecision(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.Header.Get("Authorization") != "Bearer "+testKey {
			t.Errorf("request = %s with Authorization %q", r.Method, r.Header.Get("Authorization"))
		}
		switch r.URL.Path {
		case "/v1/routing/decisions/dec_ok":
			_, _ = io.WriteString(w, `{"id":"dec_ok","tenant":"bench","chosen":"gpt-5","decided_in_us":412,"candidates":[{"model":"gpt-5","eligible":true}],`+
				`"attempts":[{"model":"gpt-5","outcome":"error","error_kind":"unavailable","duration_ms":1.25},{"model":"claude-sonnet-4","outcome":"ok","duration_ms":30.5}]}`)
		default:
			w.WriteHeader(http.StatusNotFound)
			_, _ = io.WriteString(w, `{"error":{"message":"decision not found","type":"invalid_request_error","code":"not_found"}}`)
		}
	}))
	defer srv.Close()
	c := NewClient(srv.URL, testKey)

	d, err := c.Decision(context.Background(), "dec_ok")
	if err != nil {
		t.Fatal(err)
	}
	if d.ID != "dec_ok" || d.Chosen != "gpt-5" || d.DecidedInUS != 412 || len(d.Attempts) != 2 {
		t.Fatalf("decision = %+v", d)
	}
	if a := d.Attempts[0]; a.Model != "gpt-5" || a.Outcome != "error" || a.ErrorKind != "unavailable" || a.DurationMS != 1.25 {
		t.Errorf("attempt 0 = %+v", a)
	}

	if _, err := c.Decision(context.Background(), "dec_gone"); !errors.Is(err, ErrDecisionNotFound) {
		t.Errorf("unknown decision: err = %v, want ErrDecisionNotFound", err)
	}
}

func TestDecisionServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	_, err := NewClient(srv.URL, testKey).Decision(context.Background(), "dec_x")
	if err == nil || errors.Is(err, ErrDecisionNotFound) {
		t.Fatalf("err = %v, want a failure that is not 'not found'", err)
	}
}

func TestAccountingState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/admin/accounting" || r.URL.Query().Get("tenant") != "bench" || r.URL.Query().Get("day") != "20261028" {
			t.Errorf("request URL = %s", r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer test-admin-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		_, _ = io.WriteString(w, `{"tenant":"bench","day":"20261028","spent_usd":0.5,"reserved_usd":0,"overshoot_usd":0,`+
			`"spent_nanos":500000000,"reserved_nanos":0,"overshoot_nanos":0,"usage_records":12,"usage_cost_nanos":500000000}`)
	}))
	defer srv.Close()

	st, err := NewClient(srv.URL, "test-admin-key").Accounting(context.Background(), "bench", "20261028")
	if err != nil {
		t.Fatal(err)
	}
	want := AccountingState{Tenant: "bench", Day: "20261028", SpentNanos: 500000000, UsageRecords: 12, UsageCostNanos: 500000000}
	if st != want {
		t.Errorf("state = %+v, want %+v", st, want)
	}
}

func TestAccountingStateErrors(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusNotFound, http.StatusServiceUnavailable} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"error":{"message":"m","type":"t","code":"invalid_api_key"}}`)
		}))
		_, err := NewClient(srv.URL, "wrong").Accounting(context.Background(), "bench", "20261028")
		srv.Close()
		if err == nil || !strings.Contains(err.Error(), fmt.Sprint(status)) {
			t.Errorf("status %d: err = %v, want one that names the status", status, err)
		}
	}
}
