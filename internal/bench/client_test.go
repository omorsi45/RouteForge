package bench

import (
	"context"
	"encoding/json"
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
