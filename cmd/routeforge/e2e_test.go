package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider/fake"
)

const e2eKey = "test-key-e2e"

// e2eClient opens a fresh connection per request. A pooled keep-alive client
// can leave a dialed but never-used connection behind, which the server counts
// as active until it is five seconds old and so delays graceful shutdown.
var e2eClient = &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}

func e2eConfigYAML() string {
	return fmt.Sprintf(`
tenants:
  - name: e2e
    api_key_sha256:
      - %s
providers:
  - name: fast
    type: fake
    fake:
      seed: 1
  - name: strong
    type: fake
    fake:
      seed: 2
  - name: flaky
    type: fake
    fake:
      seed: 3
      disconnect_rate: 1
      disconnect_after: 2
      output_tokens: 5
  - name: nousage
    type: fake
    fake:
      seed: 4
      omit_usage_rate: 1
  - name: slow
    type: fake
    fake:
      seed: 5
      ttft: 300ms
models:
  - name: fast
    provider: fast
  - name: strong
    provider: strong
    strongest: true
  - name: flaky
    provider: flaky
  - name: nousage
    provider: nousage
  - name: slow
    provider: slow
`, auth.HashKey(e2eKey))
}

// startGateway runs serve on a loopback listener over fake providers and waits
// for /readyz. stop cancels serve and returns its result.
func startGateway(t *testing.T) (baseURL string, stop func() error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(path, []byte(e2eConfigYAML()), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		errc <- serve(ctx, cfg, ln, func(string) string { return "" }, slog.New(slog.DiscardHandler))
	}()

	var once sync.Once
	var result error
	stop = func() error {
		once.Do(func() {
			cancel()
			select {
			case result = <-errc:
			case <-time.After(5 * time.Second):
				result = fmt.Errorf("serve did not return within 5s of cancel")
			}
		})
		return result
	}
	t.Cleanup(func() { _ = stop() })

	baseURL = "http://" + ln.Addr().String()
	deadline := time.Now().Add(5 * time.Second)
	for {
		resp, err := e2eClient.Get(baseURL + "/readyz")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return baseURL, stop
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("gateway not ready within 5s (last error %v)", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func chatBody(model string, extra string) string {
	return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hello"}]%s}`, model, extra)
}

func post(t *testing.T, base, key, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := e2eClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

type completion struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func complete(t *testing.T, base, body string) (*http.Response, completion) {
	t.Helper()
	resp := post(t, base, e2eKey, body)
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d, body %s", resp.StatusCode, raw)
	}
	var c completion
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	if len(c.Choices) != 1 {
		t.Fatalf("want one choice, got %s", raw)
	}
	return resp, c
}

type sseResult struct {
	frames []string // payloads of every data: frame, [DONE] included
	chunks []map[string]any
	done   bool
	errEv  map[string]any
}

// streamSSE posts a streaming request and reads the body to EOF.
func streamSSE(t *testing.T, base, body string) sseResult {
	t.Helper()
	resp := post(t, base, e2eKey, body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, body %s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("Content-Type %q", ct)
	}
	var res sseResult
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			t.Fatalf("line %q does not start with \"data: \"", line)
		}
		res.frames = append(res.frames, payload)
		if payload == "[DONE]" {
			res.done = true
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(payload), &m); err != nil {
			t.Fatalf("frame %q is not JSON: %v", payload, err)
		}
		if _, isErr := m["error"]; isErr {
			res.errEv = m
			continue
		}
		res.chunks = append(res.chunks, m)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading stream: %v", err)
	}
	return res
}

func deltaContent(chunk map[string]any) (string, bool) {
	choices, _ := chunk["choices"].([]any)
	if len(choices) != 1 {
		return "", false
	}
	delta, _ := choices[0].(map[string]any)["delta"].(map[string]any)
	s, ok := delta["content"].(string)
	return s, ok && s != ""
}

func finishReason(chunk map[string]any) string {
	choices, _ := chunk["choices"].([]any)
	if len(choices) != 1 {
		return ""
	}
	s, _ := choices[0].(map[string]any)["finish_reason"].(string)
	return s
}

func TestEndToEnd(t *testing.T) {
	base, stop := startGateway(t)

	var pinnedContent string

	t.Run("non-streaming pinned", func(t *testing.T) {
		resp, c := complete(t, base, chatBody("fast", ""))
		if got := resp.Header.Get("X-RouteForge-Model"); got != "fast" {
			t.Errorf("X-RouteForge-Model = %q", got)
		}
		if got := resp.Header.Get("X-RouteForge-Reason"); got != "pinned" {
			t.Errorf("X-RouteForge-Reason = %q", got)
		}
		words := strings.Fields(c.Choices[0].Message.Content)
		if len(words) != 16 {
			t.Fatalf("got %d words, want 16: %q", len(words), c.Choices[0].Message.Content)
		}
		for _, w := range words {
			if !slices.Contains(fake.Vocabulary[:], w) {
				t.Errorf("word %q is not in the fake vocabulary", w)
			}
		}
		if c.Usage.CompletionTokens != 16 {
			t.Errorf("completion_tokens = %d, want 16", c.Usage.CompletionTokens)
		}
		pinnedContent = c.Choices[0].Message.Content

		_, again := complete(t, base, chatBody("fast", ""))
		if again.Choices[0].Message.Content != pinnedContent {
			t.Errorf("identical request returned %q, first was %q", again.Choices[0].Message.Content, pinnedContent)
		}
	})

	t.Run("auto routes to strongest", func(t *testing.T) {
		resp, _ := complete(t, base, chatBody("routeforge-auto", ""))
		for h, want := range map[string]string{
			"X-RouteForge-Model":          "strong",
			"X-RouteForge-Reason":         "strongest",
			"X-RouteForge-Policy-Version": "none",
		} {
			if got := resp.Header.Get(h); got != want {
				t.Errorf("%s = %q, want %q", h, got, want)
			}
		}
	})

	t.Run("streaming with usage", func(t *testing.T) {
		_, nonStream := complete(t, base, chatBody("strong", ""))
		res := streamSSE(t, base, chatBody("strong", `,"stream":true,"stream_options":{"include_usage":true}`))
		if res.errEv != nil {
			t.Fatalf("unexpected error event %v", res.errEv)
		}
		if !res.done || res.frames[len(res.frames)-1] != "[DONE]" {
			t.Fatalf("stream does not end with [DONE]: %v", res.frames)
		}
		var content strings.Builder
		contentChunks, stopChunks, usageChunks := 0, 0, 0
		for _, ch := range res.chunks {
			if s, ok := deltaContent(ch); ok {
				content.WriteString(s)
				contentChunks++
			}
			if finishReason(ch) == "stop" {
				stopChunks++
			}
			if u, ok := ch["usage"].(map[string]any); ok {
				usageChunks++
				if choices, _ := ch["choices"].([]any); len(choices) != 0 {
					t.Errorf("usage chunk has choices: %v", ch)
				}
				if got, _ := u["completion_tokens"].(float64); got != 16 {
					t.Errorf("usage completion_tokens = %v, want 16", u["completion_tokens"])
				}
			}
		}
		if contentChunks != 16 {
			t.Errorf("got %d content chunks, want 16", contentChunks)
		}
		if stopChunks != 1 {
			t.Errorf("got %d finish chunks with stop, want 1", stopChunks)
		}
		if usageChunks != 1 {
			t.Errorf("got %d usage chunks, want 1", usageChunks)
		}
		if content.String() != nonStream.Choices[0].Message.Content {
			t.Errorf("streamed %q, non-streaming %q", content.String(), nonStream.Choices[0].Message.Content)
		}
	})

	t.Run("mid-stream failure", func(t *testing.T) {
		res := streamSSE(t, base, chatBody("flaky", `,"stream":true`))
		contentChunks := 0
		for _, ch := range res.chunks {
			if _, ok := deltaContent(ch); ok {
				contentChunks++
			}
		}
		if contentChunks != 2 {
			t.Errorf("got %d content chunks, want 2", contentChunks)
		}
		if res.errEv == nil {
			t.Fatalf("no error event in %v", res.frames)
		}
		errObj, _ := res.errEv["error"].(map[string]any)
		if errObj["code"] != "upstream_stream_interrupted" {
			t.Errorf("error = %v, want code upstream_stream_interrupted", errObj)
		}
		if res.done {
			t.Error("stream carries [DONE] after an interruption")
		}
	})

	t.Run("missing usage", func(t *testing.T) {
		res := streamSSE(t, base, chatBody("nousage", `,"stream":true,"stream_options":{"include_usage":true}`))
		for _, ch := range res.chunks {
			if _, has := ch["usage"]; has {
				t.Errorf("chunk carries usage: %v", ch)
			}
		}
		if res.errEv != nil || !res.done {
			t.Errorf("want a clean [DONE] ending, got done=%v err=%v", res.done, res.errEv)
		}
	})

	t.Run("unauthorized", func(t *testing.T) {
		resp := post(t, base, "", chatBody("fast", ""))
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusUnauthorized || !strings.Contains(string(raw), `"invalid_api_key"`) {
			t.Errorf("status %d, body %s", resp.StatusCode, raw)
		}
	})

	t.Run("unsupported parameter", func(t *testing.T) {
		body := chatBody("fast", `,"tools":[{"type":"function","function":{"name":"f"}}]`)
		resp := post(t, base, e2eKey, body)
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusBadRequest || !strings.Contains(string(raw), `"unsupported_parameter"`) {
			t.Errorf("status %d, body %s", resp.StatusCode, raw)
		}
	})

	t.Run("models", func(t *testing.T) {
		req, err := http.NewRequest(http.MethodGet, base+"/v1/models", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+e2eKey)
		resp, err := e2eClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d", resp.StatusCode)
		}
		var list struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, d := range list.Data {
			ids = append(ids, d.ID)
		}
		want := []string{"routeforge-auto", "fast", "strong", "flaky", "nousage", "slow"}
		if !slices.Equal(ids, want) {
			t.Errorf("ids = %v, want %v", ids, want)
		}
	})

	if err := stop(); err != nil {
		t.Fatal(err)
	}
}

func TestServeGracefulShutdownDrains(t *testing.T) {
	base, stop := startGateway(t)

	type outcome struct {
		status  int
		content string
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(chatBody("slow", "")))
		if err != nil {
			done <- outcome{err: err}
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+e2eKey)
		resp, err := e2eClient.Do(req)
		if err != nil {
			done <- outcome{err: err}
			return
		}
		defer resp.Body.Close()
		var c completion
		if err := json.NewDecoder(resp.Body).Decode(&c); err != nil {
			done <- outcome{status: resp.StatusCode, err: err}
			return
		}
		o := outcome{status: resp.StatusCode}
		if len(c.Choices) == 1 {
			o.content = c.Choices[0].Message.Content
		}
		done <- o
	}()

	time.Sleep(50 * time.Millisecond)
	if err := stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}

	select {
	case o := <-done:
		if o.err != nil || o.status != http.StatusOK || o.content == "" {
			t.Fatalf("in-flight request: %+v", o)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("in-flight request did not finish")
	}

	client := &http.Client{Timeout: time.Second}
	if resp, err := client.Get(base + "/readyz"); err == nil {
		_ = resp.Body.Close()
		t.Fatalf("GET /readyz after stop succeeded with %d, want a connection error", resp.StatusCode)
	}
}
