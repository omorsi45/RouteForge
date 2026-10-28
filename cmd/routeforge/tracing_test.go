package main

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/omorsi45/routeforge/internal/config"
)

// With an OTLP endpoint set, serve exports the spans of the requests it served
// and flushes the last batch when it shuts down, after the HTTP drain.
func TestServeFlushesTracesOnShutdown(t *testing.T) {
	var (
		mu    sync.Mutex
		posts []struct {
			path string
			body []byte
		}
	)
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		posts = append(posts, struct {
			path string
			body []byte
		}{r.URL.Path, body})
		mu.Unlock()
	}))
	t.Cleanup(collector.Close)

	path := filepath.Join(t.TempDir(), "gateway.yaml")
	if err := os.WriteFile(path, []byte(e2eConfigYAML()), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	getenv := func(k string) string {
		if k == "OTEL_EXPORTER_OTLP_ENDPOINT" {
			return collector.URL
		}
		return ""
	}
	base, stop := startGatewayEnv(t, cfg, slog.New(slog.DiscardHandler), getenv)

	resp := post(t, base, e2eKey, chatBody("fast", ""))
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status = %d, want 200", resp.StatusCode)
	}

	// The batch leaves when serve shuts the provider down, not before.
	if err := stop(); err != nil {
		t.Fatalf("serve: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(posts) == 0 {
		t.Fatal("collector saw nothing after shutdown: the last batch was not flushed")
	}
	var all []byte
	for _, p := range posts {
		if p.path != "/v1/traces" {
			t.Errorf("export path = %q, want /v1/traces", p.path)
		}
		all = append(all, p.body...)
	}
	for _, name := range []string{"http.request", "route.decide", "provider.attempt", "routeforge"} {
		if !bytes.Contains(all, []byte(name)) {
			t.Errorf("the exported spans lack %q", name)
		}
	}
}
