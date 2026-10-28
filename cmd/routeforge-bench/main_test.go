package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/auth"
)

const testKey = "test-key-bench"

func runCLI(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errb bytes.Buffer
	code = run(context.Background(), args, &out, &errb)
	return code, out.String(), errb.String()
}

func TestBenchCLIUsage(t *testing.T) {
	code, out, errText := runCLI(t)
	if code != 2 {
		t.Fatalf("exit %d, want 2", code)
	}
	if !strings.Contains(errText, "Usage: routeforge-bench <command>") {
		t.Errorf("stderr %q does not show the usage", errText)
	}
	if out != "" {
		t.Errorf("stdout %q, want empty", out)
	}
}

func TestBenchCLIUnknownCommand(t *testing.T) {
	code, _, errText := runCLI(t, "bogus")
	if code != 2 {
		t.Fatalf("exit %d, want 2", code)
	}
	if !strings.Contains(errText, `unknown command "bogus"`) {
		t.Errorf("stderr %q", errText)
	}
}

func TestBenchCLIHelp(t *testing.T) {
	code, out, _ := runCLI(t, "help")
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if !strings.Contains(out, "replay") {
		t.Errorf("usage %q does not list replay", out)
	}
}

func TestReplayCLIRejectsBadFlags(t *testing.T) {
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"no run id", []string{"replay"}, "--run-id is required"},
		{"run id with a path", []string{"replay", "--run-id", "../x"}, "--run-id"},
		{"run id with a slash", []string{"replay", "--run-id", "a/b"}, "--run-id"},
		{"no workers", []string{"replay", "--run-id", "r", "--workers", "0"}, "--workers"},
		{"unknown flag", []string{"replay", "--bogus"}, "bogus"},
	} {
		t.Run(c.name, func(t *testing.T) {
			code, _, errText := runCLI(t, c.args...)
			if code != 2 {
				t.Fatalf("exit %d, want 2", code)
			}
			if !strings.Contains(errText, c.want) {
				t.Errorf("stderr %q does not mention %q", errText, c.want)
			}
		})
	}
}

// replayRoot builds a repository root with a valid config and two prompts whose
// offline choice for "cheapest" is m-cheap.
func replayRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("bench/configs/replay.yaml", fmt.Sprintf(`tenants:
  - name: bench
    api_key_sha256:
      - %s
providers:
  - name: fake-1
    type: fake
models:
  - name: m-cheap
    provider: fake-1
    input_per_mtok: 1
    output_per_mtok: 2
`, auth.HashKey(testKey)))
	sha := func(i int) string { return fmt.Sprintf("%064x", i) }
	var prompts, decisions, outcomes strings.Builder
	for i := 1; i <= 2; i++ {
		fmt.Fprintf(&prompts, `{"prompt_id":"p%d","prompt":"PROMPT-TEXT-%d","prompt_sha256":%q}`+"\n", i, i, sha(i))
		fmt.Fprintf(&decisions, `{"prompt_id":"p%d","prompt_sha256":%q,"strategy":"cheapest","profile":null,"chosen":"m-cheap","reason":null}`+"\n", i, sha(i))
		for _, model := range []string{"m-cheap", "m-other"} {
			fmt.Fprintf(&outcomes, `{"prompt_sha256":%q,"model":%q,"success":true,"cost_usd":0.01,"text":"RECORDED"}`+"\n", sha(i), model)
		}
	}
	write("trainer/artifacts/replay/prompts.jsonl", prompts.String())
	write("trainer/artifacts/replay/decisions.jsonl", decisions.String())
	write("trainer/artifacts/replay/test.jsonl", outcomes.String())
	return root
}

// gatewayServing returns a stub gateway that serves model for every request.
func gatewayServing(t *testing.T, model func(prompt string) string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &req); err != nil || len(req.Messages) != 1 {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("X-RouteForge-Model", model(req.Messages[0].Content))
		w.Header().Set("X-RouteForge-Decision-Id", "dec_stub")
		_, _ = io.WriteString(w, `{"routeforge":{"decision_id":"dec_stub","cost_usd":0.01}}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestReplayCLIWritesResult(t *testing.T) {
	root := replayRoot(t)
	srv := gatewayServing(t, func(string) string { return "m-cheap" })

	code, out, errText := runCLI(t, "replay", "--base-url", srv.URL, "--key", testKey,
		"--config", "bench/configs/replay.yaml", "--root", root, "--run-id", "smoke", "--workers", "2")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "replay OK: 2 requests, overall agreement 1.0000, parity pass\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}

	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "smoke", "replay.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		SchemaVersion int    `json:"schema_version"`
		Kind          string `json:"kind"`
		RunID         string `json:"run_id"`
		CreatedAt     string `json:"created_at"`
		ConfigPath    string `json:"config_path"`
		Env           struct {
			ConfigSHA256 string `json:"config_sha256"`
			GoVersion    string `json:"go_version"`
		} `json:"env"`
		Params  map[string]any `json:"params"`
		Results struct {
			ParityPass bool `json:"parity_pass"`
			Requests   int  `json:"requests"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.SchemaVersion != 1 || doc.Kind != "replay" || doc.RunID != "smoke" || doc.ConfigPath != "bench/configs/replay.yaml" {
		t.Errorf("header = %+v", doc)
	}
	if doc.CreatedAt == "" || doc.Env.ConfigSHA256 == "" || doc.Env.GoVersion == "" {
		t.Errorf("created_at or env not filled: %+v", doc)
	}
	if !doc.Results.ParityPass || doc.Results.Requests != 2 {
		t.Errorf("results = %+v", doc.Results)
	}
	if doc.Params["workers"] != float64(2) || doc.Params["base_url"] != srv.URL {
		t.Errorf("params = %v", doc.Params)
	}
	if _, ok := doc.Params["key"]; ok {
		t.Error("params record the API key")
	}
	if strings.Contains(string(data), testKey) || strings.Contains(string(data), "PROMPT-TEXT") {
		t.Errorf("result file contains a key or a prompt:\n%s", data)
	}
}

func TestReplayCLIExitsZeroWhenParityFails(t *testing.T) {
	root := replayRoot(t)
	srv := gatewayServing(t, func(string) string { return "m-other" })

	code, out, errText := runCLI(t, "replay", "--base-url", srv.URL, "--key", testKey,
		"--root", root, "--run-id", "mismatch", "--workers", "1")
	if code != 0 {
		t.Fatalf("exit %d, want 0 because the result records the failed parity; stderr %q", code, errText)
	}
	if want := "replay OK: 2 requests, overall agreement 0.0000, parity fail\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "mismatch", "replay.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"parity_pass": false`) {
		t.Errorf("result does not record the failed parity:\n%s", data)
	}
}

func TestReplayCLIExitsOneOnErrors(t *testing.T) {
	root := replayRoot(t)
	srv := gatewayServing(t, func(string) string { return "m-cheap" })
	url := srv.URL
	srv.Close()

	noFiles := replayRoot(t)
	if err := os.RemoveAll(filepath.Join(noFiles, "trainer")); err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name string
		args []string
	}{
		{"gateway unreachable", []string{"--base-url", url, "--root", root}},
		{"config missing", []string{"--base-url", url, "--root", root, "--config", "bench/configs/none.yaml"}},
		{"no replay files", []string{"--base-url", url, "--root", noFiles}},
	} {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"replay", "--run-id", "bad", "--workers", "1"}, c.args...)
			code, out, errText := runCLI(t, args...)
			if code != 1 {
				t.Fatalf("exit %d, want 1", code)
			}
			if out != "" || !strings.HasPrefix(errText, "replay: ") {
				t.Errorf("stdout %q, stderr %q", out, errText)
			}
		})
	}
	if _, err := os.Stat(filepath.Join(root, "bench", "results", "bad", "replay.json")); err == nil {
		t.Error("a failed run left a result file")
	}
}
