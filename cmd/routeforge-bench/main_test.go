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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/bench"
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
	return replayRootFor(t, "cheapest")
}

// replayRootFor is replayRoot with an offline decision of m-cheap for every
// listed strategy.
func replayRootFor(t *testing.T, strategies ...string) string {
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
		for _, strategy := range strategies {
			fmt.Fprintf(&decisions, `{"prompt_id":"p%d","prompt_sha256":%q,"strategy":%q,"profile":null,"chosen":"m-cheap","reason":null}`+"\n", i, sha(i), strategy)
		}
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
	return gatewayServingBy(t, func(_, prompt string) string { return model(prompt) })
}

// gatewayServingBy returns a stub gateway that serves model(strategy, prompt),
// where strategy is the one the request asked for.
func gatewayServingBy(t *testing.T, model func(strategy, prompt string) string) *httptest.Server {
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
		w.Header().Set("X-RouteForge-Model", model(r.Header.Get("X-RouteForge-Strategy"), req.Messages[0].Content))
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
	if want := "replay OK: 2 requests, deterministic agreement 1.0000, overall agreement 1.0000, parity pass\n"; out != want {
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
	if want := "replay OK: 2 requests, deterministic agreement 0.0000, overall agreement 0.0000, parity fail\n"; out != want {
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

func TestReplayCLIParityIgnoresTheLatencyStrategy(t *testing.T) {
	root := replayRootFor(t, "cheapest", "latency")
	srv := gatewayServingBy(t, func(strategy, _ string) string {
		if strategy == "latency" {
			return "m-other" // not the offline choice, as a gateway that adapts its estimates may serve
		}
		return "m-cheap"
	})

	code, out, errText := runCLI(t, "replay", "--base-url", srv.URL, "--key", testKey,
		"--root", root, "--run-id", "latency", "--workers", "1")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "replay OK: 4 requests, deterministic agreement 1.0000, overall agreement 0.5000, parity pass\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "latency", "replay.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"parity_pass": true`, `"deterministic_agreement": 1`, `"overall_agreement": 0.5`, `"latency_note": "The gateway's latency strategy`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("result lacks %s:\n%s", want, data)
		}
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

// cliStub is a gateway for the load and fault commands: it serves chat
// requests, stored decisions and the accounting books.
type cliStub struct {
	*httptest.Server
	next atomic.Int64
	// books are the tenants' accounting, by tenant name.
	books map[string]map[string]int64
}

func newCLIStub(t *testing.T) *cliStub {
	t.Helper()
	s := &cliStub{books: map[string]map[string]int64{
		"bench":      {"spent_nanos": 7000, "usage_cost_nanos": 7000},
		"bench-over": {"spent_nanos": 3500000, "usage_cost_nanos": 3500000, "overshoot_nanos": 500000},
	}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream bool `json:"stream"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)
		w.Header().Set("X-RouteForge-Decision-Id", fmt.Sprintf("dec_%d", s.next.Add(1)))
		w.Header().Set("X-RouteForge-Model", "m1")
		if req.Stream {
			_, _ = io.WriteString(w, "data: {\"id\":\"c\"}\n\ndata: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(w, `{"routeforge":{"decision_id":"x"}}`)
	})
	mux.HandleFunc("GET /v1/routing/decisions/{id}", func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprintf(w, `{"id":%q,"chosen":"m1","decided_in_us":120,"attempts":[{"model":"m1","outcome":"ok","duration_ms":0.5}]}`, r.PathValue("id"))
	})
	mux.HandleFunc("GET /admin/accounting", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-admin-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		tenant := r.URL.Query().Get("tenant")
		out := map[string]any{"tenant": tenant, "day": r.URL.Query().Get("day")}
		for k, v := range s.books[tenant] {
			out[k] = v
		}
		_ = json.NewEncoder(w).Encode(out)
	})
	s.Server = httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return s
}

// fakeGateway stands for a gateway process; stopped records the stops.
type fakeGateway struct {
	url     string
	stopped *atomic.Int32
}

func (g fakeGateway) BaseURL() string { return g.url }
func (g fakeGateway) PID() int        { return os.Getpid() }
func (g fakeGateway) Stop() error     { g.stopped.Add(1); return nil }

// benchRoot is a repository root with the config file bench/configs/<name>.
func benchRoot(t *testing.T, name, configYAML string) string {
	t.Helper()
	root := t.TempDir()
	path := filepath.Join(root, "bench", "configs", name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(configYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func tenantConfig(extra string) string {
	return fmt.Sprintf(`tenants:
  - name: bench
    api_key_sha256:
      - %s
%sproviders:
  - name: fake-1
    type: fake
models:
  - name: m1
    provider: fake-1
    input_per_mtok: 1
    output_per_mtok: 2
  - name: m-over
    provider: fake-1
    input_per_mtok: 1
    output_per_mtok: 2
`, auth.HashKey(testKey), extra)
}

func TestLoadCLIValidatesFlags(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "routeforge")
	if err := os.WriteFile(bin, []byte("not a real gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"no run id", []string{"load", "--gateway-bin", bin, "--config", "bench/configs/load-zero.yaml"}, "--run-id is required"},
		{"bad run id", []string{"load", "--run-id", "a/b", "--gateway-bin", bin, "--config", "bench/configs/load-zero.yaml"}, "--run-id"},
		{"no config", []string{"load", "--run-id", "r", "--gateway-bin", bin}, "--config is required"},
		{"config is not a load config", []string{"load", "--run-id", "r", "--gateway-bin", bin, "--config", "bench/configs/replay.yaml"}, "load-"},
		{"config is not yaml", []string{"load", "--run-id", "r", "--gateway-bin", bin, "--config", "bench/configs/load-zero.json"}, "load-"},
		{"gateway binary missing", []string{"load", "--run-id", "r", "--gateway-bin", bin + ".none", "--config", "bench/configs/load-zero.yaml"}, "--gateway-bin"},
		{"gateway binary is a directory", []string{"load", "--run-id", "r", "--gateway-bin", filepath.Dir(bin), "--config", "bench/configs/load-zero.yaml"}, "--gateway-bin"},
		{"unknown flag", []string{"load", "--bogus"}, "bogus"},
	} {
		t.Run(c.name, func(t *testing.T) {
			code, _, errText := runCLI(t, c.args...)
			if code != 2 {
				t.Fatalf("exit %d, want 2; stderr %q", code, errText)
			}
			if !strings.Contains(errText, c.want) {
				t.Errorf("stderr %q does not mention %q", errText, c.want)
			}
		})
	}
}

// swapLoadSeams replaces how the load command starts gateways and which
// scenarios it runs, and restores them when the test ends.
func swapLoadSeams(t *testing.T, start func(ctx context.Context, o bench.ProcessOptions) (bench.Gateway, error), scenarios []bench.Scenario) {
	t.Helper()
	oldStart, oldScenarios := startGatewayProcess, loadScenarios
	startGatewayProcess = start
	loadScenarios = func() []bench.Scenario { return scenarios }
	t.Cleanup(func() { startGatewayProcess, loadScenarios = oldStart, oldScenarios })
}

func TestLoadCLIWritesResult(t *testing.T) {
	stub := newCLIStub(t)
	root := benchRoot(t, "load-profiled.yaml", tenantConfig(""))
	bin := filepath.Join(t.TempDir(), "routeforge")
	if err := os.WriteFile(bin, nil, 0o755); err != nil {
		t.Fatal(err)
	}

	var stopped atomic.Int32
	var started []bench.ProcessOptions
	swapLoadSeams(t, func(_ context.Context, o bench.ProcessOptions) (bench.Gateway, error) {
		started = append(started, o)
		return fakeGateway{url: stub.URL, stopped: &stopped}, nil
	}, []bench.Scenario{
		{Name: "open_loop_200rps", Mode: bench.ModeOpen, RPS: 200, Duration: 200 * time.Millisecond},
		{Name: "closed_loop_c2", Mode: bench.ModeClosed, Concurrency: 2, Requests: 10},
	})

	code, out, errText := runCLI(t, "load", "--gateway-bin", bin, "--config", "bench/configs/load-profiled.yaml",
		"--root", root, "--run-id", "smoke")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 2 || !strings.HasPrefix(lines[0], "load OK: load-profiled, 2 scenarios, ") || !strings.HasPrefix(lines[1], "overhead target: p95 ") {
		t.Errorf("stdout %q, want the summary line and the overhead target line", out)
	}

	if len(started) != 2 || stopped.Load() != 2 {
		t.Fatalf("started %d gateways and stopped %d, want 2 and 2 (one per scenario)", len(started), stopped.Load())
	}
	absBin, _ := filepath.Abs(bin)
	for i, o := range started {
		wantLog := filepath.Join(root, "bench", "results", "smoke", "raw", "gateway-load-profiled-"+[]string{"open_loop_200rps", "closed_loop_c2"}[i]+".log")
		if o.Binary != absBin || o.Addr != "127.0.0.1:18080" || o.LogPath != wantLog || !filepath.IsAbs(o.ConfigPath) ||
			filepath.Base(o.ConfigPath) != "load-profiled.yaml" {
			t.Errorf("gateway %d options = %+v, want binary %s, port 18080 and log %s", i, o, absBin, wantLog)
		}
	}

	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "smoke", "load-profiled.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Kind    string `json:"kind"`
		Results struct {
			Scenarios map[string]struct {
				Requests  int `json:"requests"`
				Completed int `json:"completed"`
			} `json:"scenarios"`
			OverheadTarget struct {
				Scenario string  `json:"scenario"`
				Below    float64 `json:"p95_ms_below"`
			} `json:"overhead_target"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Kind != "load-profiled" || len(doc.Results.Scenarios) != 2 || doc.Results.Scenarios["closed_loop_c2"].Completed != 10 {
		t.Errorf("result = %+v", doc)
	}
	if doc.Results.OverheadTarget.Scenario != "profiled open loop 200 rps" || doc.Results.OverheadTarget.Below != 20 {
		t.Errorf("overhead target = %+v", doc.Results.OverheadTarget)
	}
	if strings.Contains(string(data), testKey) {
		t.Error("the result file holds the API key")
	}
}

func TestLoadCLIZeroLatencyHasNoTargetLine(t *testing.T) {
	stub := newCLIStub(t)
	root := benchRoot(t, "load-zero.yaml", tenantConfig(""))
	bin := filepath.Join(t.TempDir(), "routeforge")
	if err := os.WriteFile(bin, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	var stopped atomic.Int32
	swapLoadSeams(t, func(context.Context, bench.ProcessOptions) (bench.Gateway, error) {
		return fakeGateway{url: stub.URL, stopped: &stopped}, nil
	}, []bench.Scenario{{Name: "closed_loop_c1", Mode: bench.ModeClosed, Concurrency: 1, Requests: 4}})

	code, out, errText := runCLI(t, "load", "--gateway-bin", bin, "--config", "bench/configs/load-zero.yaml", "--root", root, "--run-id", "z")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "load OK: load-zero, 1 scenarios, 4 requests\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	if _, err := os.Stat(filepath.Join(root, "bench", "results", "z", "load-zero.json")); err != nil {
		t.Errorf("no load-zero.json: %v", err)
	}
}

func TestLoadCLIExitsOneWhenAGatewayCannotStart(t *testing.T) {
	root := benchRoot(t, "load-zero.yaml", tenantConfig(""))
	bin := filepath.Join(t.TempDir(), "routeforge")
	if err := os.WriteFile(bin, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	swapLoadSeams(t, func(context.Context, bench.ProcessOptions) (bench.Gateway, error) {
		return nil, fmt.Errorf("port taken")
	}, []bench.Scenario{{Name: "closed_loop_c1", Mode: bench.ModeClosed, Concurrency: 1, Requests: 1}})

	code, out, errText := runCLI(t, "load", "--gateway-bin", bin, "--config", "bench/configs/load-zero.yaml", "--root", root, "--run-id", "f")
	if code != 1 || out != "" || !strings.Contains(errText, "port taken") || !strings.HasPrefix(errText, "load: ") {
		t.Errorf("exit %d, stdout %q, stderr %q, want exit 1 and the reason on stderr", code, out, errText)
	}
	if _, err := os.Stat(filepath.Join(root, "bench", "results", "f", "load-zero.json")); err == nil {
		t.Error("a failed run left a result file")
	}
}

func TestFaultsCLIRejectsBadFlags(t *testing.T) {
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"no run id", []string{"faults"}, "--run-id is required"},
		{"bad run id", []string{"faults", "--run-id", ".."}, "--run-id"},
		{"no admin key", []string{"faults", "--run-id", "r", "--admin-key", ""}, "--admin-key"},
		{"no over key", []string{"faults", "--run-id", "r", "--over-key", ""}, "--over-key"},
		{"unknown flag", []string{"faults", "--bogus"}, "bogus"},
	} {
		t.Run(c.name, func(t *testing.T) {
			code, _, errText := runCLI(t, c.args...)
			if code != 2 {
				t.Fatalf("exit %d, want 2; stderr %q", code, errText)
			}
			if !strings.Contains(errText, c.want) {
				t.Errorf("stderr %q does not mention %q", errText, c.want)
			}
		})
	}
}

// faultsTenants finishes the bench tenant of tenantConfig with a budget and adds
// the over tenant: a small budget and one allowed model.
func faultsTenants() string {
	return fmt.Sprintf(`    daily_budget_usd: 100
  - name: bench-over
    api_key_sha256:
      - %s
    daily_budget_usd: 0.003
    allowed_models: [m-over]
`, auth.HashKey("test-key-bench-over"))
}

func TestFaultsCLIWritesResult(t *testing.T) {
	stub := newCLIStub(t)
	root := benchRoot(t, "faults.yaml", tenantConfig(faultsTenants()))

	code, out, errText := runCLI(t, "faults", "--base-url", stub.URL, "--root", root, "--run-id", "smoke")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "faults OK: 2200 requests, success rate 1.0000, invariants hold\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}

	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "smoke", "faults.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Kind   string         `json:"kind"`
		Params map[string]any `json:"params"`
		Result struct {
			Bench      struct{ Requests int } `json:"bench"`
			BenchOver  struct{ Requests int } `json:"bench_over"`
			Invariants struct {
				Hold bool `json:"invariants_hold"`
			} `json:"invariants"`
			Accounting map[string]struct {
				BudgetNanos int64 `json:"budget_nanos"`
			} `json:"accounting"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Kind != "faults" || doc.Result.Bench.Requests != 2000 || doc.Result.BenchOver.Requests != 200 || !doc.Result.Invariants.Hold {
		t.Errorf("result = %+v", doc)
	}
	if doc.Result.Accounting["bench"].BudgetNanos != 100_000_000_000 || doc.Result.Accounting["bench-over"].BudgetNanos != 3_000_000 {
		t.Errorf("budgets = %+v, want the config's 100 USD and 0.003 USD in nano-USD", doc.Result.Accounting)
	}
	if doc.Params["over_model"] != "m-over" || doc.Params["concurrency"] != float64(32) || doc.Params["deadline_ms"] != float64(2000) {
		t.Errorf("params = %v", doc.Params)
	}
	for _, secret := range []string{testKey, "test-key-bench-over", "test-admin-key"} {
		if strings.Contains(string(data), secret) {
			t.Errorf("the result file holds the key %q", secret)
		}
	}
}

func TestFaultsCLIExitsZeroWhenInvariantsAreViolated(t *testing.T) {
	stub := newCLIStub(t)
	stub.books["bench"]["reserved_nanos"] = 99
	root := benchRoot(t, "faults.yaml", tenantConfig(faultsTenants()))

	code, out, errText := runCLI(t, "faults", "--base-url", stub.URL, "--root", root, "--run-id", "bad")
	if code != 0 {
		t.Fatalf("exit %d, want 0 because the result records the violation; stderr %q", code, errText)
	}
	if want := "faults OK: 2200 requests, success rate 1.0000, invariants violated\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	data, err := os.ReadFile(filepath.Join(root, "bench", "results", "bad", "faults.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"invariants_hold": false`) {
		t.Errorf("the result does not record the violation:\n%s", data)
	}
}

func TestFaultsCLIExitsOneOnErrors(t *testing.T) {
	stub := newCLIStub(t)
	url := stub.URL
	root := benchRoot(t, "faults.yaml", tenantConfig(faultsTenants()))
	noBudgets := benchRoot(t, "faults.yaml", tenantConfig("    rpm: 5\n"))

	for _, c := range []struct {
		name string
		args []string
	}{
		{"config missing", []string{"--base-url", url, "--root", root, "--config", "bench/configs/none.yaml"}},
		{"config has no budgets", []string{"--base-url", url, "--root", noBudgets}},
		{"wrong admin key", []string{"--base-url", url, "--root", root, "--admin-key", "wrong"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"faults", "--run-id", "bad"}, c.args...)
			code, out, errText := runCLI(t, args...)
			if code != 1 || out != "" || !strings.HasPrefix(errText, "faults: ") {
				t.Errorf("exit %d, stdout %q, stderr %q, want exit 1 and the reason on stderr", code, out, errText)
			}
		})
	}
}

// reportRoot is a repository root that holds the golden inputs of the report
// renderer (synthetic files of internal/bench/testdata/report) where the report
// command reads them: the results of run golden-run and the trainer reports.
func reportRoot(t *testing.T) string {
	t.Helper()
	src := filepath.Join("..", "..", "internal", "bench", "testdata", "report")
	root := t.TempDir()
	for name, dest := range map[string]string{
		"replay.json":        "bench/results/golden-run/replay.json",
		"load-zero.json":     "bench/results/golden-run/load-zero.json",
		"load-profiled.json": "bench/results/golden-run/load-profiled.json",
		"faults.json":        "bench/results/golden-run/faults.json",
		"evaluation.json":    "trainer/artifacts/reports/evaluation.json",
		"training.json":      "trainer/artifacts/reports/training.json",
	} {
		data, err := os.ReadFile(filepath.Join(src, name))
		if err != nil {
			t.Fatal(err)
		}
		writeUnder(t, root, dest, string(data))
	}
	return root
}

func writeUnder(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func readUnder(t *testing.T, root, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func goldenReport(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "internal", "bench", "testdata", "report", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

const (
	readmeBefore = "# Project\n\nIntroduction.\n\n<!-- results:begin -->\n"
	readmeAfter  = "<!-- results:end -->\n\nTail text.\n"
)

func TestReportCLIWritesBothFiles(t *testing.T) {
	root := reportRoot(t)
	writeUnder(t, root, "README.md", readmeBefore+"stale results\n"+readmeAfter)

	code, out, errText := runCLI(t, "report", "--root", root, "--run-id", "golden-run")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "report OK: wrote docs/benchmarks.md and README.md for run golden-run\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	if got, want := readUnder(t, root, "docs/benchmarks.md"), goldenReport(t, "benchmarks.md"); got != want {
		t.Errorf("docs/benchmarks.md is not the golden file")
	}
	if got, want := readUnder(t, root, "README.md"), readmeBefore+goldenReport(t, "readme-results.md")+readmeAfter; got != want {
		t.Errorf("README.md:\n%s\nwant:\n%s", got, want)
	}

	// A second run changes nothing.
	if code, _, errText := runCLI(t, "report", "--root", root, "--run-id", "golden-run"); code != 0 {
		t.Fatalf("second run: exit %d, stderr %q", code, errText)
	}
	if got, want := readUnder(t, root, "README.md"), readmeBefore+goldenReport(t, "readme-results.md")+readmeAfter; got != want {
		t.Errorf("README.md changed on the second run:\n%s", got)
	}
}

func TestReportCLIWritesWhereItIsTold(t *testing.T) {
	root := reportRoot(t)
	writeUnder(t, root, "site/index.md", readmeBefore+readmeAfter)

	code, out, errText := runCLI(t, "report", "--root", root, "--run-id", "golden-run", "--out", "site/results.md", "--readme", "site/index.md")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errText)
	}
	if want := "report OK: wrote site/results.md and site/index.md for run golden-run\n"; out != want {
		t.Errorf("stdout %q, want %q", out, want)
	}
	if got := readUnder(t, root, "site/results.md"); got != goldenReport(t, "benchmarks.md") {
		t.Error("site/results.md is not the golden file")
	}
	if _, err := os.Stat(filepath.Join(root, "docs", "benchmarks.md")); err == nil {
		t.Error("the default output was written as well")
	}
}

func TestReportCLIRejectsBadFlags(t *testing.T) {
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"no run id", []string{"report"}, "--run-id is required"},
		{"run id with a path", []string{"report", "--run-id", "../x"}, "--run-id"},
		{"no output", []string{"report", "--run-id", "r", "--out", ""}, "--out"},
		{"no README", []string{"report", "--run-id", "r", "--readme", ""}, "--readme"},
		{"unknown flag", []string{"report", "--bogus"}, "bogus"},
	} {
		t.Run(c.name, func(t *testing.T) {
			code, _, errText := runCLI(t, c.args...)
			if code != 2 {
				t.Fatalf("exit %d, want 2; stderr %q", code, errText)
			}
			if !strings.Contains(errText, c.want) {
				t.Errorf("stderr %q does not mention %q", errText, c.want)
			}
		})
	}
}

func TestReportCLIWritesNothingOnAnError(t *testing.T) {
	for _, c := range []struct {
		name   string
		runID  string
		damage func(t *testing.T, root string)
		want   string
	}{
		{"a result file is missing", "golden-run", func(t *testing.T, root string) {
			if err := os.Remove(filepath.Join(root, "bench", "results", "golden-run", "faults.json")); err != nil {
				t.Fatal(err)
			}
		}, "faults.json"},
		{"a result file is of another run", "other-run", func(t *testing.T, root string) {
			if err := os.Rename(filepath.Join(root, "bench", "results", "golden-run"), filepath.Join(root, "bench", "results", "other-run")); err != nil {
				t.Fatal(err)
			}
		}, "golden-run"},
		{"the README has no markers", "golden-run", func(t *testing.T, root string) {
			writeUnder(t, root, "README.md", "# Project\n")
		}, "results:begin"},
		{"the README is missing", "golden-run", func(t *testing.T, root string) {
			if err := os.Remove(filepath.Join(root, "README.md")); err != nil {
				t.Fatal(err)
			}
		}, "README.md"},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := reportRoot(t)
			writeUnder(t, root, "README.md", readmeBefore+"stale results\n"+readmeAfter)
			c.damage(t, root)
			readmeBeforeRun, readmeErr := os.ReadFile(filepath.Join(root, "README.md"))

			code, out, errText := runCLI(t, "report", "--root", root, "--run-id", c.runID)
			if code != 1 || out != "" || !strings.HasPrefix(errText, "report: ") || !strings.Contains(errText, c.want) {
				t.Errorf("exit %d, stdout %q, stderr %q, want exit 1 and a reason that mentions %q", code, out, errText, c.want)
			}
			if _, err := os.Stat(filepath.Join(root, "docs", "benchmarks.md")); err == nil {
				t.Error("docs/benchmarks.md was written although the command failed")
			}
			if readmeErr == nil {
				if now, _ := os.ReadFile(filepath.Join(root, "README.md")); string(now) != string(readmeBeforeRun) {
					t.Errorf("README.md changed although the command failed:\n%s", now)
				}
			}
		})
	}
}

func TestBenchCLIHelpListsReport(t *testing.T) {
	_, out, _ := runCLI(t, "help")
	if !strings.Contains(out, "report") {
		t.Errorf("usage %q does not list report", out)
	}
}

func TestBenchCLIHelpListsDemo(t *testing.T) {
	_, out, _ := runCLI(t, "help")
	if !strings.Contains(out, "demo") {
		t.Errorf("usage %q does not list demo", out)
	}
}

func TestDemoCLIRejectsBadFlags(t *testing.T) {
	for _, c := range []struct {
		name string
		args []string
		want string
	}{
		{"empty key", []string{"demo", "--key", ""}, "--key"},
		{"empty out", []string{"demo", "--out", ""}, "--out"},
		{"unknown flag", []string{"demo", "--bogus"}, "bogus"},
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

const demoCLIEvaluation = "../../internal/bench/testdata/report/evaluation.json"

// demoCLIRoot builds a repository root with one demo prompt and, when withEval
// is set, the synthetic evaluation of the report tests.
func demoCLIRoot(t *testing.T, withEval bool) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel string, content []byte) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("examples/prompts.json", []byte(`[{"name":"only","prompt":"the only demo prompt"}]`))
	if withEval {
		eval, err := os.ReadFile(demoCLIEvaluation)
		if err != nil {
			t.Fatal(err)
		}
		write("trainer/artifacts/reports/evaluation.json", eval)
	}
	return root
}

func TestDemoCLIMissingInputs(t *testing.T) {
	t.Run("no evaluation", func(t *testing.T) {
		root := demoCLIRoot(t, false)
		code, out, errText := runCLI(t, "demo", "--root", root)
		if code != 1 || out != "" || !strings.Contains(errText, "trainer/artifacts/reports/evaluation.json") {
			t.Errorf("exit %d, stdout %q, stderr %q, want exit 1 naming the evaluation file", code, out, errText)
		}
	})
	t.Run("no prompts", func(t *testing.T) {
		root := demoCLIRoot(t, true)
		if err := os.Remove(filepath.Join(root, "examples", "prompts.json")); err != nil {
			t.Fatal(err)
		}
		code, out, errText := runCLI(t, "demo", "--root", root)
		if code != 1 || out != "" || !strings.Contains(errText, "prompts.json") {
			t.Errorf("exit %d, stdout %q, stderr %q, want exit 1 naming the prompts file", code, out, errText)
		}
	})
}

func TestDemoCLIWritesOutput(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.URL.Path+" "+r.Header.Get("X-RouteForge-Profile"))
		mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer "+testKey {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		switch r.URL.Path {
		case "/v1/routing/explain":
			fmt.Fprintf(w, `{"policy_version":"pol-test-1","profile":%q,"chosen":"model-b","reason":"cheapest_passing","lambda":0.7,`+
				`"candidates":[{"model":"model-b","p":0.83,"lcb":0.7431,"block_n":100,"block_k":83,"expected_cost_usd":0.00025}]}`,
				r.Header.Get("X-RouteForge-Profile"))
		case "/v1/chat/completions":
			w.Header().Set("X-RouteForge-Model", "model-b")
			w.Header().Set("X-RouteForge-Decision-Id", "dec_cli")
			fmt.Fprint(w, `{"routeforge":{"decision_id":"dec_cli","cost_usd":0.00031}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	root := demoCLIRoot(t, true)
	code, out, errText := runCLI(t, "demo", "--base-url", srv.URL, "--key", testKey, "--root", root, "--out", "examples/demo-output.md")
	if code != 0 || errText != "" || !strings.HasPrefix(out, "demo OK: ") || !strings.Contains(out, "examples/demo-output.md") {
		t.Fatalf("exit %d, stdout %q, stderr %q, want exit 0 and a demo OK line naming the file", code, out, errText)
	}
	doc, err := os.ReadFile(filepath.Join(root, "examples", "demo-output.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"# Routing demo\n\n" + bench.DemoNotice,
		"| only | balanced | model-b | cheapest_passing | 0.700 | 0.830 | 0.743 | 83 of 100 | $0.000250 |",
		"- `only`: on the calibration set, 83 of 100 requests scored like this succeeded on model-b; 95% lower bound 0.74",
		"H1: falsified (static_oracle_family)",
	} {
		if !strings.Contains(string(doc), want) {
			t.Errorf("examples/demo-output.md does not contain %q", want)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 4 {
		t.Errorf("the gateway saw %d requests %q, want 3 explain requests and 1 chat request", len(seen), seen)
	}
}
