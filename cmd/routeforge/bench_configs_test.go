package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/bench"
	"github.com/omorsi45/routeforge/internal/config"
)

// startBenchConfig serves a shipped benchmark config, exactly as `routeforge
// serve` builds it, on a loopback listener.
func startBenchConfig(t *testing.T, name string) (baseURL string, cfg *config.Config) {
	t.Helper()
	cfg, err := config.Load(filepath.Join("..", "..", "bench", "configs", name))
	if err != nil {
		t.Fatal(err)
	}
	baseURL, _ = startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))
	return baseURL, cfg
}

// TestBenchConfigsServe starts each shipped benchmark config and sends a request
// the way the benchmark does, which shows that the keys in the files are the
// keys the benchmark presents, and that the router has a model for the request
// at the speeds the file gives.
func TestBenchConfigsServe(t *testing.T) {
	for _, name := range []string{"load-zero.yaml", "load-profiled.yaml", "faults.yaml"} {
		t.Run(name, func(t *testing.T) {
			base, _ := startBenchConfig(t, name)
			c := bench.NewClient(base, "test-key-bench")
			req := bench.ChatRequest{
				Prompt:  "What is 2 plus 2?",
				Headers: map[string]string{"X-RouteForge-Strategy": "quality", "X-RouteForge-Profile": "balanced"},
			}
			if name == "faults.yaml" {
				// The router may send a request to the model that times out; a
				// model without a fault shows that the file serves.
				req.Model = "claude-sonnet-4"
				req.Headers = map[string]string{"X-RouteForge-Deadline-Ms": "2000"}
			}
			out, err := c.Do(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			if out.Status != http.StatusOK || out.DecisionID == "" || out.Model == "" {
				t.Fatalf("outcome = %+v, want a served 200 with a decision", out)
			}
			if _, err := c.Decision(context.Background(), out.DecisionID); err != nil {
				t.Errorf("the decision of the request cannot be read back: %v", err)
			}
			wrong, err := bench.NewClient(base, "test-key-wrong").Chat(context.Background(), "hi", nil)
			if err != nil || wrong.Status != http.StatusUnauthorized {
				t.Errorf("a wrong key got %+v, %v, want 401", wrong, err)
			}
		})
	}
}

func TestBenchConfigDigestsAreOfTheBenchmarkKeys(t *testing.T) {
	cases := map[string][]string{
		"load-zero.yaml":     {"test-key-bench"},
		"load-profiled.yaml": {"test-key-bench"},
		"faults.yaml":        {"test-key-bench", "test-key-bench-over"},
	}
	for name, keys := range cases {
		cfg, err := config.Load(filepath.Join("..", "..", "bench", "configs", name))
		if err != nil {
			t.Fatal(err)
		}
		var digests []string
		for _, tn := range cfg.Tenants {
			digests = append(digests, tn.APIKeySHA256...)
		}
		for _, k := range keys {
			found := false
			for _, d := range digests {
				found = found || d == auth.HashKey(k)
			}
			if !found {
				t.Errorf("%s has no tenant whose digest is that of %s", name, k)
			}
		}
	}
	cfg, err := config.Load(filepath.Join("..", "..", "bench", "configs", "faults.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Admin.APIKeySHA256) != 1 || cfg.Admin.APIKeySHA256[0] != auth.HashKey("test-admin-key") {
		t.Errorf("admin digests = %v, want the digest of test-admin-key", cfg.Admin.APIKeySHA256)
	}
}

func TestLoadConfigsRaiseTheAdmissionLimits(t *testing.T) {
	for _, name := range []string{"load-zero.yaml", "load-profiled.yaml"} {
		cfg, err := config.Load(filepath.Join("..", "..", "bench", "configs", name))
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Server.MaxInFlight != 8192 {
			t.Errorf("%s: max_inflight = %d, want 8192", name, cfg.Server.MaxInFlight)
		}
		for _, m := range cfg.Models {
			if m.MaxConcurrency != 2048 || m.QueueSize != 2048 || m.QueueTimeout != 10*time.Second {
				t.Errorf("%s: model %s has concurrency %d, queue %d, queue timeout %s, want 2048, 2048, 10s",
					name, m.Name, m.MaxConcurrency, m.QueueSize, m.QueueTimeout)
			}
		}
		tn := cfg.Tenants[0]
		if tn.Name != "bench" || tn.RPM != 0 || tn.TPM != 0 || tn.DailyBudgetUSD != 0 || !tn.AllowStrategyOverride {
			t.Errorf("%s: tenant = %+v, want bench with no limits that may override the strategy", name, tn)
		}
	}
}

// inProcessGateway stands for a gateway process in RunLoad: it is the shipped
// config served by this process, so sampling it samples the test.
type inProcessGateway struct{ url string }

func (g inProcessGateway) BaseURL() string { return g.url }
func (g inProcessGateway) PID() int        { return os.Getpid() }
func (g inProcessGateway) Stop() error     { return nil }

// TestLoadBenchmarkOnTheShippedConfigs runs a small load benchmark against
// gateways built from the shipped load configs, over the real routes: the chat
// endpoint, the event stream and the stored decisions. It shows that what the
// benchmark reads, the decision id of a stream, the end of a stream and the
// attempt durations of a decision, is what the gateway writes.
func TestLoadBenchmarkOnTheShippedConfigs(t *testing.T) {
	for _, c := range []struct {
		name, kind  string
		concurrency int
	}{
		{"load-zero.yaml", bench.KindLoadZero, 8},
		{"load-profiled.yaml", bench.KindLoadProfiled, 16},
	} {
		t.Run(c.name, func(t *testing.T) {
			res, err := bench.RunLoad(context.Background(), bench.LoadOptions{
				Kind: c.kind,
				Start: func(context.Context, string) (bench.Gateway, error) {
					base, _ := startBenchConfig(t, c.name)
					return inProcessGateway{url: base}, nil
				},
				Key: "test-key-bench",
				Scenarios: []bench.Scenario{
					{Name: bench.OverheadScenario, Mode: bench.ModeOpen, RPS: 100, Duration: 300 * time.Millisecond},
					{Name: "closed", Mode: bench.ModeClosed, Concurrency: c.concurrency, Requests: 2 * c.concurrency},
				},
				Seed:           1,
				SampleInterval: 50 * time.Millisecond,
			})
			if err != nil {
				t.Fatal(err)
			}
			for name, sc := range res.Scenarios {
				if sc.Completed != sc.Requests || sc.Requests == 0 || len(sc.Errors) != 0 {
					t.Errorf("%s: completed %d of %d, errors %v, want every request served", name, sc.Completed, sc.Requests, sc.Errors)
				}
				if sc.DecisionsFetched != sc.Completed || sc.DecisionsMissing != 0 {
					t.Errorf("%s: read %d decisions (%d missing) for %d requests", name, sc.DecisionsFetched, sc.DecisionsMissing, sc.Completed)
				}
				if sc.OverheadMS["n"] != float64(sc.Completed) || sc.OverheadMS["p50"] < 0 {
					t.Errorf("%s: overhead = %v, want a non-negative sample for every request", name, sc.OverheadMS)
				}
				if sc.DecisionUS["n"] != float64(sc.Completed) || sc.DecisionUS["max"] <= 0 {
					t.Errorf("%s: decision_us = %v, want the routing time of every request", name, sc.DecisionUS)
				}
				if sc.Samples == 0 || sc.RSSBytes["max"] <= 0 {
					t.Errorf("%s: samples = %d, rss = %v (error %q), want ps to read this process", name, sc.Samples, sc.RSSBytes, sc.SamplingError)
				}
			}
			if closed := res.Scenarios["closed"]; closed.StreamRequests != closed.Requests/2 {
				t.Errorf("closed loop streamed %d of %d requests, want half", closed.StreamRequests, closed.Requests)
			}
			if (res.OverheadTarget != nil) != (c.kind == bench.KindLoadProfiled) {
				t.Errorf("overhead target = %+v for %s", res.OverheadTarget, c.kind)
			}
		})
	}
}

// TestFaultsBenchmarkOnTheShippedConfig runs the fault benchmark, with fewer
// bench requests than the real run, against a gateway built from
// bench/configs/faults.yaml. It checks that the faults of the file reach the
// requests and the books, and that the accounting invariants hold: the same
// code that the real run uses, over the same providers and tenants.
func TestFaultsBenchmarkOnTheShippedConfig(t *testing.T) {
	base, cfg := startBenchConfig(t, "faults.yaml")
	setup, err := bench.FaultsSetup(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if setup.OverModel != "gemini-2.5-flash" {
		t.Fatalf("the over tenant is pinned to %q, want gemini-2.5-flash", setup.OverModel)
	}
	opts := bench.FaultsOptions{
		Bench:            bench.NewClient(base, "test-key-bench"),
		Over:             bench.NewClient(base, "test-key-bench-over"),
		Admin:            bench.NewClient(base, "test-admin-key"),
		BenchTenant:      bench.BenchTenant,
		OverTenant:       bench.OverTenant,
		OverModel:        setup.OverModel,
		BenchBudgetNanos: setup.BenchBudgetNanos,
		OverBudgetNanos:  setup.OverBudgetNanos,
		Requests:         300,
		// OverRequests, Concurrency and DeadlineMS are the real run's.
		Seed: 1,
	}
	res, err := bench.RunFaults(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.MarshalIndent(res, "", "  ")
	t.Logf("results:\n%s", data)

	inv := res.Invariants
	if !inv.InvariantsHold {
		t.Errorf("invariants = %+v, want them all to hold", inv)
	}
	b := res.Bench
	if b.Retries == 0 || b.PartialStreams == 0 {
		t.Errorf("retries = %d, partial streams = %d, want the injected faults to show", b.Retries, b.PartialStreams)
	}
	if b.ErrorCodes["deadline_exceeded"] == 0 || b.AttemptErrorKinds["timeout"] == 0 {
		t.Errorf("deadline_exceeded = %d, timeout attempts = %d, want the model that times out to be tried",
			b.ErrorCodes["deadline_exceeded"], b.AttemptErrorKinds["timeout"])
	}
	if strings.Contains(string(data), "test-key") {
		t.Error("the results hold a key")
	}
}
