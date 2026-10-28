package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/anthropic"
	"github.com/omorsi45/routeforge/internal/provider/fake"
	"github.com/omorsi45/routeforge/internal/provider/openai"
	"github.com/omorsi45/routeforge/internal/provider/replay"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
)

const wireConfigYAML = `
tenants:
  - name: tenant-a
    api_key_sha256:
      - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
providers:
  - name: oa
    type: openai
    base_url: http://127.0.0.1:1/v1
  - name: anthropic
    type: anthropic
    base_url: http://127.0.0.1:1
    api_key_env: RF_TEST_ANTHROPIC_KEY
  - name: fk
    type: fake
models:
  - name: m-oa
    provider: oa
  - name: m-an
    provider: anthropic
  - name: m-fk
    provider: fk
`

func parseWireConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(wireConfigYAML))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestBuildProvidersMissingEnv(t *testing.T) {
	cfg := parseWireConfig(t)
	_, err := buildProviders(cfg, func(string) string { return "" }, http.DefaultClient)
	if err == nil {
		t.Fatal("want an error for an unset api_key_env variable")
	}
	for _, want := range []string{`provider "anthropic"`, "RF_TEST_ANTHROPIC_KEY is not set"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not contain %q", err, want)
		}
	}
}

func TestBuildProvidersOK(t *testing.T) {
	cfg := parseWireConfig(t)
	const key = "test-key-bbbb"
	getenv := func(name string) string {
		if name == "RF_TEST_ANTHROPIC_KEY" {
			return key
		}
		return ""
	}
	providers, err := buildProviders(cfg, getenv, http.DefaultClient)
	if err != nil {
		t.Fatal(err)
	}
	if len(providers) != 3 {
		t.Fatalf("got %d providers, want 3", len(providers))
	}
	if _, ok := providers["oa"].(*openai.Provider); !ok {
		t.Errorf("oa is %T, want *openai.Provider", providers["oa"])
	}
	if _, ok := providers["anthropic"].(*anthropic.Provider); !ok {
		t.Errorf("anthropic is %T, want *anthropic.Provider", providers["anthropic"])
	}
	if _, ok := providers["fk"].(*fake.Provider); !ok {
		t.Errorf("fk is %T, want *fake.Provider", providers["fk"])
	}
}

func TestBuildProvidersOpenAIMissingEnv(t *testing.T) {
	cfg := parseWireConfig(t)
	cfg.Providers[0].APIKeyEnv = "RF_TEST_OPENAI_KEY"
	_, err := buildProviders(cfg, func(string) string { return "" }, http.DefaultClient)
	if err == nil || !strings.Contains(err.Error(), `provider "oa": environment variable RF_TEST_OPENAI_KEY is not set`) {
		t.Fatalf("got %v", err)
	}
}

func TestBuildTargets(t *testing.T) {
	cfg, err := config.Load("../../internal/config/testdata/minimal.yaml")
	if err != nil {
		t.Fatal(err)
	}
	providers := map[string]provider.Provider{"fake": fake.New("fake", cfg.Providers[0].Fake)}
	targets := buildTargets(cfg, providers)
	if len(targets) != 2 {
		t.Fatalf("got %d targets, want 2", len(targets))
	}
	for _, name := range []string{"m-fast", "m-strong"} {
		tg, ok := targets[name]
		if !ok {
			t.Fatalf("missing target %q", name)
		}
		if tg.Model != name || tg.Upstream != name {
			t.Errorf("target %q = %+v, want Model and Upstream %q", name, tg, name)
		}
		if tg.Provider != providers["fake"] {
			t.Errorf("target %q has the wrong provider", name)
		}
	}
}

func TestServeMissingEnvClosesListener(t *testing.T) {
	cfg := parseWireConfig(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	logger := slog.New(slog.DiscardHandler)
	err = serve(context.Background(), cfg, ln, func(string) string { return "" }, logger)
	if err == nil || !strings.Contains(err.Error(), "RF_TEST_ANTHROPIC_KEY is not set") {
		t.Fatalf("serve = %v, want the missing env error", err)
	}
	if _, err := ln.Accept(); !errors.Is(err, net.ErrClosed) {
		t.Fatalf("Accept after failed serve = %v, want net.ErrClosed", err)
	}
}

const priceConfigYAML = `
tenants:
  - name: tenant-a
    api_key_sha256:
      - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
providers:
  - name: fk
    type: fake
models:
  - name: gpt-5
    provider: fk
  - name: renamed
    provider: fk
    policy_name: qwen3-235b-a22b-2507
  - name: override
    provider: fk
    policy_name: claude-sonnet-4
    input_per_mtok: 2.0
    output_per_mtok: 4.5
  - name: local
    provider: fk
    policy_name: not-in-the-price-file
    input_per_mtok: 0
    output_per_mtok: 0
`

func parsePriceConfig(t *testing.T, pricingFile string) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(priceConfigYAML))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Pricing.File = pricingFile
	return cfg
}

func TestModelPrices(t *testing.T) {
	cfg := parsePriceConfig(t, "../../deploy/prices.yaml")
	prices, err := modelPrices(cfg)
	if err != nil {
		t.Fatalf("modelPrices: %v", err)
	}
	if len(prices) != 4 {
		t.Fatalf("got %d prices, want 4", len(prices))
	}
	for name, want := range map[string]struct{ in, out int64 }{
		"gpt-5":    {1250, 10000}, // price file entry for its policy_name
		"renamed":  {90, 600},     // price file entry for policy_name qwen3-235b-a22b-2507
		"override": {2000, 4500},  // explicit prices win over the price file
		"local":    {0, 0},        // explicit zero prices, no file lookup
	} {
		p, ok := prices[name]
		if !ok {
			t.Errorf("%s: missing", name)
			continue
		}
		if p.Name != name {
			t.Errorf("%s: Name = %q, want the configured model name", name, p.Name)
		}
		if p.InNanos != want.in || p.OutNanos != want.out {
			t.Errorf("%s: nanos = %d / %d, want %d / %d", name, p.InNanos, p.OutNanos, want.in, want.out)
		}
	}
}

func TestModelPricesExplicitOnlyNeedsNoFile(t *testing.T) {
	cfg := parsePriceConfig(t, "")
	cfg.Models = cfg.Models[2:] // override and local carry explicit prices
	prices, err := modelPrices(cfg)
	if err != nil {
		t.Fatalf("modelPrices: %v", err)
	}
	if len(prices) != 2 || prices["override"].InNanos != 2000 || prices["local"].OutNanos != 0 {
		t.Fatalf("prices = %+v", prices)
	}
}

func TestModelPricesUnpricedModel(t *testing.T) {
	t.Run("no price file", func(t *testing.T) {
		_, err := modelPrices(parsePriceConfig(t, ""))
		if err == nil {
			t.Fatal("want an error")
		}
		for _, name := range []string{`"gpt-5"`, `"renamed"`} {
			if !strings.Contains(err.Error(), name) {
				t.Errorf("error %q does not name %s", err, name)
			}
		}
		for _, name := range []string{`"override"`, `"local"`} {
			if strings.Contains(err.Error(), name) {
				t.Errorf("error %q names the priced model %s", err, name)
			}
		}
	})
	t.Run("absent from the price file", func(t *testing.T) {
		cfg := parsePriceConfig(t, "../../deploy/prices.yaml")
		cfg.Models[1].PolicyName = "not-in-the-price-file"
		_, err := modelPrices(cfg)
		if err == nil || !strings.Contains(err.Error(), `"renamed"`) || !strings.Contains(err.Error(), "not-in-the-price-file") {
			t.Fatalf("err = %v, want one naming renamed and its policy_name", err)
		}
	})
}

func TestModelPricesErrors(t *testing.T) {
	t.Run("unreadable price file", func(t *testing.T) {
		cfg := parsePriceConfig(t, filepath.Join(t.TempDir(), "absent.yaml"))
		if _, err := modelPrices(cfg); err == nil || !strings.Contains(err.Error(), "read prices") {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("explicit price below one nano-USD", func(t *testing.T) {
		cfg := parsePriceConfig(t, "../../deploy/prices.yaml")
		tiny := 0.0001
		cfg.Models[2].InputPerMTok = &tiny
		if _, err := modelPrices(cfg); err == nil || !strings.Contains(err.Error(), `"override"`) {
			t.Fatalf("err = %v, want one naming override", err)
		}
	})
}

const replayProfilesYAML = `simulated: true
seed: 1
jitter: 0.1
models:
  - {name: m-rp, rank: 0, ttft_ms: 100, decode_tokens_per_s: 100}
`

const replayGatewayYAML = `
tenants:
  - name: tenant-a
    api_key_sha256:
      - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
providers:
  - name: replay
    type: replay
    replay:
      file: data/test.jsonl
      latency_profiles: profiles.yaml
      time_scale: 0.001
models:
  - name: m-rp
    provider: replay
    input_per_mtok: 1
    output_per_mtok: 2
`

func TestBuildProvidersReplay(t *testing.T) {
	dir := t.TempDir()
	sum := sha256.Sum256([]byte("hello"))
	record := fmt.Sprintf(`{"prompt_sha256":%q,"model":"m-rp","prompt_id":"ds:1","dataset":"ds","text":"world","prompt_tokens":3,"completion_tokens":5,"cost_usd":0.1,"score":1.0,"success":true}`+"\n", hex.EncodeToString(sum[:]))
	writeFile(t, dir, "profiles.yaml", replayProfilesYAML)
	writeFile(t, dir, "gateway.yaml", replayGatewayYAML)

	// The records file is not there yet: the provider cannot be built.
	cfg, err := config.Load(filepath.Join(dir, "gateway.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildProviders(cfg, func(string) string { return "" }, http.DefaultClient); err == nil || !strings.Contains(err.Error(), `provider "replay"`) {
		t.Fatalf("err = %v, want one naming provider replay", err)
	}

	if err := os.Mkdir(filepath.Join(dir, "data"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, dir, "data/test.jsonl", record)
	providers, err := buildProviders(cfg, func(string) string { return "" }, http.DefaultClient)
	if err != nil {
		t.Fatal(err)
	}
	rp, ok := providers["replay"].(*replay.Provider)
	if !ok {
		t.Fatalf("replay is %T, want *replay.Provider", providers["replay"])
	}
	resp, err := rp.Complete(context.Background(), provider.Request{
		Model:     "m-rp",
		MaxTokens: 10,
		Messages:  []provider.Message{{Role: provider.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "world" || resp.Usage == nil || *resp.Usage != (provider.Usage{PromptTokens: 3, CompletionTokens: 5}) {
		t.Fatalf("resp = %+v", resp)
	}
}

func TestValidateConfigDoesNotRequireReplayFile(t *testing.T) {
	// Neither the records file (gitignored third-party data) nor the profile
	// file exists; validate-config only checks that the paths are set.
	dir := t.TempDir()
	cfg := writeFile(t, dir, "gateway.yaml", replayGatewayYAML)
	code, out, errs := runCmd("validate-config", "--config", cfg)
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	if out != "config OK: 1 tenants, 1 providers, 1 models\n" {
		t.Fatalf("stdout = %q", out)
	}
}

func TestReplayBenchConfig(t *testing.T) {
	c, err := config.Load("../../bench/configs/replay.yaml")
	if err != nil {
		t.Fatal(err)
	}
	pol, err := policy.LoadFile("../../trainer/artifacts/policy.json")
	if err != nil {
		t.Fatal(err)
	}
	profiles, err := replay.LoadProfiles("../../bench/latency_profiles.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var limits struct {
		MaxOutputTokens int `json:"max_output_tokens"`
	}
	if err := json.Unmarshal([]byte(mustRead(t, "../../bench/limits.json")), &limits); err != nil {
		t.Fatal(err)
	}
	abs := func(p string) string {
		a, err := filepath.Abs(p)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}

	if c.Server.DefaultDeadline != 10*time.Minute {
		t.Errorf("DefaultDeadline = %v, want 10m", c.Server.DefaultDeadline)
	}
	if c.Policy.Path != abs("../../trainer/artifacts/policy.json") || c.Pricing.File != abs("../../deploy/prices.yaml") {
		t.Errorf("Policy.Path, Pricing.File = %q, %q", c.Policy.Path, c.Pricing.File)
	}

	if len(c.Tenants) != 1 {
		t.Fatalf("got %d tenants, want 1", len(c.Tenants))
	}
	ten := c.Tenants[0]
	if ten.Name != "bench" || !ten.AllowStrategyOverride || ten.MaxDeadline != 10*time.Minute || ten.MaxOutputTokens != limits.MaxOutputTokens {
		t.Errorf("tenant = %+v, want bench with override, 10m deadline and %d output tokens", ten, limits.MaxOutputTokens)
	}
	if got, ok := auth.NewStore(c.Tenants).Lookup("test-key-bench"); !ok || got.Name != "bench" {
		t.Errorf("Lookup(test-key-bench) = %v, %v", got, ok)
	}

	if len(c.Providers) != 1 {
		t.Fatalf("got %d providers, want 1", len(c.Providers))
	}
	p := c.Providers[0]
	if p.Name != "replay" || p.Type != "replay" || p.Replay.TimeScale != 0.01 {
		t.Errorf("provider = %+v", p)
	}
	if p.Replay.File != abs("../../trainer/artifacts/replay/test.jsonl") || p.Replay.LatencyProfiles != abs("../../bench/latency_profiles.yaml") {
		t.Errorf("replay files = %q, %q", p.Replay.File, p.Replay.LatencyProfiles)
	}

	if len(c.Models) != len(pol.Models) {
		t.Fatalf("got %d models, want %d", len(c.Models), len(pol.Models))
	}
	for i, pm := range pol.Models {
		m := c.Models[i]
		if m.Name != pm.Name || m.PolicyName != pm.Name || m.UpstreamModel != pm.Name || m.Provider != "replay" {
			t.Errorf("models[%d] = %q / %q / %q on %q, want %q on replay", i, m.Name, m.PolicyName, m.UpstreamModel, m.Provider, pm.Name)
		}
		if m.Strongest != (pm.Name == pol.Strongest) {
			t.Errorf("%s: strongest = %v, policy strongest model is %s", pm.Name, m.Strongest, pol.Strongest)
		}
		prof, ok := profiles[pm.Name]
		if !ok {
			t.Fatalf("%s: no latency profile", pm.Name)
		}
		if want := (config.LatencyPrior{TTFTMS: prof.TTFTMS, DecodeTokensPerS: prof.DecodeTokensPerS, Jitter: prof.Jitter}); m.LatencyPrior != want {
			t.Errorf("%s: latency_prior = %+v, want %+v", pm.Name, m.LatencyPrior, want)
		}
		if m.InputPerMTok != nil || m.OutputPerMTok != nil {
			t.Errorf("%s: explicit prices set, want prices from the price file", pm.Name)
		}
	}

	code, out, errs := runCmd("validate-config", "--config", "../../bench/configs/replay.yaml")
	if code != 0 {
		t.Fatalf("validate-config exit %d, stderr %q", code, errs)
	}
	if want := "config OK: 1 tenants, 1 providers, 6 models, policy " + pol.Version + "\n"; out != want {
		t.Fatalf("stdout = %q, want %q", out, want)
	}
}

func TestBuildHealthUsesConfiguredPriors(t *testing.T) {
	cfg := parsePoolConfig(t, "m-", poolPolicyNames...)
	for i := range cfg.Models {
		cfg.Models[i].LatencyPrior = config.LatencyPrior{
			TTFTMS:           float64(100 * (i + 1)),
			DecodeTokensPerS: float64(10 * (i + 1)),
			Jitter:           0.1,
		}
	}
	hr := buildHealth(cfg)
	for i, m := range cfg.Models {
		lp := m.LatencyPrior
		want := (lp.TTFTMS + 1000*250/lp.DecodeTokensPerS) * (1 + 1.645*lp.Jitter/math.Sqrt(3))
		if got := hr.PredictedP95MS(m.Name, 250); math.Abs(got-want) > 1e-9 {
			t.Errorf("models[%d] %s: PredictedP95MS = %v, want %v", i, m.Name, got, want)
		}
	}
	// Estimators are keyed by configured name, not by policy name.
	if got := hr.PredictedP95MS("gpt-5", 250); !math.IsInf(got, 1) {
		t.Errorf("PredictedP95MS(policy name) = %v, want +Inf", got)
	}
}

func TestGatewayRoutesWithLatencyPriors(t *testing.T) {
	cfg, err := config.Load("../../deploy/configs/policy-fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	pol, err := policy.LoadFile("../../trainer/artifacts/policy.json")
	if err != nil {
		t.Fatal(err)
	}
	for i := range cfg.Providers {
		cfg.Providers[i].Fake.TTFT = 0
		cfg.Providers[i].Fake.PerToken = 0
	}
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))

	const body = `{"model":"routeforge-auto","messages":[{"role":"user","content":"hello"}]}`
	resp, raw := apiRequest(t, http.MethodPost, base+"/v1/routing/explain", demoKey, body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("explain status %d, body %s", resp.StatusCode, raw)
	}
	var d struct {
		Candidates []struct {
			Model          string   `json:"model"`
			PredictedP95MS *float64 `json:"predicted_p95_ms"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	if len(d.Candidates) != len(pol.Models) {
		t.Fatalf("got %d candidates, want %d", len(d.Candidates), len(pol.Models))
	}
	for _, c := range d.Candidates {
		m, ok := cfg.Model(c.Model)
		if !ok {
			t.Fatalf("candidate %q is not a configured model", c.Model)
		}
		var eot float64
		for _, pm := range pol.Models {
			if pm.Name == m.PolicyName {
				eot = pm.ExpectedOutputTokens
			}
		}
		lp := m.LatencyPrior
		want := (lp.TTFTMS + 1000*eot/lp.DecodeTokensPerS) * (1 + 1.645*lp.Jitter/math.Sqrt(3))
		if c.PredictedP95MS == nil || math.Abs(*c.PredictedP95MS-want) > 1e-6 {
			t.Errorf("%s: predicted_p95_ms = %v, want %v", c.Model, c.PredictedP95MS, want)
		}
	}
}

func TestBuildQueues(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cfg := parseWireConfig(t) // m-oa, m-an, m-fk
		cfg.Models[0].MaxConcurrency, cfg.Models[0].QueueSize = 1, 0
		cfg.Models[1].MaxConcurrency, cfg.Models[1].QueueSize, cfg.Models[1].QueueTimeout = 1, 1, 3*time.Second
		q := buildQueues(cfg)
		ctx := context.Background()

		// m-oa: one slot and no waiting room, so it is full as soon as it is busy.
		if q.Full("m-oa") {
			t.Fatal("m-oa is full while idle")
		}
		held, err := q.Acquire(ctx, "m-oa")
		if err != nil {
			t.Fatalf("Acquire(m-oa): %v", err)
		}
		if !q.Full("m-oa") {
			t.Error("m-oa is not full with its only slot busy and no waiting room")
		}
		if _, err := q.Acquire(ctx, "m-oa"); !errors.Is(err, reliability.ErrQueueFull) {
			t.Errorf("second Acquire(m-oa) = %v, want ErrQueueFull", err)
		}
		held()

		// m-an: one slot, one place in line and a 3s wait, from its own settings.
		slot, err := q.Acquire(ctx, "m-an")
		if err != nil {
			t.Fatalf("Acquire(m-an): %v", err)
		}
		start := time.Now()
		if _, err := q.Acquire(ctx, "m-an"); !errors.Is(err, reliability.ErrQueueTimeout) {
			t.Errorf("waiting Acquire(m-an) = %v, want ErrQueueTimeout", err)
		}
		if got := time.Since(start); got != 3*time.Second {
			t.Errorf("m-an waited %v, want its queue_timeout of 3s", got)
		}
		slot()

		// m-fk keeps the defaults: 64 slots, 128 places in line.
		var releases []func()
		for range 64 {
			release, err := q.Acquire(ctx, "m-fk")
			if err != nil {
				t.Fatalf("Acquire(m-fk): %v", err)
			}
			releases = append(releases, release)
		}
		if q.Full("m-fk") {
			t.Error("m-fk is full with 64 slots busy and an empty line")
		}
		for _, release := range releases {
			release()
		}
	})
}

func TestBuildQueuesCoversEveryModel(t *testing.T) {
	cfg := parsePoolConfig(t, "m-", poolPolicyNames...)
	q := buildQueues(cfg)
	for _, m := range cfg.Models {
		release, err := q.Acquire(context.Background(), m.Name)
		if err != nil {
			t.Errorf("Acquire(%s): %v", m.Name, err)
			continue
		}
		release()
	}
}

// holdingUpstream is an OpenAI-style upstream that holds every request until
// release is closed. entered is closed when the first request arrives.
func holdingUpstream(t *testing.T) (url string, entered <-chan struct{}, release func()) {
	t.Helper()
	enteredCh := make(chan struct{})
	releaseCh := make(chan struct{})
	var enteredOnce, releaseOnce sync.Once
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enteredOnce.Do(func() { close(enteredCh) })
		select {
		case <-releaseCh:
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	release = func() { releaseOnce.Do(func() { close(releaseCh) }) }
	return srv.URL, enteredCh, release
}

func TestGatewayLimitsInFlightRequests(t *testing.T) {
	upstream, entered, release := holdingUpstream(t)
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
server:
  max_inflight: 1
tenants:
  - name: e2e
    api_key_sha256:
      - %s
providers:
  - name: up
    type: openai
    base_url: %s/v1
models:
  - name: m-up
    provider: up
    input_per_mtok: 1
    output_per_mtok: 2
`, auth.HashKey(e2eKey), upstream)))
	if err != nil {
		t.Fatal(err)
	}
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))
	// Cleanups run last in, first out: free the held request before the
	// gateway's own shutdown waits for it.
	t.Cleanup(release)

	status := make(chan int, 1)
	go func() {
		req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(chatBody("m-up", "")))
		if err != nil {
			status <- -1
			return
		}
		req.Header.Set("Authorization", "Bearer "+e2eKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := e2eClient.Do(req)
		if err != nil {
			status <- -1
			return
		}
		_ = resp.Body.Close()
		status <- resp.StatusCode
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("the first request never reached the upstream")
	}

	resp := post(t, base, e2eKey, chatBody("m-up", ""))
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("second request status = %d, want 503 while max_inflight 1 is held", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil || env.Error.Code != "overloaded" {
		t.Errorf("error body code = %q (decode error %v), want overloaded", env.Error.Code, err)
	}

	release()
	select {
	case got := <-status:
		if got != http.StatusOK {
			t.Fatalf("held request status = %d, want 200", got)
		}
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("the held request never finished")
	}
}

// scriptedUpstream is an OpenAI-style upstream that answers its n-th chat
// request with statuses[n]; the last status repeats. A 200 carries a short
// completion with usage, any other status a bare error body.
func scriptedUpstream(t *testing.T, statuses ...int) (url string, hits func() int) {
	t.Helper()
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		status := statuses[min(int(n.Add(1))-1, len(statuses)-1)]
		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"error":{"message":"scripted failure"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"c-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() int { return int(n.Load()) }
}

// upstreamConfig is a one-model configuration over an OpenAI-style upstream.
// extra is YAML placed before the tenants, for example a reliability section.
func upstreamConfig(t *testing.T, upstream, extra string) *config.Config {
	t.Helper()
	cfg, err := config.Parse([]byte(fmt.Sprintf(`
%s
tenants:
  - name: e2e
    api_key_sha256:
      - %s
providers:
  - name: up
    type: openai
    base_url: %s/v1
models:
  - name: m-up
    provider: up
    input_per_mtok: 1
    output_per_mtok: 2
`, extra, auth.HashKey(e2eKey), upstream)))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// The gateway admits and meters requests: the wiring gives the API an
// accountant and the router its budget signal.
func TestGatewayEnforcesTenantLimits(t *testing.T) {
	readBody := func(t *testing.T, resp *http.Response) (code string, raw []byte) {
		t.Helper()
		defer resp.Body.Close()
		raw, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		var env struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		_ = json.Unmarshal(raw, &env) // a success body has no error
		return env.Error.Code, raw
	}
	start := func(t *testing.T, edit func(*config.Tenant)) (base string, hits func() int) {
		t.Helper()
		upstream, hits := scriptedUpstream(t, http.StatusOK)
		cfg := upstreamConfig(t, upstream, "")
		edit(&cfg.Tenants[0])
		base, _ = startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))
		return base, hits
	}

	t.Run("requests per minute", func(t *testing.T) {
		base, hits := start(t, func(ten *config.Tenant) { ten.RPM = 1 })
		resp := post(t, base, e2eKey, chatBody("m-up", ""))
		if code, raw := readBody(t, resp); resp.StatusCode != http.StatusOK {
			t.Fatalf("first request: status = %d (%s); body %s", resp.StatusCode, code, raw)
		}
		resp = post(t, base, e2eKey, chatBody("m-up", ""))
		if code, raw := readBody(t, resp); resp.StatusCode != http.StatusTooManyRequests || code != "rate_limited" {
			t.Fatalf("second request: status = %d (%s), want 429 rate_limited; body %s", resp.StatusCode, code, raw)
		}
		if resp.Header.Get("Retry-After") == "" {
			t.Error("no Retry-After on the 429")
		}
		if got := hits(); got != 1 {
			t.Errorf("upstream saw %d requests, want 1", got)
		}
	})

	t.Run("the router sees the budget", func(t *testing.T) {
		// The budget cannot hold the reservation of 4096 output tokens. The
		// router refuses the model first, which only happens when it is given
		// the accountant's budget signal; without it the reservation would be
		// refused later as budget_exceeded.
		base, hits := start(t, func(ten *config.Tenant) { ten.DailyBudgetUSD = 0.001 })
		resp := post(t, base, e2eKey, chatBody("m-up", ""))
		code, raw := readBody(t, resp)
		if resp.StatusCode != http.StatusServiceUnavailable || code != "no_eligible_model" || !strings.Contains(string(raw), "budget") {
			t.Fatalf("status = %d (%s), want 503 no_eligible_model naming the budget; body %s", resp.StatusCode, code, raw)
		}
		if got := hits(); got != 0 {
			t.Errorf("upstream saw %d requests, want 0", got)
		}
	})

	t.Run("a request that fits is metered", func(t *testing.T) {
		base, _ := start(t, func(ten *config.Tenant) { ten.DailyBudgetUSD = 1 })
		resp := post(t, base, e2eKey, chatBody("m-up", ""))
		code, raw := readBody(t, resp)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d (%s); body %s", resp.StatusCode, code, raw)
		}
		var body struct {
			RouteForge struct {
				CostUSD    *float64 `json:"cost_usd"`
				CostSource string   `json:"cost_source"`
			} `json:"routeforge"`
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		// The upstream reports 1 prompt and 1 completion token, at 1000 and
		// 2000 nano-USD per token.
		if got := body.RouteForge.CostUSD; got == nil || *got != 3000.0/1e9 || body.RouteForge.CostSource != "reported" {
			t.Errorf("cost = %v %q, want 3e-06 reported; body %s", got, body.RouteForge.CostSource, raw)
		}
	})
}

func TestBuildBreakersUsesTheConfiguredThresholds(t *testing.T) {
	cfg := parseWireConfig(t)
	cfg.Reliability.Breaker.ConsecutiveFailures = 2
	b := buildBreakers(cfg)

	fail := func() {
		t.Helper()
		ticket, ok := b.Allow("m-oa")
		if !ok {
			t.Fatal("Allow refused a request")
		}
		ticket.Done(false)
	}
	fail()
	if b.IsOpen("m-oa") {
		t.Fatal("the breaker opened after one failure, want two")
	}
	fail()
	if !b.IsOpen("m-oa") {
		t.Fatal("the breaker did not open after two consecutive failures")
	}
	if b.IsOpen("m-an") {
		t.Fatal("the breaker of another model opened")
	}
}

// downProvider fails every call as unavailable and counts the calls.
type downProvider struct{ calls atomic.Int32 }

func (p *downProvider) Complete(context.Context, provider.Request) (*provider.Response, error) {
	p.calls.Add(1)
	return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "down", Message: "down"}
}

func (p *downProvider) Stream(context.Context, provider.Request) (provider.Stream, error) {
	p.calls.Add(1)
	return nil, &provider.Error{Kind: provider.KindUnavailable, Provider: "down", Message: "down"}
}

// buildDownExecutor returns the executor buildExecutor makes for cfg when every
// model's provider is down, and that provider.
func buildDownExecutor(cfg *config.Config) (*executor.Executor, *downProvider) {
	down := &downProvider{}
	providers := map[string]provider.Provider{"oa": down, "anthropic": down, "fk": down}
	return buildExecutor(cfg, providers, buildBreakers(cfg), buildQueues(cfg), buildHealth(cfg), nil, nil), down
}

func TestBuildExecutorAppliesTheReliabilitySettings(t *testing.T) {
	pinned := func() *router.Decision {
		return &router.Decision{Chosen: "m-oa", Fallbacks: []string{}, Attempts: []router.Attempt{}}
	}

	t.Run("max attempts and retry budget", func(t *testing.T) {
		cases := []struct {
			name         string
			attempts     int
			budgetFloor  int
			wantAttempts int32
		}{
			{"attempts bound the calls", 2, 3, 2},
			{"the budget bounds the retries", 4, 1, 2}, // 20% of one request is 0, plus the floor of 1
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				cfg := parseWireConfig(t)
				cfg.Reliability.MaxAttempts = tc.attempts
				cfg.Reliability.RetryBudgetMin = tc.budgetFloor
				cfg.Reliability.BackoffBase, cfg.Reliability.BackoffCap = time.Millisecond, time.Millisecond
				ex, down := buildDownExecutor(cfg)

				if _, err := ex.Complete(context.Background(), pinned(), provider.Request{}, nil); err == nil {
					t.Fatal("want the upstream error")
				}
				if got := down.calls.Load(); got != tc.wantAttempts {
					t.Fatalf("provider called %d times, want %d", got, tc.wantAttempts)
				}
			})
		}
	})

	t.Run("backoff", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			cfg := parseWireConfig(t)
			cfg.Reliability.MaxAttempts = 2
			cfg.Reliability.BackoffBase, cfg.Reliability.BackoffCap = 100*time.Millisecond, 100*time.Millisecond
			ex, down := buildDownExecutor(cfg)

			start := time.Now()
			if _, err := ex.Complete(context.Background(), pinned(), provider.Request{}, nil); err == nil {
				t.Fatal("want the upstream error")
			}
			if got := down.calls.Load(); got != 2 {
				t.Fatalf("provider called %d times, want 2", got)
			}
			// Full jitter: the wait before the retry is drawn from [0, 100ms).
			if got := time.Since(start); got <= 0 || got >= 100*time.Millisecond {
				t.Fatalf("waited %v before the retry, want a jittered delay inside (0, 100ms)", got)
			}
		})
	})
}

func TestGatewayRetriesAnUnavailableUpstream(t *testing.T) {
	cases := []struct {
		name       string
		reliab     string
		wantStatus int
		wantHits   int
	}{
		{"default attempts", "reliability:\n  backoff_base: 1ms\n  backoff_cap: 2ms\n", http.StatusOK, 2},
		{"one attempt", "reliability:\n  max_attempts: 1\n", http.StatusBadGateway, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			upstream, hits := scriptedUpstream(t, http.StatusServiceUnavailable, http.StatusOK)
			base, _ := startGatewayConfig(t, upstreamConfig(t, upstream, tc.reliab), slog.New(slog.DiscardHandler))

			resp := post(t, base, e2eKey, chatBody("m-up", ""))
			defer resp.Body.Close()
			raw, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != tc.wantStatus {
				t.Fatalf("status = %d, want %d; body %s", resp.StatusCode, tc.wantStatus, raw)
			}
			if got := hits(); got != tc.wantHits {
				t.Fatalf("upstream saw %d requests, want %d", got, tc.wantHits)
			}
			if tc.wantStatus != http.StatusOK {
				return
			}
			var body struct {
				RouteForge struct {
					Attempts int `json:"attempts"`
				} `json:"routeforge"`
			}
			if err := json.Unmarshal(raw, &body); err != nil || body.RouteForge.Attempts != 2 {
				t.Fatalf("routeforge.attempts = %d (decode error %v), want 2; body %s", body.RouteForge.Attempts, err, raw)
			}
		})
	}
}

// A model whose breaker has opened stops being routed to: the router sees it
// through Signals.BreakerOpen and refuses before any upstream call.
func TestGatewayStopsRoutingToAnOpenBreaker(t *testing.T) {
	upstream, hits := scriptedUpstream(t, http.StatusServiceUnavailable)
	cfg := upstreamConfig(t, upstream, "reliability:\n  max_attempts: 1\n  breaker:\n    consecutive_failures: 1\n")
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))

	resp := post(t, base, e2eKey, chatBody("m-up", ""))
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("first request status = %d, want 502", resp.StatusCode)
	}

	resp = post(t, base, e2eKey, chatBody("m-up", ""))
	defer resp.Body.Close()
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable || env.Error.Code != "no_eligible_model" || !strings.Contains(env.Error.Message, "breaker_open") {
		t.Fatalf("second request: status %d, error %+v; want 503 no_eligible_model naming breaker_open", resp.StatusCode, env.Error)
	}
	if got := hits(); got != 1 {
		t.Fatalf("upstream saw %d requests, want 1: the open breaker must keep the second away", got)
	}
}

// The executor holds a model slot for the whole call, so with one slot and no
// waiting room the router sees the model as full: 503 overloaded.
func TestGatewayHoldsAModelSlotWhileACallRuns(t *testing.T) {
	upstream, entered, release := holdingUpstream(t)
	cfg := upstreamConfig(t, upstream, "")
	cfg.Models[0].MaxConcurrency, cfg.Models[0].QueueSize = 1, 0 // after defaults: 0 means no room
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))
	t.Cleanup(release)

	status := make(chan int, 1)
	go func() {
		resp := post(t, base, e2eKey, chatBody("m-up", ""))
		_ = resp.Body.Close()
		status <- resp.StatusCode
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("the first request never reached the upstream")
	}

	resp := post(t, base, e2eKey, chatBody("m-up", ""))
	defer resp.Body.Close()
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable || env.Error.Code != "overloaded" {
		t.Fatalf("second request: status %d, code %q; want 503 overloaded while the only slot is held", resp.StatusCode, env.Error.Code)
	}
	if got := resp.Header.Get("Retry-After"); got != "1" {
		t.Errorf("Retry-After = %q, want 1", got)
	}

	release()
	select {
	case got := <-status:
		if got != http.StatusOK {
			t.Fatalf("held request status = %d, want 200", got)
		}
	case <-time.After(5 * time.Second): // a failure guard, not a wait
		t.Fatal("the held request never finished")
	}
	// The slot came back with the response.
	final := post(t, base, e2eKey, chatBody("m-up", ""))
	_ = final.Body.Close()
	if final.StatusCode != http.StatusOK {
		t.Fatalf("request after the slot was freed: status %d, want 200", final.StatusCode)
	}
}

// Successful calls feed the latency estimators, so the prediction the router
// reads moves off the configured prior for the model that served the call and
// for no other. (A fast call far below the estimate lowers the mean but widens
// the variance, so the p95 prediction may move either way.)
func TestGatewayLearnsLatencyFromSuccessfulCalls(t *testing.T) {
	cfg, err := config.Load("../../deploy/configs/policy-fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for i := range cfg.Providers {
		cfg.Providers[i].Fake.TTFT = 0
		cfg.Providers[i].Fake.PerToken = 0
	}
	base, _ := startGatewayConfig(t, cfg, slog.New(slog.DiscardHandler))

	const body = `{"model":"routeforge-auto","messages":[{"role":"user","content":"hello"}]}`
	predictions := func() map[string]float64 {
		t.Helper()
		resp, raw := apiRequest(t, http.MethodPost, base+"/v1/routing/explain", demoKey, body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("explain status %d, body %s", resp.StatusCode, raw)
		}
		var d struct {
			Candidates []struct {
				Model          string   `json:"model"`
				PredictedP95MS *float64 `json:"predicted_p95_ms"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal(raw, &d); err != nil {
			t.Fatalf("decode %s: %v", raw, err)
		}
		out := map[string]float64{}
		for _, c := range d.Candidates {
			if c.PredictedP95MS == nil {
				t.Fatalf("%s: no predicted_p95_ms", c.Model)
			}
			out[c.Model] = *c.PredictedP95MS
		}
		return out
	}

	before := predictions()
	resp, raw := apiRequest(t, http.MethodPost, base+"/v1/chat/completions", demoKey, body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status %d, body %s", resp.StatusCode, raw)
	}
	served := resp.Header.Get("X-RouteForge-Model")
	after := predictions()

	if _, ok := before[served]; !ok {
		t.Fatalf("served model %q is not a candidate", served)
	}
	for model, was := range before {
		switch got := after[model]; {
		case model == served && got == was:
			t.Errorf("%s served the call, but its prediction stayed at %v", model, was)
		case model != served && got != was:
			t.Errorf("%s did not serve the call, but its prediction went from %v to %v", model, was, got)
		}
	}
}

func TestServeRequiresConfigFlag(t *testing.T) {
	var out, errb bytes.Buffer
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if code := run(ctx, []string{"serve"}, &out, &errb, func(string) string { return "" }); code != 2 {
		t.Fatalf("exit %d, stderr %q", code, errb.String())
	}
	if !strings.Contains(errb.String(), "--config is required") {
		t.Fatalf("stderr %q", errb.String())
	}
}

// poolPolicyNames are the models of the committed policy, in pool order.
var poolPolicyNames = []string{
	"gpt-5",
	"claude-sonnet-4",
	"gemini-2.5-pro",
	"gemini-2.5-flash",
	"deepseek-v3.1-terminus",
	"qwen3-235b-a22b-2507",
}

const (
	committedPolicy = "../../trainer/artifacts/policy.json"
	pricesFile      = "../../deploy/prices.yaml"
)

// parsePoolConfig returns a configuration with one model per policy name,
// served by a fake provider. Each model is named prefix+policyName.
func parsePoolConfig(t *testing.T, prefix string, policyNames ...string) *config.Config {
	t.Helper()
	var b strings.Builder
	b.WriteString(`
tenants:
  - name: tenant-a
    api_key_sha256:
      - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
providers:
  - name: fk
    type: fake
models:
`)
	for _, name := range policyNames {
		fmt.Fprintf(&b, "  - name: %s%s\n    provider: fk\n    policy_name: %s\n", prefix, name, name)
	}
	cfg, err := config.Parse([]byte(b.String()))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestBuildRegistry(t *testing.T) {
	t.Run("loads and activates the configured policy", func(t *testing.T) {
		cfg := parsePoolConfig(t, "", poolPolicyNames...)
		cfg.Policy.Path = committedPolicy
		reg, err := buildRegistry(cfg)
		if err != nil {
			t.Fatalf("buildRegistry: %v", err)
		}
		active := reg.Active()
		if active == nil {
			t.Fatal("no active policy")
		}
		if !regexp.MustCompile(`^[0-9a-f]{12}$`).MatchString(active.Version) {
			t.Errorf("version = %q, want 12 lowercase hex characters", active.Version)
		}
		acts := reg.Activations()
		if len(acts) != 1 || acts[0].Version != active.Version || acts[0].Actor != "startup" || acts[0].Previous != "" {
			t.Errorf("activations = %+v, want one startup activation of %s", acts, active.Version)
		}
	})

	t.Run("models map to the policy through policy_name", func(t *testing.T) {
		cfg := parsePoolConfig(t, "m-", poolPolicyNames...)
		cfg.Policy.Path = committedPolicy
		reg, err := buildRegistry(cfg)
		if err != nil {
			t.Fatalf("buildRegistry: %v", err)
		}
		if reg.Active() == nil {
			t.Fatal("no active policy")
		}
	})

	t.Run("no policy path gives an empty registry", func(t *testing.T) {
		reg, err := buildRegistry(parsePoolConfig(t, "", poolPolicyNames...))
		if err != nil {
			t.Fatalf("buildRegistry: %v", err)
		}
		if reg == nil || reg.Active() != nil || len(reg.List()) != 0 {
			t.Fatalf("registry = %+v, want an empty one", reg)
		}
	})

	t.Run("unreadable policy", func(t *testing.T) {
		cfg := parsePoolConfig(t, "", poolPolicyNames...)
		cfg.Policy.Path = filepath.Join(t.TempDir(), "absent.json")
		if _, err := buildRegistry(cfg); err == nil || !strings.Contains(err.Error(), "read policy") {
			t.Fatalf("err = %v, want a read error", err)
		}
	})

	t.Run("policy model without a configured model", func(t *testing.T) {
		cfg := parsePoolConfig(t, "", poolPolicyNames[:5]...)
		cfg.Policy.Path = committedPolicy
		_, err := buildRegistry(cfg)
		if err == nil || !strings.Contains(err.Error(), "qwen3-235b-a22b-2507") {
			t.Fatalf("err = %v, want one naming the missing model", err)
		}
	})
}

func TestPolicyReadyCheck(t *testing.T) {
	cfg := parsePoolConfig(t, "", poolPolicyNames...)
	reg, err := buildRegistry(cfg) // no policy path: nothing is active
	if err != nil {
		t.Fatal(err)
	}
	check := policyReadyCheck(reg)
	if check.Name != "policy" {
		t.Errorf("Name = %q, want policy", check.Name)
	}
	if err := check.Check(context.Background()); err == nil || err.Error() != "no active policy" {
		t.Fatalf("Check with no active policy = %v, want the fixed text \"no active policy\"", err)
	}

	cfg.Policy.Path = committedPolicy
	if reg, err = buildRegistry(cfg); err != nil {
		t.Fatal(err)
	}
	if err := policyReadyCheck(reg).Check(context.Background()); err != nil {
		t.Errorf("Check with an active policy = %v, want nil", err)
	}
}

func TestServeStartupFailuresCloseListener(t *testing.T) {
	cases := []struct {
		name string
		edit func(t *testing.T, cfg *config.Config)
		want string
	}{
		{"unpriced model", func(*testing.T, *config.Config) {}, "no price"},
		{"unreadable policy", func(t *testing.T, cfg *config.Config) {
			cfg.Pricing.File = pricesFile
			cfg.Policy.Path = filepath.Join(t.TempDir(), "absent.json")
		}, "read policy"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := parsePoolConfig(t, "", poolPolicyNames...)
			tc.edit(t, cfg)
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ln.Close() })

			err = serve(context.Background(), cfg, ln, func(string) string { return "" }, slog.New(slog.DiscardHandler))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("serve = %v, want an error containing %q", err, tc.want)
			}
			if _, err := ln.Accept(); !errors.Is(err, net.ErrClosed) {
				t.Fatalf("Accept after failed serve = %v, want net.ErrClosed", err)
			}
		})
	}
}
