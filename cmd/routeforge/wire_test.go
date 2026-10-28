package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/anthropic"
	"github.com/omorsi45/routeforge/internal/provider/fake"
	"github.com/omorsi45/routeforge/internal/provider/openai"
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
