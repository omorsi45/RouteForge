package main

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"path/filepath"
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
