package main

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
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
