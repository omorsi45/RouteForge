package config_test

import (
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

const minimalDoc = `
tenants:
  - name: tenant-a
    api_key_sha256:
      - aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
providers:
  - name: fake
    type: fake
models:
  - name: m-fast
    provider: fake
`

func TestParseAppliesDefaults(t *testing.T) {
	c, err := config.Load("testdata/minimal.yaml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	s := c.Server
	if s.Listen != ":8080" {
		t.Errorf("Listen = %q", s.Listen)
	}
	if s.ReadHeaderTimeout != 5*time.Second {
		t.Errorf("ReadHeaderTimeout = %v", s.ReadHeaderTimeout)
	}
	if s.IdleTimeout != 120*time.Second {
		t.Errorf("IdleTimeout = %v", s.IdleTimeout)
	}
	if s.MaxHeaderBytes != 1<<20 {
		t.Errorf("MaxHeaderBytes = %d", s.MaxHeaderBytes)
	}
	if s.MaxBodyBytes != 2<<20 {
		t.Errorf("MaxBodyBytes = %d", s.MaxBodyBytes)
	}
	if s.DefaultDeadline != 120*time.Second {
		t.Errorf("DefaultDeadline = %v", s.DefaultDeadline)
	}
	if s.ShutdownTimeout != 30*time.Second {
		t.Errorf("ShutdownTimeout = %v", s.ShutdownTimeout)
	}
	if c.Defaults.Profile != "balanced" || c.Defaults.MaxDeadline != 120*time.Second || c.Defaults.MaxOutputTokens != 4096 {
		t.Errorf("Defaults = %+v", c.Defaults)
	}
	ten := c.Tenants[0]
	if ten.Profile != "balanced" || ten.MaxDeadline != 120*time.Second || ten.MaxOutputTokens != 4096 {
		t.Errorf("tenant = %+v", ten)
	}
	m, ok := c.Model("m-fast")
	if !ok {
		t.Fatal("m-fast not found")
	}
	if m.UpstreamModel != "m-fast" || m.PolicyName != "m-fast" {
		t.Errorf("model = %+v", m)
	}
	p, ok := c.Provider("fake")
	if !ok {
		t.Fatal("provider fake not found")
	}
	if p.Fake.OutputTokens != 16 || p.Fake.DisconnectAfter != 1 || p.Fake.ErrorKind != "unavailable" {
		t.Errorf("fake = %+v", p.Fake)
	}
	if _, ok := c.Model("missing"); ok {
		t.Error("Model(missing) found")
	}
	if _, ok := c.Provider("missing"); ok {
		t.Error("Provider(missing) found")
	}
}

func TestParseOpenAIAndAnthropicDefaults(t *testing.T) {
	doc := minimalDoc + `
  - name: m-oa
    provider: oa
  - name: m-an
    provider: an
`
	doc = strings.Replace(doc, "providers:\n", `providers:
  - name: oa
    type: openai
  - name: an
    type: anthropic
    api_key_env: ANTHROPIC_API_KEY
`, 1)
	c, err := config.Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	oa, _ := c.Provider("oa")
	if oa.BaseURL != "https://api.openai.com/v1" || oa.MaxTokensParam != "max_completion_tokens" {
		t.Errorf("openai = %+v", oa)
	}
	an, _ := c.Provider("an")
	if an.BaseURL != "https://api.anthropic.com" {
		t.Errorf("anthropic = %+v", an)
	}
}

func TestParseDurationStrings(t *testing.T) {
	doc := "server: {default_deadline: 45s}\n" + strings.Replace(minimalDoc, "    api_key_sha256:", "    max_deadline: 1500ms\n    api_key_sha256:", 1)
	c, err := config.Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Server.DefaultDeadline != 45*time.Second {
		t.Errorf("DefaultDeadline = %v", c.Server.DefaultDeadline)
	}
	if c.Tenants[0].MaxDeadline != 1500*time.Millisecond {
		t.Errorf("MaxDeadline = %v", c.Tenants[0].MaxDeadline)
	}
	if c.Defaults.MaxDeadline != 45*time.Second {
		t.Errorf("Defaults.MaxDeadline = %v", c.Defaults.MaxDeadline)
	}
}

func TestParseRejectsUnknownField(t *testing.T) {
	_, err := config.Parse([]byte("server: {listn: \":1\"}\n" + minimalDoc))
	if err == nil || !strings.Contains(err.Error(), "listn") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseRejectsEmptyAndMultiDoc(t *testing.T) {
	if _, err := config.Parse([]byte("")); err == nil || !strings.Contains(err.Error(), "config is empty") {
		t.Fatalf("empty err = %v", err)
	}
	if _, err := config.Parse([]byte(minimalDoc + "\n---\n" + minimalDoc)); err == nil || !strings.Contains(err.Error(), "multiple YAML documents") {
		t.Fatalf("multi err = %v", err)
	}
}

func TestTenantAllows(t *testing.T) {
	open := config.Tenant{}
	if !open.Allows("x") {
		t.Error("nil AllowedModels should allow x")
	}
	limited := config.Tenant{AllowedModels: []string{"a"}}
	if !limited.Allows("a") || limited.Allows("b") {
		t.Error("limited tenant allow mismatch")
	}
}

func TestParseRejectsNaNJitter(t *testing.T) {
	doc := strings.Replace(minimalDoc, "    type: fake\n", "    type: fake\n    fake:\n      jitter: .nan\n", 1)
	_, err := config.Parse([]byte(doc))
	if err == nil {
		t.Fatal("expected error for jitter: .nan")
	}
	if want := "providers[0].fake.jitter: must be in [0, 1)"; !strings.Contains(err.Error(), want) {
		t.Fatalf("error %q does not contain %q", err, want)
	}
}
