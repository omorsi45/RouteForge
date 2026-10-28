package config_test

import (
	"os"
	"path/filepath"
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

func TestModelDefaults(t *testing.T) {
	c, err := config.Load("testdata/minimal.yaml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	m, _ := c.Model("m-fast")
	if m.LatencyPrior.TTFTMS != 500 || m.LatencyPrior.DecodeTokensPerS != 50 || m.LatencyPrior.Jitter != 0.1 {
		t.Errorf("LatencyPrior = %+v", m.LatencyPrior)
	}
	if m.MaxConcurrency != 64 || m.QueueSize != 128 || m.QueueTimeout != 2*time.Second {
		t.Errorf("MaxConcurrency, QueueSize, QueueTimeout = %d, %d, %v", m.MaxConcurrency, m.QueueSize, m.QueueTimeout)
	}
	if m.ContextWindow != 0 || m.InputPerMTok != nil || m.OutputPerMTok != nil {
		t.Errorf("unset fields = %d, %v, %v", m.ContextWindow, m.InputPerMTok, m.OutputPerMTok)
	}
	if c.Policy.Path != "" || c.Pricing.File != "" {
		t.Errorf("Policy, Pricing = %+v, %+v", c.Policy, c.Pricing)
	}
}

func TestParseAdminSection(t *testing.T) {
	digest := strings.Repeat("b", 64)

	c, err := config.Parse([]byte(minimalDoc + "admin:\n  api_key_sha256:\n    - " + digest + "\n"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got := c.Admin.APIKeySHA256; len(got) != 1 || got[0] != digest {
		t.Errorf("Admin.APIKeySHA256 = %v, want [%s]", got, digest)
	}

	c, err = config.Parse([]byte(minimalDoc))
	if err != nil {
		t.Fatalf("Parse without admin: %v", err)
	}
	if len(c.Admin.APIKeySHA256) != 0 {
		t.Errorf("Admin.APIKeySHA256 = %v, want none", c.Admin.APIKeySHA256)
	}

	// The tenant's own digest cannot also be an admin key.
	tenant := strings.Repeat("a", 64)
	_, err = config.Parse([]byte(minimalDoc + "admin:\n  api_key_sha256:\n    - " + tenant + "\n"))
	if err == nil || !strings.Contains(err.Error(), "admin.api_key_sha256[0]") {
		t.Errorf("Parse with a tenant digest as admin key: err = %v", err)
	}

	_, err = config.Parse([]byte(minimalDoc + "admin:\n  keys: []\n"))
	if err == nil || !strings.Contains(err.Error(), "keys") {
		t.Errorf("Parse with an unknown admin field: err = %v", err)
	}
}

func TestParseRoutingFields(t *testing.T) {
	doc := strings.Replace(minimalDoc, "    provider: fake\n", `    provider: fake
    input_per_mtok: 0
    output_per_mtok: 2.5
    context_window: 128000
    latency_prior: {ttft_ms: 250}
    max_concurrency: 8
    queue_size: 4
    queue_timeout: 750ms
`, 1)
	c, err := config.Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	m := c.Models[0]
	if m.InputPerMTok == nil || *m.InputPerMTok != 0 {
		t.Errorf("InputPerMTok = %v, want a pointer to 0", m.InputPerMTok)
	}
	if m.OutputPerMTok == nil || *m.OutputPerMTok != 2.5 {
		t.Errorf("OutputPerMTok = %v, want a pointer to 2.5", m.OutputPerMTok)
	}
	if m.ContextWindow != 128000 {
		t.Errorf("ContextWindow = %d", m.ContextWindow)
	}
	// Defaults fill the unset latency_prior fields one by one.
	if m.LatencyPrior.TTFTMS != 250 || m.LatencyPrior.DecodeTokensPerS != 50 || m.LatencyPrior.Jitter != 0.1 {
		t.Errorf("LatencyPrior = %+v", m.LatencyPrior)
	}
	if m.MaxConcurrency != 8 || m.QueueSize != 4 || m.QueueTimeout != 750*time.Millisecond {
		t.Errorf("MaxConcurrency, QueueSize, QueueTimeout = %d, %d, %v", m.MaxConcurrency, m.QueueSize, m.QueueTimeout)
	}
}

func TestConfigResolvesRelativePaths(t *testing.T) {
	root := t.TempDir()
	sub := filepath.Join(root, "sub")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	abs := filepath.ToSlash(filepath.Join(root, "elsewhere", "prices.yaml"))
	doc := minimalDoc + "policy:\n  path: ../p.json\npricing:\n  file: " + abs + "\n"
	path := filepath.Join(sub, "c.yaml")
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}

	c, err := config.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Dir != sub {
		t.Errorf("Dir = %q, want %q", c.Dir, sub)
	}
	if want := filepath.Join(root, "p.json"); c.Policy.Path != want {
		t.Errorf("Policy.Path = %q, want %q", c.Policy.Path, want)
	}
	if c.Pricing.File != abs {
		t.Errorf("Pricing.File = %q, want the absolute path %q unchanged", c.Pricing.File, abs)
	}
}

func TestConfigResolvesRelativePathsFromRelativeConfig(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "c.yaml")
	if err := os.WriteFile(path, []byte(minimalDoc+"pricing:\n  file: prices.yaml\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	c, err := config.Load("c.yaml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !filepath.IsAbs(c.Pricing.File) || filepath.Base(c.Pricing.File) != "prices.yaml" {
		t.Errorf("Pricing.File = %q, want an absolute path to prices.yaml", c.Pricing.File)
	}
	if !filepath.IsAbs(c.Dir) {
		t.Errorf("Dir = %q, want an absolute path", c.Dir)
	}
}

func TestResolvePath(t *testing.T) {
	dir := t.TempDir()
	c := &config.Config{Dir: dir}
	if got, want := c.ResolvePath("a/b.json"), filepath.Join(dir, "a", "b.json"); got != want {
		t.Errorf("relative = %q, want %q", got, want)
	}
	abs := filepath.Join(t.TempDir(), "x.json")
	if got := c.ResolvePath(abs); got != abs {
		t.Errorf("absolute = %q, want %q", got, abs)
	}
	if got := c.ResolvePath(""); got != "" {
		t.Errorf("empty = %q, want empty", got)
	}
	if got := (&config.Config{}).ResolvePath("rel.json"); got != "rel.json" {
		t.Errorf("no Dir = %q, want the path unchanged", got)
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
