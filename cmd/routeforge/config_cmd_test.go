package main

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"go.yaml.in/yaml/v3"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
)

func runCmd(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := run(context.Background(), args, &out, &errb, func(string) string { return "" })
	return code, out.String(), errb.String()
}

func TestValidateConfigOK(t *testing.T) {
	code, out, errs := runCmd("validate-config", "--config", "../../deploy/configs/fake.yaml")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	if out != "config OK: 1 tenants, 2 providers, 2 models\n" {
		t.Fatalf("stdout = %q", out)
	}
}

func TestValidateConfigInvalid(t *testing.T) {
	doc := strings.Replace(mustRead(t, "../../internal/config/testdata/minimal.yaml"), "tenants:\n  - name: tenant-a\n", "tenants:\n  - name: tenant-a\n    profile: fast\n", 1)
	path := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	code, _, errs := runCmd("validate-config", "--config", path)
	if code != 1 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(errs, "invalid config") || !strings.Contains(errs, "tenants[0].profile") {
		t.Fatalf("stderr = %q", errs)
	}
}

func TestValidateConfigMissingFlag(t *testing.T) {
	if code, _, _ := runCmd("validate-config"); code != 2 {
		t.Fatalf("exit %d", code)
	}
}

func TestExampleConfigsValidate(t *testing.T) {
	files, err := filepath.Glob("../../deploy/configs/*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 2 {
		t.Fatalf("found %d example configs, want at least 2", len(files))
	}
	for _, f := range files {
		if _, err := config.Load(f); err != nil {
			t.Errorf("%s: %v", f, err)
		}
		if code, _, errs := runCmd("validate-config", "--config", f); code != 0 {
			t.Errorf("%s: validate-config exit %d, stderr %q", f, code, errs)
		}
	}
}

func mustRead(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestFakeExampleConfigDemoKey(t *testing.T) {
	c, err := config.Load("../../deploy/configs/fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	got, ok := auth.NewStore(c.Tenants).Lookup("rf_local_demo_only")
	if !ok || got.Name != "demo" {
		t.Fatalf("Lookup = %v, %v", got, ok)
	}
}

// writeFile writes body to dir/name and returns the path.
func writeFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const onePriceFile = `effective_date: "2026-10-26"
source: "test table"
currency: USD
models:
  - {name: m-fast, input_per_mtok: 0.1, output_per_mtok: 0.4}
`

func TestValidateConfigLoadsPolicyAndPrices(t *testing.T) {
	pol, err := policy.LoadFile("../../trainer/artifacts/policy.json")
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	code, out, errs := runCmd("validate-config", "--config", "../../deploy/configs/policy-fake.yaml")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	if want := "config OK: 1 tenants, 6 providers, 6 models, policy " + pol.Version + "\n"; out != want {
		t.Fatalf("stdout = %q, want %q", out, want)
	}
	if !regexp.MustCompile(`, policy [0-9a-f]{12}\n$`).MatchString(out) {
		t.Fatalf("stdout %q does not end with a 12 hex policy version", out)
	}
}

func TestValidateConfigPricesFromFile(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "prices.yaml", onePriceFile+"  - {name: m-strong, input_per_mtok: 1.0, output_per_mtok: 4.0}\n")
	cfg := writeFile(t, dir, "gateway.yaml", mustRead(t, "../../internal/config/testdata/minimal.yaml")+"\npricing:\n  file: prices.yaml\n")

	code, out, errs := runCmd("validate-config", "--config", cfg)
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	if out != "config OK: 1 tenants, 1 providers, 2 models\n" {
		t.Fatalf("stdout = %q", out)
	}
}

func TestValidateConfigUnpricedModel(t *testing.T) {
	dir := t.TempDir()
	// The price file lists m-fast only, so m-strong has no price.
	writeFile(t, dir, "prices.yaml", onePriceFile)
	cfg := writeFile(t, dir, "gateway.yaml", mustRead(t, "../../internal/config/testdata/minimal.yaml")+"\npricing:\n  file: prices.yaml\n")

	code, out, errs := runCmd("validate-config", "--config", cfg)
	if code != 1 {
		t.Fatalf("exit %d, stdout %q, stderr %q", code, out, errs)
	}
	if !strings.Contains(errs, "invalid config") || !strings.Contains(errs, `"m-strong"`) {
		t.Fatalf("stderr = %q, want it to name m-strong", errs)
	}
	if strings.Contains(errs, `"m-fast"`) {
		t.Fatalf("stderr = %q names the priced model m-fast", errs)
	}
	if out != "" {
		t.Fatalf("stdout = %q, want empty", out)
	}
}

func TestValidateConfigNoPriceSource(t *testing.T) {
	// minimal.yaml sets no model price and no pricing.file.
	code, _, errs := runCmd("validate-config", "--config", "../../internal/config/testdata/minimal.yaml")
	if code != 1 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	for _, want := range []string{`"m-fast"`, `"m-strong"`} {
		if !strings.Contains(errs, want) {
			t.Errorf("stderr %q does not name %s", errs, want)
		}
	}
}

func TestValidateConfigBadPriceFile(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "prices.yaml", strings.Replace(onePriceFile, "USD", "EUR", 1))
	cfg := writeFile(t, dir, "gateway.yaml", mustRead(t, "../../deploy/configs/fake.yaml")+"\npricing:\n  file: prices.yaml\n")
	// Both models carry explicit prices, yet a price file that is set must load.
	code, _, errs := runCmd("validate-config", "--config", cfg)
	if code != 1 || !strings.Contains(errs, "currency") {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
}

func TestValidateConfigPolicyProblems(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "bad.json", "{}")
	for name, policyPath := range map[string]string{"invalid artifact": "bad.json", "missing file": "absent.json"} {
		t.Run(name, func(t *testing.T) {
			cfg := writeFile(t, dir, "gateway.yaml", mustRead(t, "../../deploy/configs/fake.yaml")+"\npolicy:\n  path: "+policyPath+"\n")
			code, out, errs := runCmd("validate-config", "--config", cfg)
			if code != 1 {
				t.Fatalf("exit %d, stdout %q, stderr %q", code, out, errs)
			}
			if !strings.Contains(errs, "invalid config") || !strings.Contains(errs, filepath.Join(dir, policyPath)) {
				t.Fatalf("stderr = %q, want it to name the policy file", errs)
			}
		})
	}
}

func TestPolicyFakeConfigMatchesProfiles(t *testing.T) {
	c, err := config.Load("../../deploy/configs/policy-fake.yaml")
	if err != nil {
		t.Fatal(err)
	}
	pol, err := policy.LoadFile("../../trainer/artifacts/policy.json")
	if err != nil {
		t.Fatal(err)
	}
	var profiles struct {
		Jitter float64 `yaml:"jitter"`
		Models []struct {
			Name             string  `yaml:"name"`
			TTFTMS           float64 `yaml:"ttft_ms"`
			DecodeTokensPerS float64 `yaml:"decode_tokens_per_s"`
		} `yaml:"models"`
	}
	if err := yaml.Unmarshal([]byte(mustRead(t, "../../bench/latency_profiles.yaml")), &profiles); err != nil {
		t.Fatal(err)
	}

	wantPolicy, _ := filepath.Abs("../../trainer/artifacts/policy.json")
	wantPrices, _ := filepath.Abs("../../deploy/prices.yaml")
	if c.Policy.Path != wantPolicy || c.Pricing.File != wantPrices {
		t.Errorf("Policy.Path, Pricing.File = %q, %q; want %q, %q", c.Policy.Path, c.Pricing.File, wantPolicy, wantPrices)
	}
	if len(c.Providers) != len(pol.Models) || len(c.Models) != len(pol.Models) || len(profiles.Models) != len(pol.Models) {
		t.Fatalf("providers, models, profiles = %d, %d, %d; want %d each", len(c.Providers), len(c.Models), len(profiles.Models), len(pol.Models))
	}

	for i, pm := range pol.Models {
		var ttft, decode float64
		for _, p := range profiles.Models {
			if p.Name == pm.Name {
				ttft, decode = p.TTFTMS, p.DecodeTokensPerS
			}
		}
		if ttft == 0 || decode == 0 {
			t.Fatalf("%s: no latency profile", pm.Name)
		}

		m, p := c.Models[i], c.Providers[i]
		if m.Name != pm.Name || m.PolicyName != pm.Name || m.UpstreamModel != pm.Name {
			t.Errorf("models[%d] = %q / %q / %q, want %q", i, m.Name, m.PolicyName, m.UpstreamModel, pm.Name)
		}
		if want := fmt.Sprintf("fake-%d", i+1); p.Name != want || m.Provider != want || p.Type != "fake" {
			t.Errorf("%s: provider = %q (model refers to %q, type %q), want %q", pm.Name, p.Name, m.Provider, p.Type, want)
		}
		if m.Strongest != (pm.Name == pol.Strongest) {
			t.Errorf("%s: strongest = %v, policy strongest model is %s", pm.Name, m.Strongest, pol.Strongest)
		}
		if m.InputPerMTok != nil || m.OutputPerMTok != nil {
			t.Errorf("%s: explicit prices set, want prices from the price file", pm.Name)
		}
		f := p.Fake
		wantTTFT := time.Duration(math.Round(ttft*1e6)) * time.Nanosecond
		wantPerToken := time.Duration(math.Round(1e6/decode)) * time.Microsecond
		if f.Seed != int64(i+1) || f.TTFT != wantTTFT || f.PerToken != wantPerToken || f.Jitter != 0.1 || f.OutputTokens != 64 {
			t.Errorf("%s: fake = seed %d ttft %v per_token %v jitter %v output_tokens %d; want seed %d ttft %v per_token %v jitter 0.1 output_tokens 64",
				pm.Name, f.Seed, f.TTFT, f.PerToken, f.Jitter, f.OutputTokens, i+1, wantTTFT, wantPerToken)
		}
		wantPrior := config.LatencyPrior{TTFTMS: ttft, DecodeTokensPerS: decode, Jitter: profiles.Jitter}
		if m.LatencyPrior != wantPrior {
			t.Errorf("%s: latency_prior = %+v, want %+v", pm.Name, m.LatencyPrior, wantPrior)
		}
	}

	if len(c.Tenants) != 1 || !c.Tenants[0].AllowStrategyOverride {
		t.Fatalf("tenants = %+v", c.Tenants)
	}
	if got, ok := auth.NewStore(c.Tenants).Lookup("rf_local_demo_only"); !ok || got.Name != "demo" {
		t.Fatalf("Lookup = %v, %v", got, ok)
	}
}
