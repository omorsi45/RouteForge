package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
)

func runCmd(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := run(context.Background(), args, &out, &errb, func(string) string { return "" })
	return code, out.String(), errb.String()
}

func TestValidateConfigOK(t *testing.T) {
	code, out, errs := runCmd("validate-config", "--config", "../../internal/config/testdata/minimal.yaml")
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errs)
	}
	if out != "config OK: 1 tenants, 1 providers, 2 models\n" {
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
