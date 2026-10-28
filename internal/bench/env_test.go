package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/auth"
)

// testKey is the benchmark key the test configurations accept.
const testKey = "test-key-bench"

// writeFile creates path, and its parent directories, with the given content.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// minimalConfig is the smallest gateway configuration that loads. policyPath
// is written to policy.path when it is not empty.
func minimalConfig(policyPath string) string {
	policy := ""
	if policyPath != "" {
		policy = "policy:\n  path: " + policyPath + "\n"
	}
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
`, auth.HashKey(testKey), policy)
}

func TestCollectEnvFields(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "trainer/artifacts/reports/training.json"),
		`{"environment": {"python": "3.14.10", "numpy": "2.5.4"}, "policy_version": "other"}`)
	writeFile(t, filepath.Join(root, "data/manifest.json"),
		`{"source": {"dataset": "NPULH/LLMRouterBench", "revision": "0e5af1b84bf7"}}`)
	writeFile(t, filepath.Join(root, "trainer/artifacts/policy.json"), `{"version": "7ae71d0e6deb"}`)
	cfg := minimalConfig("../../trainer/artifacts/policy.json")
	writeFile(t, filepath.Join(root, "bench/configs/replay.yaml"), cfg)

	env, err := CollectEnv(root, "bench/configs/replay.yaml")
	if err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256([]byte(cfg))
	for _, c := range []struct{ name, got, want string }{
		{"python_version", env.PythonVersion, "3.14.10"},
		{"dataset_revision", env.DatasetRevision, "0e5af1b84bf7"},
		{"policy_version", env.PolicyVersion, "7ae71d0e6deb"},
		{"config_sha256", env.ConfigSHA256, hex.EncodeToString(sum[:])},
		{"go_version", env.GoVersion, runtime.Version()},
	} {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}
	if env.CPUs != runtime.NumCPU() {
		t.Errorf("cpus = %d, want %d", env.CPUs, runtime.NumCPU())
	}
	if !strings.HasPrefix(env.OS, runtime.GOOS) {
		t.Errorf("os = %q, want a value that starts with %q", env.OS, runtime.GOOS)
	}
	if env.CPUModel == "" {
		t.Error("cpu_model is empty, want the model name or the architecture")
	}
	if env.RAMBytes < 0 {
		t.Errorf("ram_bytes = %d, want 0 or more", env.RAMBytes)
	}
}

func TestCollectEnvMissingOptionalValuesAreEmpty(t *testing.T) {
	root := t.TempDir()
	// No training report, no manifest and no policy file: none of them is
	// required to describe a run.
	writeFile(t, filepath.Join(root, "bench/configs/replay.yaml"), minimalConfig("../../trainer/artifacts/policy.json"))

	env, err := CollectEnv(root, "bench/configs/replay.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for name, got := range map[string]string{
		"python_version":   env.PythonVersion,
		"dataset_revision": env.DatasetRevision,
		"policy_version":   env.PolicyVersion,
	} {
		if got != "" {
			t.Errorf("%s = %q, want empty", name, got)
		}
	}
	if env.ConfigSHA256 == "" {
		t.Error("config_sha256 is empty, want the digest of the config file")
	}
}

func TestCollectEnvRejectsBadInputs(t *testing.T) {
	t.Run("missing config", func(t *testing.T) {
		if _, err := CollectEnv(t.TempDir(), "bench/configs/none.yaml"); err == nil {
			t.Fatal("want an error for a config file that does not exist")
		}
	})
	t.Run("invalid config", func(t *testing.T) {
		root := t.TempDir()
		writeFile(t, filepath.Join(root, "bad.yaml"), "tenants: []\n")
		if _, err := CollectEnv(root, "bad.yaml"); err == nil {
			t.Fatal("want an error for an invalid config")
		}
	})
	t.Run("malformed manifest", func(t *testing.T) {
		root := t.TempDir()
		writeFile(t, filepath.Join(root, "bench/configs/replay.yaml"), minimalConfig(""))
		writeFile(t, filepath.Join(root, "data/manifest.json"), "{not json")
		if _, err := CollectEnv(root, "bench/configs/replay.yaml"); err == nil {
			t.Fatal("want an error for a manifest that is not JSON")
		}
	})
}

func TestCollectEnvAbsoluteConfigPath(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "elsewhere", "replay.yaml")
	writeFile(t, path, minimalConfig(""))
	env, err := CollectEnv(root, path)
	if err != nil {
		t.Fatal(err)
	}
	if env.ConfigSHA256 == "" {
		t.Error("config_sha256 is empty for an absolute config path")
	}
}

// fakeHost answers the commands and files an Env asks the host for.
func fakeHost(files map[string]string, commands map[string]string) hostSource {
	return hostSource{
		readFile: func(name string) ([]byte, error) {
			if s, ok := files[name]; ok {
				return []byte(s), nil
			}
			return nil, os.ErrNotExist
		},
		run: func(name string, args ...string) ([]byte, error) {
			key := filepath.Base(name) + " " + strings.Join(args, " ")
			if s, ok := commands[key]; ok {
				return []byte(s), nil
			}
			return nil, errors.New("command not available: " + key)
		},
	}
}

func TestHostFacts(t *testing.T) {
	host := fakeHost(map[string]string{
		"/etc/os-release": "NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\nID=ubuntu\n",
		"/proc/cpuinfo":   "processor\t: 0\nmodel name\t: AMD Ryzen 9 7900X 12-Core Processor\nmodel name\t: second\n",
		"/proc/meminfo":   "MemTotal:       32762000 kB\nMemFree:         1000 kB\n",
	}, map[string]string{"uname -r": "6.18.33.2-microsoft-standard-WSL2\n"})

	if got, want := host.osName(), runtime.GOOS+" (Ubuntu 24.04.2 LTS)"; got != want {
		t.Errorf("os = %q, want %q", got, want)
	}
	if got, want := host.kernel(), "6.18.33.2-microsoft-standard-WSL2"; got != want {
		t.Errorf("kernel = %q, want %q", got, want)
	}
	if got, want := host.cpuModel(), "AMD Ryzen 9 7900X 12-Core Processor"; got != want {
		t.Errorf("cpu_model = %q, want %q", got, want)
	}
	if got, want := host.ramBytes(), int64(32762000)*1024; got != want {
		t.Errorf("ram_bytes = %d, want %d", got, want)
	}
}

func TestHostFactsFallBack(t *testing.T) {
	host := fakeHost(nil, nil)
	if got := host.osName(); got != runtime.GOOS {
		t.Errorf("os = %q, want %q without /etc/os-release", got, runtime.GOOS)
	}
	if got := host.kernel(); got != "" {
		t.Errorf("kernel = %q, want empty without uname", got)
	}
	if got := host.cpuModel(); got != runtime.GOARCH {
		t.Errorf("cpu_model = %q, want %q without /proc/cpuinfo", got, runtime.GOARCH)
	}
	if got := host.ramBytes(); got != 0 {
		t.Errorf("ram_bytes = %d, want 0 without /proc/meminfo", got)
	}
}

func TestParseOSRelease(t *testing.T) {
	for _, c := range []struct{ name, in, want string }{
		{"double quotes", "PRETTY_NAME=\"Debian GNU/Linux 12 (bookworm)\"\n", "Debian GNU/Linux 12 (bookworm)"},
		{"single quotes", "PRETTY_NAME='Alpine Linux v3.20'\n", "Alpine Linux v3.20"},
		{"no quotes", "PRETTY_NAME=Arch\n", "Arch"},
		{"absent", "NAME=x\nID=y\n", ""},
		{"empty", "", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := parseOSRelease([]byte(c.in)); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestParseMemTotal(t *testing.T) {
	for _, c := range []struct {
		name, in string
		want     int64
	}{
		{"kilobytes", "MemTotal:        2048 kB\n", 2048 * 1024},
		{"after other lines", "MemFree: 1 kB\nMemTotal: 4 kB\n", 4 * 1024},
		{"absent", "MemFree: 1 kB\n", 0},
		{"not a number", "MemTotal: lots kB\n", 0},
		{"wrong unit", "MemTotal: 5 MB\n", 0},
		{"empty", "", 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := parseMemTotal([]byte(c.in)); got != c.want {
				t.Errorf("got %d, want %d", got, c.want)
			}
		})
	}
}

func TestGitSHA(t *testing.T) {
	const sha = "0123456789abcdef0123456789abcdef01234567"
	status := "status --porcelain --untracked-files=no"
	for _, c := range []struct {
		name     string
		commands map[string]string
		want     string
	}{
		{"clean", map[string]string{"git rev-parse HEAD": sha + "\n", "git " + status: ""}, sha},
		{"dirty", map[string]string{"git rev-parse HEAD": sha + "\n", "git " + status: " M internal/bench/env.go\n"}, sha + "-dirty"},
		{"no repository", map[string]string{}, ""},
		{"status unavailable", map[string]string{"git rev-parse HEAD": sha + "\n"}, ""},
		{"empty head", map[string]string{"git rev-parse HEAD": "\n", "git " + status: ""}, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			// The -C directory is part of the arguments the runner sees.
			commands := map[string]string{}
			for k, v := range c.commands {
				name, rest, _ := strings.Cut(k, " ")
				commands[name+" -C /repo "+rest] = v
			}
			if got := fakeHost(nil, commands).gitSHA("/repo"); got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}
