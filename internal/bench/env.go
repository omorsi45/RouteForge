package bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

// Env describes the machine, the code and the inputs of a run, so that a result
// can be traced to them. A value that cannot be found is the empty string (0
// for RAMBytes), never a guess.
type Env struct {
	OS              string `json:"os"`               // GOOS, with the distribution name when the host gives one
	Kernel          string `json:"kernel"`           // uname -r
	CPUModel        string `json:"cpu_model"`        // the first "model name" of /proc/cpuinfo, else GOARCH
	CPUs            int    `json:"cpus"`             // logical CPUs
	RAMBytes        int64  `json:"ram_bytes"`        // MemTotal of /proc/meminfo
	GoVersion       string `json:"go_version"`       // of the benchmark binary
	PythonVersion   string `json:"python_version"`   // of the trainer run, from training.json
	GitSHA          string `json:"git_sha"`          // HEAD, with -dirty when tracked files differ
	ConfigSHA256    string `json:"config_sha256"`    // of the gateway config file
	DatasetRevision string `json:"dataset_revision"` // of the dataset, from the manifest
	PolicyVersion   string `json:"policy_version"`   // of the policy file the config names
}

// commandTimeout bounds each command CollectEnv runs.
const commandTimeout = 30 * time.Second

// hostSource is where Env reads the facts of the host. Tests replace it.
type hostSource struct {
	readFile func(name string) ([]byte, error)
	run      func(name string, args ...string) ([]byte, error)
}

var systemHost = hostSource{
	readFile: os.ReadFile,
	run: func(name string, args ...string) ([]byte, error) {
		ctx, cancel := context.WithTimeout(context.Background(), commandTimeout)
		defer cancel()
		return exec.CommandContext(ctx, name, args...).Output()
	},
}

// CollectEnv describes the host and the inputs of a run. root is the
// repository root; configPath is the gateway config of the run, resolved
// against root when it is relative. A config that cannot be read or loaded is
// an error. The trainer report, the data manifest and the policy file are
// optional: one that is missing leaves its field empty, and one that exists
// but is not valid JSON is an error.
func CollectEnv(root, configPath string) (Env, error) {
	return collectEnv(root, configPath, systemHost)
}

func collectEnv(root, configPath string, host hostSource) (Env, error) {
	configFile := configPath
	if !filepath.IsAbs(configFile) {
		configFile = filepath.Join(root, configFile)
	}
	raw, err := os.ReadFile(configFile)
	if err != nil {
		return Env{}, fmt.Errorf("collect env: read config: %w", err)
	}
	cfg, err := config.Load(configFile)
	if err != nil {
		return Env{}, fmt.Errorf("collect env: %w", err)
	}
	sum := sha256.Sum256(raw)

	env := Env{
		OS:           host.osName(),
		Kernel:       host.kernel(),
		CPUModel:     host.cpuModel(),
		CPUs:         runtime.NumCPU(),
		RAMBytes:     host.ramBytes(),
		GoVersion:    runtime.Version(),
		GitSHA:       host.gitSHA(root),
		ConfigSHA256: hex.EncodeToString(sum[:]),
	}

	var report struct {
		Environment struct {
			Python string `json:"python"`
		} `json:"environment"`
	}
	if err := readOptionalJSON(filepath.Join(root, "trainer", "artifacts", "reports", "training.json"), &report); err != nil {
		return Env{}, err
	}
	env.PythonVersion = report.Environment.Python

	var manifest struct {
		Source struct {
			Revision string `json:"revision"`
		} `json:"source"`
	}
	if err := readOptionalJSON(filepath.Join(root, "data", "manifest.json"), &manifest); err != nil {
		return Env{}, err
	}
	env.DatasetRevision = manifest.Source.Revision

	if cfg.Policy.Path != "" {
		var pol struct {
			Version string `json:"version"`
		}
		if err := readOptionalJSON(cfg.Policy.Path, &pol); err != nil {
			return Env{}, err
		}
		env.PolicyVersion = pol.Version
	}
	return env, nil
}

// readOptionalJSON decodes the JSON file at path into v. A file that does not
// exist is not an error and leaves v untouched.
func readOptionalJSON(path string, v any) error {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("collect env: %w", err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("collect env: %s: %w", filepath.Base(path), err)
	}
	return nil
}

func (h hostSource) osName() string {
	data, err := h.readFile("/etc/os-release")
	if err != nil {
		return runtime.GOOS
	}
	if pretty := parseOSRelease(data); pretty != "" {
		return runtime.GOOS + " (" + pretty + ")"
	}
	return runtime.GOOS
}

func (h hostSource) kernel() string {
	out, err := h.run("uname", "-r")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func (h hostSource) cpuModel() string {
	if data, err := h.readFile("/proc/cpuinfo"); err == nil {
		if model := parseCPUModel(data); model != "" {
			return model
		}
	}
	return runtime.GOARCH
}

func (h hostSource) ramBytes() int64 {
	data, err := h.readFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	return parseMemTotal(data)
}

// gitSHA is HEAD of the repository at root, suffixed -dirty when tracked files
// differ from it. It is empty when either git command fails: a repository whose
// state is unknown is not called clean.
func (h hostSource) gitSHA(root string) string {
	git := gitExecutable()
	head, err := h.run(git, "-C", root, "rev-parse", "HEAD")
	if err != nil {
		return ""
	}
	sha := strings.TrimSpace(string(head))
	if sha == "" {
		return ""
	}
	status, err := h.run(git, "-C", root, "status", "--porcelain", "--untracked-files=no")
	if err != nil {
		return ""
	}
	if strings.TrimSpace(string(status)) != "" {
		return sha + "-dirty"
	}
	return sha
}

// gitExecutable is the system git when it is where it usually is, as the
// trainer does, else whatever git is on the PATH.
func gitExecutable() string {
	const system = "/usr/bin/git"
	if _, err := os.Stat(system); err == nil {
		return system
	}
	return "git"
}

// parseOSRelease returns the PRETTY_NAME of an os-release file, or "".
func parseOSRelease(data []byte) string {
	for _, line := range strings.Split(string(data), "\n") {
		value, ok := strings.CutPrefix(strings.TrimSpace(line), "PRETTY_NAME=")
		if !ok {
			continue
		}
		if n := len(value); n >= 2 && (value[0] == '"' || value[0] == '\'') && value[n-1] == value[0] {
			value = value[1 : n-1]
		}
		return value
	}
	return ""
}

// parseCPUModel returns the first "model name" of a cpuinfo file, or "".
func parseCPUModel(data []byte) string {
	for _, line := range strings.Split(string(data), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if ok && strings.TrimSpace(key) == "model name" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// parseMemTotal returns MemTotal of a meminfo file in bytes, or 0 when it is
// missing or not written in kB.
func parseMemTotal(data []byte) int64 {
	for _, line := range strings.Split(string(data), "\n") {
		rest, ok := strings.CutPrefix(line, "MemTotal:")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) != 2 || fields[1] != "kB" {
			return 0
		}
		kb, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil || kb < 0 {
			return 0
		}
		return kb * 1024
	}
	return 0
}
