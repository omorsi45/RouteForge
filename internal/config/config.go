// Package config loads, defaults and validates the gateway configuration.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"time"

	"go.yaml.in/yaml/v3"

	"github.com/omorsi45/routeforge/internal/provider"
)

// AutoModel is the model name that asks the gateway to route the request.
const AutoModel = "routeforge-auto"

// Profiles lists the accepted tenant routing profiles.
var Profiles = []string{"strict", "balanced", "economy"}

// Config is the full gateway configuration.
type Config struct {
	Server      Server      `yaml:"server"`
	Defaults    Defaults    `yaml:"defaults"`
	Reliability Reliability `yaml:"reliability"`
	Policy      Policy      `yaml:"policy"`
	Pricing     Pricing     `yaml:"pricing"`
	Admin       Admin       `yaml:"admin"`
	Tenants     []Tenant    `yaml:"tenants"`
	Providers   []Provider  `yaml:"providers"`
	Models      []Model     `yaml:"models"`

	// Dir is the absolute directory of the file Load read; it is empty for
	// a Config built by Parse. Relative file paths in the config resolve
	// against it.
	Dir string `yaml:"-"`
}

// Reliability holds the retry and circuit breaker settings. Unset fields take
// the defaults of spec section 10.
type Reliability struct {
	MaxAttempts int           `yaml:"max_attempts"` // tries per request, the first included
	BackoffBase time.Duration `yaml:"backoff_base"`
	BackoffCap  time.Duration `yaml:"backoff_cap"`

	// Retries are allowed while they number fewer than RetryBudgetRatio times
	// the requests in the last RetryBudgetWindow plus RetryBudgetMin (the
	// floor, Ruling R5, lets an isolated request fail over).
	RetryBudgetRatio  float64       `yaml:"retry_budget_ratio"`
	RetryBudgetWindow time.Duration `yaml:"retry_budget_window"`
	RetryBudgetMin    int           `yaml:"retry_budget_min"`

	Breaker Breaker `yaml:"breaker"`
}

// Breaker configures the per-model circuit breaker: it opens after
// ConsecutiveFailures failures in a row, or when at least MinRequests requests
// in the last Window failed at ErrorRate or more. It stays open for OpenFor,
// then admits HalfOpenProbes concurrent probes.
type Breaker struct {
	ConsecutiveFailures int           `yaml:"consecutive_failures"`
	ErrorRate           float64       `yaml:"error_rate"`
	MinRequests         int           `yaml:"min_requests"`
	Window              time.Duration `yaml:"window"`
	OpenFor             time.Duration `yaml:"open_for"`
	HalfOpenProbes      int           `yaml:"half_open_probes"`
}

// Admin holds the keys that may call the /admin routes, as SHA-256 digests
// like tenant keys. Without a key the admin routes do not exist.
type Admin struct {
	APIKeySHA256 []string `yaml:"api_key_sha256"`
}

// Policy locates the trained routing policy artifact.
type Policy struct {
	Path string `yaml:"path"`
}

// Pricing locates the price table.
type Pricing struct {
	File string `yaml:"file"`
}

// Server holds HTTP server settings.
type Server struct {
	Listen            string        `yaml:"listen"`
	ReadHeaderTimeout time.Duration `yaml:"read_header_timeout"`
	IdleTimeout       time.Duration `yaml:"idle_timeout"`
	MaxHeaderBytes    int           `yaml:"max_header_bytes"`
	MaxBodyBytes      int64         `yaml:"max_body_bytes"`
	DefaultDeadline   time.Duration `yaml:"default_deadline"`
	ShutdownTimeout   time.Duration `yaml:"shutdown_timeout"`
}

// Defaults holds values inherited by tenants that leave them unset.
type Defaults struct {
	Profile         string        `yaml:"profile"`
	MaxDeadline     time.Duration `yaml:"max_deadline"`
	MaxOutputTokens int           `yaml:"max_output_tokens"`
}

// Tenant is one API consumer and its limits.
type Tenant struct {
	Name                  string        `yaml:"name"`
	APIKeySHA256          []string      `yaml:"api_key_sha256"`
	Profile               string        `yaml:"profile"`
	MaxDeadline           time.Duration `yaml:"max_deadline"`
	MaxOutputTokens       int           `yaml:"max_output_tokens"`
	AllowStrategyOverride bool          `yaml:"allow_strategy_override"`
	AllowedModels         []string      `yaml:"allowed_models"`
}

// Allows reports whether the tenant may use model. An empty AllowedModels
// list allows every configured model.
func (t *Tenant) Allows(model string) bool {
	if len(t.AllowedModels) == 0 {
		return true
	}
	return slices.Contains(t.AllowedModels, model)
}

// Provider describes one upstream provider.
type Provider struct {
	Name           string `yaml:"name"`
	Type           string `yaml:"type"`
	BaseURL        string `yaml:"base_url"`
	APIKeyEnv      string `yaml:"api_key_env"`
	MaxTokensParam string `yaml:"max_tokens_param"`
	Fake           Fake   `yaml:"fake"`
	Replay         Replay `yaml:"replay"`
}

// Replay configures the replay provider, which serves recorded outputs with
// simulated latency. File and LatencyProfiles resolve against the directory of
// the configuration file; neither is read until the provider is built.
type Replay struct {
	File            string  `yaml:"file"`
	LatencyProfiles string  `yaml:"latency_profiles"`
	TimeScale       float64 `yaml:"time_scale"` // multiplies simulated latency, in (0, 1]
	Seed            uint64  `yaml:"seed"`       // seeds the latency jitter
}

// Fake configures the deterministic fake provider.
type Fake struct {
	Seed            int64              `yaml:"seed"`
	TTFT            time.Duration      `yaml:"ttft"`
	PerToken        time.Duration      `yaml:"per_token"`
	Jitter          float64            `yaml:"jitter"`
	OutputTokens    int                `yaml:"output_tokens"`
	ErrorRate       float64            `yaml:"error_rate"`
	ErrorKind       provider.ErrorKind `yaml:"error_kind"`
	RetryAfter      time.Duration      `yaml:"retry_after"`
	DisconnectRate  float64            `yaml:"disconnect_rate"`
	DisconnectAfter int                `yaml:"disconnect_after"`
	OmitUsageRate   float64            `yaml:"omit_usage_rate"`
}

// Model maps a client-facing model name to a provider model.
type Model struct {
	Name          string `yaml:"name"`
	Provider      string `yaml:"provider"`
	UpstreamModel string `yaml:"upstream_model"`
	PolicyName    string `yaml:"policy_name"`
	Strongest     bool   `yaml:"strongest"`

	// InputPerMTok and OutputPerMTok are optional USD prices per 1M tokens.
	// Set both or neither; both set override the price file.
	InputPerMTok  *float64 `yaml:"input_per_mtok"`
	OutputPerMTok *float64 `yaml:"output_per_mtok"`
	// ContextWindow is the model's context size in tokens; 0 means unknown
	// and disables the context check.
	ContextWindow  int           `yaml:"context_window"`
	LatencyPrior   LatencyPrior  `yaml:"latency_prior"`
	MaxConcurrency int           `yaml:"max_concurrency"`
	QueueSize      int           `yaml:"queue_size"`
	QueueTimeout   time.Duration `yaml:"queue_timeout"`
}

// LatencyPrior is the latency assumed for a model before any is observed.
type LatencyPrior struct {
	TTFTMS           float64 `yaml:"ttft_ms"`
	DecodeTokensPerS float64 `yaml:"decode_tokens_per_s"`
	Jitter           float64 `yaml:"jitter"`
}

// Parse strictly decodes one YAML document, applies defaults and validates.
func Parse(data []byte) (*Config, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)

	var c Config
	if err := dec.Decode(&c); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("config is empty")
		}
		return nil, fmt.Errorf("decode config: %w", err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("multiple YAML documents are not supported")
	}

	c.ApplyDefaults()
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

// Load reads and parses the configuration file at path. Relative file paths
// in the configuration (policy.path, pricing.file and each replay provider's
// file and latency_profiles) resolve against the directory of the file and
// are stored absolute.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	c, err := Parse(data)
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("resolve config path: %w", err)
	}
	c.Dir = filepath.Dir(abs)
	c.Policy.Path = c.ResolvePath(c.Policy.Path)
	c.Pricing.File = c.ResolvePath(c.Pricing.File)
	for i := range c.Providers {
		r := &c.Providers[i].Replay
		r.File = c.ResolvePath(r.File)
		r.LatencyProfiles = c.ResolvePath(r.LatencyProfiles)
	}
	return c, nil
}

// ResolvePath resolves a file path from the configuration: absolute paths are
// unchanged and relative ones are joined to Dir. An empty p stays empty. A
// Config without a Dir (built by Parse) leaves a relative path relative to the
// working directory.
func (c *Config) ResolvePath(p string) string {
	if p == "" || filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(c.Dir, p)
}

// Model returns the configured model with the given client-facing name.
func (c *Config) Model(name string) (Model, bool) {
	for _, m := range c.Models {
		if m.Name == name {
			return m, true
		}
	}
	return Model{}, false
}

// Provider returns the configured provider with the given name.
func (c *Config) Provider(name string) (Provider, bool) {
	for _, p := range c.Providers {
		if p.Name == name {
			return p, true
		}
	}
	return Provider{}, false
}
