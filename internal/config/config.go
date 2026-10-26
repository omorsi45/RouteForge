// Package config loads, defaults and validates the gateway configuration.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
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
	Server    Server     `yaml:"server"`
	Defaults  Defaults   `yaml:"defaults"`
	Tenants   []Tenant   `yaml:"tenants"`
	Providers []Provider `yaml:"providers"`
	Models    []Model    `yaml:"models"`
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

// Load reads and parses the configuration file at path.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return Parse(data)
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
