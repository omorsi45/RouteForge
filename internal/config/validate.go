package config

import (
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

var (
	nameRE      = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)
	modelNameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)
	hexDigestRE = regexp.MustCompile(`^[0-9a-f]{64}$`)
	envNameRE   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
)

var (
	providerTypes   = []string{"openai", "anthropic", "fake"}
	maxTokensParams = []string{"max_completion_tokens", "max_tokens"}
)

// Validate checks the configuration and returns every problem found, joined,
// each as "<yaml path>: <problem>".
func (c *Config) Validate() error {
	var errs []error
	add := func(path, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", path, fmt.Sprintf(format, args...)))
	}

	c.validateServer(add)

	if len(c.Providers) == 0 {
		add("providers", "at least one provider is required")
	}
	if len(c.Models) == 0 {
		add("models", "at least one model is required")
	}
	if len(c.Tenants) == 0 {
		add("tenants", "at least one tenant is required")
	}

	c.validateProviders(add)
	c.validateModels(add)
	c.validateTenants(add)

	return errors.Join(errs...)
}

type addFunc func(path, format string, args ...any)

func (c *Config) validateServer(add addFunc) {
	s := c.Server
	if s.Listen == "" {
		add("server.listen", "must not be empty")
	}
	for _, d := range []struct {
		path string
		ok   bool
	}{
		{"server.read_header_timeout", s.ReadHeaderTimeout > 0},
		{"server.idle_timeout", s.IdleTimeout > 0},
		{"server.default_deadline", s.DefaultDeadline > 0},
		{"server.shutdown_timeout", s.ShutdownTimeout > 0},
		{"server.max_header_bytes", s.MaxHeaderBytes > 0},
		{"server.max_body_bytes", s.MaxBodyBytes > 0},
	} {
		if !d.ok {
			add(d.path, "must be positive")
		}
	}
}

func (c *Config) validateProviders(add addFunc) {
	seen := map[string]int{}
	for i, p := range c.Providers {
		base := fmt.Sprintf("providers[%d]", i)
		if !nameRE.MatchString(p.Name) {
			add(base+".name", "must match %s", nameRE)
		} else if j, dup := seen[p.Name]; dup {
			add(base+".name", "duplicates providers[%d]", j)
		} else {
			seen[p.Name] = i
		}

		if !oneOf(p.Type, providerTypes) {
			add(base+".type", "must be one of %s", strings.Join(providerTypes, ", "))
			continue
		}

		if p.Type != "fake" {
			if u, err := url.Parse(p.BaseURL); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
				add(base+".base_url", "must be an absolute http or https URL")
			}
			switch {
			case p.APIKeyEnv == "" && p.Type == "anthropic":
				add(base+".api_key_env", "required for anthropic")
			case p.APIKeyEnv != "" && !envNameRE.MatchString(p.APIKeyEnv):
				add(base+".api_key_env", "must match %s", envNameRE)
			}
		}
		if p.Type == "openai" && !oneOf(p.MaxTokensParam, maxTokensParams) {
			add(base+".max_tokens_param", "must be one of %s", strings.Join(maxTokensParams, ", "))
		}
		if p.Type == "fake" {
			validateFake(base+".fake", p.Fake, add)
		}
	}
}

func validateFake(base string, f Fake, add addFunc) {
	for _, r := range []struct {
		path string
		v    float64
	}{
		{"error_rate", f.ErrorRate},
		{"disconnect_rate", f.DisconnectRate},
		{"omit_usage_rate", f.OmitUsageRate},
	} {
		if !(r.v >= 0 && r.v <= 1) {
			add(base+"."+r.path, "must be between 0 and 1")
		}
	}
	if !(f.Jitter >= 0 && f.Jitter < 1) {
		add(base+".jitter", "must be in [0, 1)")
	}
	if f.OutputTokens < 1 {
		add(base+".output_tokens", "must be at least 1")
	}
	if f.DisconnectAfter < 1 {
		add(base+".disconnect_after", "must be at least 1")
	}
	for _, d := range []struct {
		path string
		neg  bool
	}{
		{"ttft", f.TTFT < 0},
		{"per_token", f.PerToken < 0},
		{"retry_after", f.RetryAfter < 0},
	} {
		if d.neg {
			add(base+"."+d.path, "must not be negative")
		}
	}
	if !f.ErrorKind.Valid() {
		add(base+".error_kind", "unknown kind %q", string(f.ErrorKind))
	}
}

func (c *Config) validateModels(add addFunc) {
	names := map[string]int{}
	policies := map[string]int{}
	for i, m := range c.Models {
		base := fmt.Sprintf("models[%d]", i)
		switch {
		case m.Name == AutoModel:
			add(base+".name", "%s is reserved", AutoModel)
		case !modelNameRE.MatchString(m.Name):
			add(base+".name", "must match %s", modelNameRE)
		default:
			if j, dup := names[m.Name]; dup {
				add(base+".name", "duplicates models[%d]", j)
			} else {
				names[m.Name] = i
			}
		}

		if _, ok := c.Provider(m.Provider); !ok {
			add(base+".provider", "unknown provider %q", m.Provider)
		}

		if j, dup := policies[m.PolicyName]; dup {
			add(base+".policy_name", "duplicates models[%d]", j)
		} else {
			policies[m.PolicyName] = i
		}
	}
}

func (c *Config) validateTenants(add addFunc) {
	names := map[string]int{}
	digests := map[string]string{}
	for i, t := range c.Tenants {
		base := fmt.Sprintf("tenants[%d]", i)
		if !nameRE.MatchString(t.Name) {
			add(base+".name", "must match %s", nameRE)
		} else if j, dup := names[t.Name]; dup {
			add(base+".name", "duplicates tenants[%d]", j)
		} else {
			names[t.Name] = i
		}

		if len(t.APIKeySHA256) == 0 {
			add(base+".api_key_sha256", "at least one digest is required")
		}
		for k, d := range t.APIKeySHA256 {
			path := fmt.Sprintf("%s.api_key_sha256[%d]", base, k)
			if !hexDigestRE.MatchString(d) {
				add(path, "must be 64 lowercase hex characters")
				continue
			}
			if first, dup := digests[d]; dup {
				add(path, "duplicates %s", first)
			} else {
				digests[d] = path
			}
		}

		if !oneOf(t.Profile, Profiles) {
			add(base+".profile", "must be one of %s", strings.Join(Profiles, ", "))
		}
		if t.MaxDeadline <= 0 {
			add(base+".max_deadline", "must be positive")
		}
		if t.MaxOutputTokens <= 0 {
			add(base+".max_output_tokens", "must be positive")
		}
		for k, m := range t.AllowedModels {
			if _, ok := c.Model(m); !ok {
				add(fmt.Sprintf("%s.allowed_models[%d]", base, k), "unknown model %q", m)
			}
		}
	}
}

func oneOf(v string, allowed []string) bool {
	for _, a := range allowed {
		if v == a {
			return true
		}
	}
	return false
}
