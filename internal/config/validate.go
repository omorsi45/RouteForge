package config

import (
	"errors"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"strconv"
	"strings"
)

var (
	nameRE      = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)
	modelNameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$`)
	hexDigestRE = regexp.MustCompile(`^[0-9a-f]{64}$`)
	envNameRE   = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
)

var (
	providerTypes   = []string{"openai", "anthropic", "fake", "replay"}
	maxTokensParams = []string{"max_completion_tokens", "max_tokens"}
)

// maxAttemptsLimit is the most attempts per request that reliability.max_attempts
// may allow.
const maxAttemptsLimit = 10

// Validate checks the configuration and returns every problem found, joined,
// each as "<yaml path>: <problem>".
func (c *Config) Validate() error {
	var errs []error
	add := func(path, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", path, fmt.Sprintf(format, args...)))
	}

	c.validateServer(add)
	c.validateReliability(add)
	c.validateAccounting(add)

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
	tenantDigests := c.validateTenants(add)
	c.validateAdmin(tenantDigests, add)

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
	if s.MaxInFlight < 1 {
		add("server.max_inflight", "must be at least 1")
	}
}

func (c *Config) validateReliability(add addFunc) {
	r := c.Reliability
	if r.MaxAttempts < 1 || r.MaxAttempts > maxAttemptsLimit {
		add("reliability.max_attempts", "must be between 1 and %d", maxAttemptsLimit)
	}
	for _, d := range []struct {
		path string
		ok   bool
	}{
		{"reliability.backoff_base", r.BackoffBase > 0},
		{"reliability.backoff_cap", r.BackoffCap > 0},
		{"reliability.retry_budget_window", r.RetryBudgetWindow > 0},
		{"reliability.retry_budget_min", r.RetryBudgetMin > 0},
		{"reliability.breaker.consecutive_failures", r.Breaker.ConsecutiveFailures > 0},
		{"reliability.breaker.min_requests", r.Breaker.MinRequests > 0},
		{"reliability.breaker.window", r.Breaker.Window > 0},
		{"reliability.breaker.open_for", r.Breaker.OpenFor > 0},
		{"reliability.breaker.half_open_probes", r.Breaker.HalfOpenProbes > 0},
	} {
		if !d.ok {
			add(d.path, "must be positive")
		}
	}
	// Written as the accepted range so that NaN is rejected.
	for _, f := range []struct {
		path string
		ok   bool
	}{
		{"reliability.retry_budget_ratio", r.RetryBudgetRatio > 0 && r.RetryBudgetRatio <= 1},
		{"reliability.breaker.error_rate", r.Breaker.ErrorRate > 0 && r.Breaker.ErrorRate <= 1},
	} {
		if !f.ok {
			add(f.path, "must be in (0, 1]")
		}
	}
}

func (c *Config) validateAccounting(add addFunc) {
	if !oneOf(c.Accounting.FailMode, []string{FailModeClosed, FailModeOpen}) {
		add("accounting.fail_mode", "must be one of %s, %s", FailModeClosed, FailModeOpen)
	}
	if c.Accounting.UsageQueue < 1 {
		add("accounting.usage_queue", "must be at least 1")
	}
}

// Budgets are exact amounts of nano-USD, so a budget must survive a trip
// through float64 unchanged. accounting.USDToNanos applies the same rule when
// the gateway converts the budget; config cannot import accounting, which
// imports config, and a test keeps the two in step.
const (
	nanosPerUSD    = 1e9
	maxBudgetNanos = 1 << 50 // about 1.1 million USD; above it float64 product rounding is not exact
)

// budgetProblem says why usd is not a valid daily budget, or returns "".
func budgetProblem(usd float64) string {
	if math.IsNaN(usd) || math.IsInf(usd, 0) || usd < 0 {
		return "must be finite and not negative"
	}
	scaled := math.Round(usd * nanosPerUSD)
	if scaled > maxBudgetNanos {
		return "must be at most " + strconv.FormatFloat(float64(maxBudgetNanos)/nanosPerUSD, 'f', -1, 64) + " USD"
	}
	if float64(int64(scaled))/nanosPerUSD != usd {
		return "must be a whole number of nano-USD (at most 9 decimals)"
	}
	return ""
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

		if p.Type != "fake" && p.Type != "replay" {
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
		if p.Type == "replay" {
			validateReplay(base+".replay", p.Replay, add)
		}
	}
}

// validateReplay checks that the replay paths are set. It never reads them: the
// records file is large third-party data that a checkout does not carry.
func validateReplay(base string, r Replay, add addFunc) {
	if r.File == "" {
		add(base+".file", "required for replay")
	}
	if r.LatencyProfiles == "" {
		add(base+".latency_profiles", "required for replay")
	}
	if !(r.TimeScale > 0 && r.TimeScale <= 1) {
		add(base+".time_scale", "must be in (0, 1]")
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
		{"over_report_tokens", f.OverReportTokens < 0},
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

		validateModelRouting(base, m, add)
	}
}

func validateModelRouting(base string, m Model, add addFunc) {
	switch {
	case m.InputPerMTok != nil && m.OutputPerMTok == nil:
		add(base+".output_per_mtok", "required when input_per_mtok is set")
	case m.InputPerMTok == nil && m.OutputPerMTok != nil:
		add(base+".input_per_mtok", "required when output_per_mtok is set")
	}
	for _, r := range []struct {
		path string
		v    *float64
	}{
		{"input_per_mtok", m.InputPerMTok},
		{"output_per_mtok", m.OutputPerMTok},
	} {
		if r.v != nil && !(*r.v >= 0 && !math.IsInf(*r.v, 1)) {
			add(base+"."+r.path, "must be finite and not negative")
		}
	}

	if m.ContextWindow < 0 {
		add(base+".context_window", "must not be negative")
	}
	lp := m.LatencyPrior
	for _, v := range []struct {
		path string
		v    float64
	}{
		{"ttft_ms", lp.TTFTMS},
		{"decode_tokens_per_s", lp.DecodeTokensPerS},
	} {
		if !(v.v > 0 && !math.IsInf(v.v, 1)) {
			add(base+".latency_prior."+v.path, "must be positive")
		}
	}
	if !(lp.Jitter >= 0 && lp.Jitter < 1) {
		add(base+".latency_prior.jitter", "must be in [0, 1)")
	}
	if m.MaxConcurrency < 1 {
		add(base+".max_concurrency", "must be at least 1")
	}
	if m.QueueSize < 0 {
		add(base+".queue_size", "must not be negative")
	}
	if m.QueueTimeout <= 0 {
		add(base+".queue_timeout", "must be positive")
	}
}

// validateTenants checks the tenants and returns the YAML path of every
// well-formed key digest, by digest.
func (c *Config) validateTenants(add addFunc) map[string]string {
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
		if t.RPM < 0 {
			add(base+".rpm", "must not be negative")
		}
		if t.TPM < 0 {
			add(base+".tpm", "must not be negative")
		}
		if problem := budgetProblem(t.DailyBudgetUSD); problem != "" {
			add(base+".daily_budget_usd", "%s", problem)
		}
		for k, m := range t.AllowedModels {
			if _, ok := c.Model(m); !ok {
				add(fmt.Sprintf("%s.allowed_models[%d]", base, k), "unknown model %q", m)
			}
		}
	}
	return digests
}

// validateAdmin checks the admin key digests: well formed, unique, and not
// also a tenant key, so an admin key never opens a tenant's routes.
func (c *Config) validateAdmin(tenantDigests map[string]string, add addFunc) {
	seen := map[string]string{}
	for i, d := range c.Admin.APIKeySHA256 {
		path := fmt.Sprintf("admin.api_key_sha256[%d]", i)
		if !hexDigestRE.MatchString(d) {
			add(path, "must be 64 lowercase hex characters")
			continue
		}
		if tenant, dup := tenantDigests[d]; dup {
			add(path, "an admin key may not also be a tenant key (duplicates %s)", tenant)
			continue
		}
		if first, dup := seen[d]; dup {
			add(path, "duplicates %s", first)
			continue
		}
		seen[d] = path
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
