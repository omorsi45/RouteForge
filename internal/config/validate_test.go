package config_test

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
)

func ptr(v float64) *float64 { return &v }

func valid(t *testing.T) *config.Config {
	t.Helper()
	c, err := config.Load("testdata/minimal.yaml")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return c
}

func TestValidateAcceptsMinimal(t *testing.T) {
	if err := valid(t).Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestValidateRejects(t *testing.T) {
	addProvider := func(p config.Provider) func(*config.Config) {
		return func(c *config.Config) {
			c.Providers = append(c.Providers, p)
			c.ApplyDefaults()
		}
	}
	tests := []struct {
		name   string
		mutate func(*config.Config)
		want   string
	}{
		{"no tenants", func(c *config.Config) { c.Tenants = nil }, "tenants: at least one tenant is required"},
		{"no providers", func(c *config.Config) { c.Providers = nil }, "providers: at least one provider is required"},
		{"no models", func(c *config.Config) { c.Models = nil }, "models: at least one model is required"},
		{"bad tenant name", func(c *config.Config) { c.Tenants[0].Name = "Bad Name" }, "tenants[0].name: must match"},
		{"bad hex", func(c *config.Config) { c.Tenants[0].APIKeySHA256 = []string{"ABC"} }, "tenants[0].api_key_sha256[0]: must be 64 lowercase hex characters"},
		{"no digests", func(c *config.Config) { c.Tenants[0].APIKeySHA256 = nil }, "tenants[0].api_key_sha256: at least one digest is required"},
		{"dup tenant name", func(c *config.Config) {
			c.Tenants = append(c.Tenants, config.Tenant{Name: "tenant-a", APIKeySHA256: []string{strings.Repeat("b", 64)}})
			c.ApplyDefaults()
		}, "tenants[1].name: duplicates tenants[0]"},
		{"dup digest", func(c *config.Config) {
			c.Tenants = append(c.Tenants, config.Tenant{Name: "tenant-b", APIKeySHA256: []string{c.Tenants[0].APIKeySHA256[0]}})
			c.ApplyDefaults()
		}, "duplicates tenants[0].api_key_sha256[0]"},
		{"bad profile", func(c *config.Config) { c.Tenants[0].Profile = "fast" }, `tenants[0].profile: must be one of strict, balanced, economy`},
		{"zero tenant deadline", func(c *config.Config) { c.Tenants[0].MaxDeadline = -1 }, "tenants[0].max_deadline: must be positive"},
		{"zero tenant tokens", func(c *config.Config) { c.Tenants[0].MaxOutputTokens = -1 }, "tenants[0].max_output_tokens: must be positive"},
		{"unknown allowed model", func(c *config.Config) { c.Tenants[0].AllowedModels = []string{"nope"} }, `tenants[0].allowed_models[0]: unknown model "nope"`},
		{"bad provider type", func(c *config.Config) { c.Providers[0].Type = "bedrock" }, `providers[0].type: must be one of openai, anthropic, fake`},
		{"bad provider name", func(c *config.Config) { c.Providers[0].Name = "UP" }, "providers[0].name: must match"},
		{"anthropic needs key env", addProvider(config.Provider{Name: "anth", Type: "anthropic", BaseURL: "https://api.anthropic.com"}), "providers[1].api_key_env: required for anthropic"},
		{"bad env name", addProvider(config.Provider{Name: "oa", Type: "openai", APIKeyEnv: "openai-key"}), "providers[1].api_key_env: must match ^[A-Z_][A-Z0-9_]*$"},
		{"bad base url", addProvider(config.Provider{Name: "oa", Type: "openai", BaseURL: "ftp://x"}), "providers[1].base_url: must be an absolute http or https URL"},
		{"bad max tokens param", addProvider(config.Provider{Name: "oa", Type: "openai", MaxTokensParam: "limit"}), "providers[1].max_tokens_param: must be one of max_completion_tokens, max_tokens"},
		{"fake rate", func(c *config.Config) { c.Providers[0].Fake.ErrorRate = 1.5 }, "providers[0].fake.error_rate: must be between 0 and 1"},
		{"fake jitter", func(c *config.Config) { c.Providers[0].Fake.Jitter = 1 }, "providers[0].fake.jitter: must be in [0, 1)"},
		{"fake rate nan", func(c *config.Config) { c.Providers[0].Fake.ErrorRate = math.NaN() }, "providers[0].fake.error_rate: must be between 0 and 1"},
		{"fake disconnect rate nan", func(c *config.Config) { c.Providers[0].Fake.DisconnectRate = math.NaN() }, "providers[0].fake.disconnect_rate: must be between 0 and 1"},
		{"fake omit usage rate nan", func(c *config.Config) { c.Providers[0].Fake.OmitUsageRate = math.NaN() }, "providers[0].fake.omit_usage_rate: must be between 0 and 1"},
		{"fake jitter nan", func(c *config.Config) { c.Providers[0].Fake.Jitter = math.NaN() }, "providers[0].fake.jitter: must be in [0, 1)"},
		{"fake output tokens", func(c *config.Config) { c.Providers[0].Fake.OutputTokens = -1 }, "providers[0].fake.output_tokens: must be at least 1"},
		{"fake disconnect after", func(c *config.Config) { c.Providers[0].Fake.DisconnectAfter = -1 }, "providers[0].fake.disconnect_after: must be at least 1"},
		{"fake negative ttft", func(c *config.Config) { c.Providers[0].Fake.TTFT = -1 }, "providers[0].fake.ttft: must not be negative"},
		{"fake kind", func(c *config.Config) { c.Providers[0].Fake.ErrorKind = "boom" }, `providers[0].fake.error_kind: unknown kind "boom"`},
		{"reserved model name", func(c *config.Config) { c.Models[0].Name = "routeforge-auto" }, "models[0].name: routeforge-auto is reserved"},
		{"bad model name", func(c *config.Config) { c.Models[0].Name = "-x" }, "models[0].name: must match"},
		{"dup model name", func(c *config.Config) { c.Models[1].Name = c.Models[0].Name }, "models[1].name: duplicates models[0]"},
		{"model unknown provider", func(c *config.Config) { c.Models[0].Provider = "ghost" }, `models[0].provider: unknown provider "ghost"`},
		{"dup policy name", func(c *config.Config) { c.Models[1].PolicyName = c.Models[0].PolicyName }, `models[1].policy_name: duplicates models[0]`},
		{"only output price", func(c *config.Config) { c.Models[1].OutputPerMTok = ptr(1.0) }, "models[1].input_per_mtok: required when output_per_mtok is set"},
		{"negative input price", func(c *config.Config) { c.Models[0].InputPerMTok, c.Models[0].OutputPerMTok = ptr(-1.0), ptr(1.0) }, "models[0].input_per_mtok: must be finite and not negative"},
		{"negative output price", func(c *config.Config) { c.Models[0].InputPerMTok, c.Models[0].OutputPerMTok = ptr(1.0), ptr(-0.5) }, "models[0].output_per_mtok: must be finite and not negative"},
		{"nan input price", func(c *config.Config) {
			c.Models[0].InputPerMTok, c.Models[0].OutputPerMTok = ptr(math.NaN()), ptr(1.0)
		}, "models[0].input_per_mtok: must be finite and not negative"},
		{"inf output price", func(c *config.Config) {
			c.Models[0].InputPerMTok, c.Models[0].OutputPerMTok = ptr(1.0), ptr(math.Inf(1))
		}, "models[0].output_per_mtok: must be finite and not negative"},
		{"negative context window", func(c *config.Config) { c.Models[0].ContextWindow = -1 }, "models[0].context_window: must not be negative"},
		{"zero ttft prior", func(c *config.Config) { c.Models[0].LatencyPrior.TTFTMS = 0 }, "models[0].latency_prior.ttft_ms: must be positive"},
		{"negative ttft prior", func(c *config.Config) { c.Models[0].LatencyPrior.TTFTMS = -5 }, "models[0].latency_prior.ttft_ms: must be positive"},
		{"nan ttft prior", func(c *config.Config) { c.Models[0].LatencyPrior.TTFTMS = math.NaN() }, "models[0].latency_prior.ttft_ms: must be positive"},
		{"inf decode prior", func(c *config.Config) { c.Models[0].LatencyPrior.DecodeTokensPerS = math.Inf(1) }, "models[0].latency_prior.decode_tokens_per_s: must be positive"},
		{"zero decode prior", func(c *config.Config) { c.Models[1].LatencyPrior.DecodeTokensPerS = 0 }, "models[1].latency_prior.decode_tokens_per_s: must be positive"},
		{"prior jitter one", func(c *config.Config) { c.Models[0].LatencyPrior.Jitter = 1 }, "models[0].latency_prior.jitter: must be in [0, 1)"},
		{"prior jitter negative", func(c *config.Config) { c.Models[0].LatencyPrior.Jitter = -0.1 }, "models[0].latency_prior.jitter: must be in [0, 1)"},
		{"prior jitter nan", func(c *config.Config) { c.Models[0].LatencyPrior.Jitter = math.NaN() }, "models[0].latency_prior.jitter: must be in [0, 1)"},
		{"zero max concurrency", func(c *config.Config) { c.Models[0].MaxConcurrency = 0 }, "models[0].max_concurrency: must be at least 1"},
		{"negative queue size", func(c *config.Config) { c.Models[0].QueueSize = -1 }, "models[0].queue_size: must not be negative"},
		{"zero queue timeout", func(c *config.Config) { c.Models[0].QueueTimeout = 0 }, "models[0].queue_timeout: must be positive"},
		{"negative queue timeout", func(c *config.Config) { c.Models[0].QueueTimeout = -time.Second }, "models[0].queue_timeout: must be positive"},
		{"zero deadline", func(c *config.Config) { c.Server.DefaultDeadline = 0 }, "server.default_deadline: must be positive"},
		{"empty listen", func(c *config.Config) { c.Server.Listen = "" }, "server.listen: must not be empty"},
		{"zero body bytes", func(c *config.Config) { c.Server.MaxBodyBytes = 0 }, "server.max_body_bytes: must be positive"},
		{"zero header bytes", func(c *config.Config) { c.Server.MaxHeaderBytes = 0 }, "server.max_header_bytes: must be positive"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := valid(t)
			tc.mutate(c)
			err := c.Validate()
			if err == nil {
				t.Fatal("Validate returned nil")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}

func TestAdminKeyDistinctFromTenant(t *testing.T) {
	t.Run("no admin keys is valid", func(t *testing.T) {
		if err := valid(t).Validate(); err != nil {
			t.Fatalf("Validate: %v", err)
		}
	})
	t.Run("a distinct admin digest is valid", func(t *testing.T) {
		c := valid(t)
		c.Admin.APIKeySHA256 = []string{strings.Repeat("b", 64), strings.Repeat("c", 64)}
		if err := c.Validate(); err != nil {
			t.Fatalf("Validate: %v", err)
		}
	})

	rejects := []struct {
		name   string
		digest func(c *config.Config) []string
		want   string
	}{
		{"equal to a tenant digest", func(c *config.Config) []string {
			return []string{c.Tenants[0].APIKeySHA256[0]}
		}, "admin.api_key_sha256[0]: an admin key may not also be a tenant key (duplicates tenants[0].api_key_sha256[0])"},
		{"equal to a tenant digest after a valid one", func(c *config.Config) []string {
			return []string{strings.Repeat("b", 64), c.Tenants[0].APIKeySHA256[0]}
		}, "admin.api_key_sha256[1]: an admin key may not also be a tenant key"},
		{"malformed", func(*config.Config) []string { return []string{"ABC"} },
			"admin.api_key_sha256[0]: must be 64 lowercase hex characters"},
		{"repeated", func(*config.Config) []string {
			return []string{strings.Repeat("b", 64), strings.Repeat("b", 64)}
		}, "admin.api_key_sha256[1]: duplicates admin.api_key_sha256[0]"},
	}
	for _, tc := range rejects {
		t.Run(tc.name, func(t *testing.T) {
			c := valid(t)
			c.Admin.APIKeySHA256 = tc.digest(c)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestValidateModelPricePair(t *testing.T) {
	c := valid(t)
	c.Models[0].InputPerMTok = ptr(1.0)
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), "models[0].output_per_mtok: required when input_per_mtok is set") {
		t.Fatalf("only input_per_mtok: err = %v", err)
	}

	c.Models[0].OutputPerMTok = ptr(2.0)
	if err := c.Validate(); err != nil {
		t.Fatalf("both prices set: %v", err)
	}

	c.Models[0].InputPerMTok, c.Models[0].OutputPerMTok = ptr(0), ptr(0)
	if err := c.Validate(); err != nil {
		t.Fatalf("both prices zero: %v", err)
	}
}

func TestValidateReplayProvider(t *testing.T) {
	withReplay := func(c *config.Config, mutate func(*config.Replay)) {
		c.Providers[0] = config.Provider{
			Name: "replay",
			Type: "replay",
			Replay: config.Replay{
				File:            "test.jsonl",
				LatencyProfiles: "profiles.yaml",
				TimeScale:       0.01,
				Seed:            1,
			},
		}
		if mutate != nil {
			mutate(&c.Providers[0].Replay)
		}
		for i := range c.Models {
			c.Models[i].Provider = "replay"
		}
	}

	t.Run("accepted without a base url or key", func(t *testing.T) {
		c := valid(t)
		withReplay(c, nil)
		if err := c.Validate(); err != nil {
			t.Fatalf("Validate: %v", err)
		}
	})
	t.Run("time scale of one is accepted", func(t *testing.T) {
		c := valid(t)
		withReplay(c, func(r *config.Replay) { r.TimeScale = 1 })
		if err := c.Validate(); err != nil {
			t.Fatalf("Validate: %v", err)
		}
	})

	rejects := []struct {
		name   string
		mutate func(*config.Replay)
		want   string
	}{
		{"zero time scale", func(r *config.Replay) { r.TimeScale = 0 }, "providers[0].replay.time_scale: must be in (0, 1]"},
		{"negative time scale", func(r *config.Replay) { r.TimeScale = -0.5 }, "providers[0].replay.time_scale: must be in (0, 1]"},
		{"time scale above one", func(r *config.Replay) { r.TimeScale = 1.01 }, "providers[0].replay.time_scale: must be in (0, 1]"},
		{"nan time scale", func(r *config.Replay) { r.TimeScale = math.NaN() }, "providers[0].replay.time_scale: must be in (0, 1]"},
		{"no file", func(r *config.Replay) { r.File = "" }, "providers[0].replay.file: required for replay"},
		{"no latency profiles", func(r *config.Replay) { r.LatencyProfiles = "" }, "providers[0].replay.latency_profiles: required for replay"},
	}
	for _, tc := range rejects {
		t.Run(tc.name, func(t *testing.T) {
			c := valid(t)
			withReplay(c, tc.mutate)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestServerMaxInFlight(t *testing.T) {
	if got := valid(t).Server.MaxInFlight; got != 1024 {
		t.Errorf("default MaxInFlight = %d, want 1024", got)
	}

	c, err := config.Parse([]byte("server: {max_inflight: 8}\n" + minimalDoc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if c.Server.MaxInFlight != 8 {
		t.Errorf("MaxInFlight = %d, want 8", c.Server.MaxInFlight)
	}

	for _, n := range []int{0, -1} {
		c := valid(t)
		c.Server.MaxInFlight = n
		err := c.Validate()
		if err == nil || !strings.Contains(err.Error(), "server.max_inflight: must be at least 1") {
			t.Errorf("max_inflight %d: Validate = %v, want server.max_inflight: must be at least 1", n, err)
		}
	}

	// A negative value in the file is rejected too, not replaced by the default.
	_, err = config.Parse([]byte("server: {max_inflight: -3}\n" + minimalDoc))
	if err == nil || !strings.Contains(err.Error(), "server.max_inflight: must be at least 1") {
		t.Errorf("Parse with max_inflight -3: err = %v", err)
	}
}

func TestReliabilityDefaults(t *testing.T) {
	want := config.Reliability{
		MaxAttempts:       3,
		BackoffBase:       100 * time.Millisecond,
		BackoffCap:        2 * time.Second,
		RetryBudgetRatio:  0.2,
		RetryBudgetWindow: 10 * time.Second,
		RetryBudgetMin:    3,
		Breaker: config.Breaker{
			ConsecutiveFailures: 5,
			ErrorRate:           0.5,
			MinRequests:         10,
			Window:              30 * time.Second,
			OpenFor:             15 * time.Second,
			HalfOpenProbes:      2,
		},
	}
	if got := valid(t).Reliability; got != want {
		t.Fatalf("Reliability = %+v, want %+v", got, want)
	}
}

func TestParseReliabilitySection(t *testing.T) {
	doc := `
reliability:
  max_attempts: 5
  backoff_base: 50ms
  retry_budget_ratio: 0.5
  breaker:
    consecutive_failures: 8
    open_for: 1m
` + minimalDoc
	c, err := config.Parse([]byte(doc))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	r := c.Reliability
	if r.MaxAttempts != 5 || r.BackoffBase != 50*time.Millisecond || r.RetryBudgetRatio != 0.5 {
		t.Errorf("explicit values lost: %+v", r)
	}
	if r.Breaker.ConsecutiveFailures != 8 || r.Breaker.OpenFor != time.Minute {
		t.Errorf("explicit breaker values lost: %+v", r.Breaker)
	}
	// Fields left out keep their defaults, one by one.
	if r.BackoffCap != 2*time.Second || r.RetryBudgetWindow != 10*time.Second || r.RetryBudgetMin != 3 {
		t.Errorf("defaults missing: %+v", r)
	}
	if r.Breaker.ErrorRate != 0.5 || r.Breaker.MinRequests != 10 || r.Breaker.Window != 30*time.Second || r.Breaker.HalfOpenProbes != 2 {
		t.Errorf("breaker defaults missing: %+v", r.Breaker)
	}

	_, err = config.Parse([]byte("reliability: {retries: 2}\n" + minimalDoc))
	if err == nil || !strings.Contains(err.Error(), "retries") {
		t.Errorf("Parse with an unknown reliability field: err = %v", err)
	}
}

func TestValidateReliability(t *testing.T) {
	t.Run("boundary values are accepted", func(t *testing.T) {
		for _, attempts := range []int{1, 10} {
			c := valid(t)
			c.Reliability.MaxAttempts = attempts
			c.Reliability.RetryBudgetRatio = 1
			c.Reliability.Breaker.ErrorRate = 1
			if err := c.Validate(); err != nil {
				t.Errorf("max_attempts %d: %v", attempts, err)
			}
		}
	})

	rejects := []struct {
		name   string
		mutate func(*config.Reliability)
		want   string
	}{
		{"zero attempts", func(r *config.Reliability) { r.MaxAttempts = 0 }, "reliability.max_attempts: must be between 1 and 10"},
		{"negative attempts", func(r *config.Reliability) { r.MaxAttempts = -1 }, "reliability.max_attempts: must be between 1 and 10"},
		{"too many attempts", func(r *config.Reliability) { r.MaxAttempts = 11 }, "reliability.max_attempts: must be between 1 and 10"},
		{"zero backoff base", func(r *config.Reliability) { r.BackoffBase = 0 }, "reliability.backoff_base: must be positive"},
		{"negative backoff cap", func(r *config.Reliability) { r.BackoffCap = -time.Second }, "reliability.backoff_cap: must be positive"},
		{"zero ratio", func(r *config.Reliability) { r.RetryBudgetRatio = 0 }, "reliability.retry_budget_ratio: must be in (0, 1]"},
		{"ratio above one", func(r *config.Reliability) { r.RetryBudgetRatio = 1.01 }, "reliability.retry_budget_ratio: must be in (0, 1]"},
		{"nan ratio", func(r *config.Reliability) { r.RetryBudgetRatio = math.NaN() }, "reliability.retry_budget_ratio: must be in (0, 1]"},
		{"zero budget window", func(r *config.Reliability) { r.RetryBudgetWindow = 0 }, "reliability.retry_budget_window: must be positive"},
		{"zero budget min", func(r *config.Reliability) { r.RetryBudgetMin = 0 }, "reliability.retry_budget_min: must be positive"},
		{"zero consecutive failures", func(r *config.Reliability) { r.Breaker.ConsecutiveFailures = 0 }, "reliability.breaker.consecutive_failures: must be positive"},
		{"zero error rate", func(r *config.Reliability) { r.Breaker.ErrorRate = 0 }, "reliability.breaker.error_rate: must be in (0, 1]"},
		{"error rate above one", func(r *config.Reliability) { r.Breaker.ErrorRate = 2 }, "reliability.breaker.error_rate: must be in (0, 1]"},
		{"nan error rate", func(r *config.Reliability) { r.Breaker.ErrorRate = math.NaN() }, "reliability.breaker.error_rate: must be in (0, 1]"},
		{"zero min requests", func(r *config.Reliability) { r.Breaker.MinRequests = 0 }, "reliability.breaker.min_requests: must be positive"},
		{"zero breaker window", func(r *config.Reliability) { r.Breaker.Window = 0 }, "reliability.breaker.window: must be positive"},
		{"negative open for", func(r *config.Reliability) { r.Breaker.OpenFor = -time.Second }, "reliability.breaker.open_for: must be positive"},
		{"zero probes", func(r *config.Reliability) { r.Breaker.HalfOpenProbes = 0 }, "reliability.breaker.half_open_probes: must be positive"},
	}
	for _, tc := range rejects {
		t.Run(tc.name, func(t *testing.T) {
			c := valid(t)
			tc.mutate(&c.Reliability)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate = %v, want an error containing %q", err, tc.want)
			}
		})
	}
}

func TestValidateJoinsAllErrors(t *testing.T) {
	c := valid(t)
	c.Tenants[0].Profile = "fast"
	c.Server.DefaultDeadline = 0
	err := c.Validate()
	if err == nil {
		t.Fatal("Validate returned nil")
	}
	for _, want := range []string{"tenants[0].profile", "server.default_deadline"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q missing %q", err.Error(), want)
		}
	}
}
