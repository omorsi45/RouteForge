package config

import (
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

// defaultReplaySeed is the replay jitter seed, the seed of
// bench/latency_profiles.yaml.
const defaultReplaySeed = 20261026

// applyDefaults fills the unset reliability fields with the defaults listed
// under "reliability" in docs/operations.md. The retry budget floor of 3 lets an
// isolated request fail over.
func (r *Reliability) applyDefaults() {
	if r.MaxAttempts == 0 {
		r.MaxAttempts = 3
	}
	if r.BackoffBase == 0 {
		r.BackoffBase = 100 * time.Millisecond
	}
	if r.BackoffCap == 0 {
		r.BackoffCap = 2 * time.Second
	}
	if r.RetryBudgetRatio == 0 {
		r.RetryBudgetRatio = 0.2
	}
	if r.RetryBudgetWindow == 0 {
		r.RetryBudgetWindow = 10 * time.Second
	}
	if r.RetryBudgetMin == 0 {
		r.RetryBudgetMin = 3
	}

	b := &r.Breaker
	if b.ConsecutiveFailures == 0 {
		b.ConsecutiveFailures = 5
	}
	if b.ErrorRate == 0 {
		b.ErrorRate = 0.5
	}
	if b.MinRequests == 0 {
		b.MinRequests = 10
	}
	if b.Window == 0 {
		b.Window = 30 * time.Second
	}
	if b.OpenFor == 0 {
		b.OpenFor = 15 * time.Second
	}
	if b.HalfOpenProbes == 0 {
		b.HalfOpenProbes = 2
	}
}

// ApplyDefaults fills every unset field with its documented default. Zero
// values mean "unset", so it never overrides an explicit setting.
func (c *Config) ApplyDefaults() {
	s := &c.Server
	if s.Listen == "" {
		s.Listen = ":8080"
	}
	if s.ReadHeaderTimeout == 0 {
		s.ReadHeaderTimeout = 5 * time.Second
	}
	if s.IdleTimeout == 0 {
		s.IdleTimeout = 120 * time.Second
	}
	if s.MaxHeaderBytes == 0 {
		s.MaxHeaderBytes = 1 << 20
	}
	if s.MaxBodyBytes == 0 {
		s.MaxBodyBytes = 2 << 20
	}
	if s.DefaultDeadline == 0 {
		s.DefaultDeadline = 120 * time.Second
	}
	if s.ShutdownTimeout == 0 {
		s.ShutdownTimeout = 30 * time.Second
	}
	if s.MaxInFlight == 0 {
		s.MaxInFlight = 1024
	}

	d := &c.Defaults
	if d.Profile == "" {
		d.Profile = "balanced"
	}
	if d.MaxDeadline == 0 {
		d.MaxDeadline = s.DefaultDeadline
	}
	if d.MaxOutputTokens == 0 {
		d.MaxOutputTokens = 4096
	}

	c.Reliability.applyDefaults()

	if c.Accounting.FailMode == "" {
		c.Accounting.FailMode = FailModeClosed
	}
	if c.Accounting.UsageQueue == 0 {
		c.Accounting.UsageQueue = 10000
	}

	if c.Storage.Type == "" {
		c.Storage.Type = StorageMemory
	}
	if c.Limiter.Type == "" {
		c.Limiter.Type = LimiterMemory
	}

	for i := range c.Tenants {
		t := &c.Tenants[i]
		if t.Profile == "" {
			t.Profile = d.Profile
		}
		if t.MaxDeadline == 0 {
			t.MaxDeadline = d.MaxDeadline
		}
		if t.MaxOutputTokens == 0 {
			t.MaxOutputTokens = d.MaxOutputTokens
		}
	}

	for i := range c.Providers {
		p := &c.Providers[i]
		switch p.Type {
		case "openai":
			if p.BaseURL == "" {
				p.BaseURL = "https://api.openai.com/v1"
			}
			if p.MaxTokensParam == "" {
				p.MaxTokensParam = "max_completion_tokens"
			}
		case "anthropic":
			if p.BaseURL == "" {
				p.BaseURL = "https://api.anthropic.com"
			}
		case "fake":
			f := &p.Fake
			if f.OutputTokens == 0 {
				f.OutputTokens = 16
			}
			if f.ErrorKind == "" {
				f.ErrorKind = provider.KindUnavailable
			}
			if f.DisconnectAfter == 0 {
				f.DisconnectAfter = 1
			}
		case "replay":
			r := &p.Replay
			if r.TimeScale == 0 {
				r.TimeScale = 1
			}
			if r.Seed == 0 {
				r.Seed = defaultReplaySeed
			}
		}
	}

	for i := range c.Models {
		m := &c.Models[i]
		if m.UpstreamModel == "" {
			m.UpstreamModel = m.Name
		}
		if m.PolicyName == "" {
			m.PolicyName = m.Name
		}
		lp := &m.LatencyPrior
		if lp.TTFTMS == 0 {
			lp.TTFTMS = 500
		}
		if lp.DecodeTokensPerS == 0 {
			lp.DecodeTokensPerS = 50
		}
		if lp.Jitter == 0 {
			lp.Jitter = 0.1
		}
		if m.MaxConcurrency == 0 {
			m.MaxConcurrency = 64
		}
		if m.QueueSize == 0 {
			m.QueueSize = 128
		}
		if m.QueueTimeout == 0 {
			m.QueueTimeout = 2 * time.Second
		}
	}
}
