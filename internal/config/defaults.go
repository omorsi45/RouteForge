package config

import (
	"time"

	"github.com/omorsi45/routeforge/internal/provider"
)

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
	}
}
