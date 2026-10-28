package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/api"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/health"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/anthropic"
	"github.com/omorsi45/routeforge/internal/provider/fake"
	"github.com/omorsi45/routeforge/internal/provider/openai"
	"github.com/omorsi45/routeforge/internal/provider/replay"
	"github.com/omorsi45/routeforge/internal/reliability"
)

// startupActor is recorded as the actor of the activation at startup.
const startupActor = "startup"

// buildRegistry returns the policy registry for cfg. A policy may only use
// models that some configured model names as its policy_name. When
// policy.path is set the policy is loaded, added and activated; otherwise the
// registry starts empty and the router behaves like the basic router until a
// policy is uploaded.
func buildRegistry(cfg *config.Config) (*policy.Registry, error) {
	names := make([]string, len(cfg.Models))
	for i, m := range cfg.Models {
		names[i] = m.PolicyName
	}
	reg := policy.NewRegistry(names, nil)
	if cfg.Policy.Path == "" {
		return reg, nil
	}
	pol, err := policy.LoadFile(cfg.Policy.Path)
	if err != nil {
		return nil, err
	}
	if err := reg.Add(pol); err != nil {
		return nil, fmt.Errorf("%s: %w", cfg.Policy.Path, err)
	}
	if _, err := reg.Activate(pol.Version, startupActor); err != nil {
		return nil, fmt.Errorf("activate policy %s: %w", pol.Version, err)
	}
	return reg, nil
}

// policyReadyCheck fails readiness while reg has no active policy. The text
// is fixed so /readyz does not echo internals.
func policyReadyCheck(reg *policy.Registry) api.ReadyCheck {
	return api.ReadyCheck{
		Name: "policy",
		Check: func(context.Context) error {
			if reg.Active() == nil {
				return errors.New("no active policy")
			}
			return nil
		},
	}
}

// buildProviders constructs every configured provider, keyed by provider
// name. A provider whose api_key_env names an unset or empty variable fails
// the build; the variable name is reported and its value never is.
func buildProviders(cfg *config.Config, getenv func(string) string, client *http.Client) (map[string]provider.Provider, error) {
	out := make(map[string]provider.Provider, len(cfg.Providers))
	for _, p := range cfg.Providers {
		var key string
		if p.APIKeyEnv != "" {
			key = getenv(p.APIKeyEnv)
			if key == "" {
				return nil, fmt.Errorf("provider %q: environment variable %s is not set", p.Name, p.APIKeyEnv)
			}
		}
		switch p.Type {
		case "openai":
			out[p.Name] = openai.New(openai.Options{
				Name:           p.Name,
				BaseURL:        p.BaseURL,
				APIKey:         key,
				MaxTokensParam: p.MaxTokensParam,
				Client:         client,
			})
		case "anthropic":
			out[p.Name] = anthropic.New(anthropic.Options{
				Name:    p.Name,
				BaseURL: p.BaseURL,
				APIKey:  key,
				Client:  client,
			})
		case "fake":
			out[p.Name] = fake.New(p.Name, p.Fake)
		case "replay":
			rp, err := replay.LoadFile(p.Name, p.Replay.File, p.Replay.LatencyProfiles, p.Replay.TimeScale, p.Replay.Seed)
			if err != nil {
				return nil, fmt.Errorf("provider %q: %w", p.Name, err)
			}
			out[p.Name] = rp
		default:
			return nil, fmt.Errorf("provider %q: unsupported type %q", p.Name, p.Type)
		}
	}
	return out, nil
}

// buildHealth returns the latency estimators of the configured models, keyed
// by configured name and seeded from each model's latency_prior. Until calls
// are observed their predictions equal the trainer's offline ones.
func buildHealth(cfg *config.Config) *health.Registry {
	priors := make(map[string]health.Prior, len(cfg.Models))
	for _, m := range cfg.Models {
		lp := m.LatencyPrior
		priors[m.Name] = health.Prior{TTFTMS: lp.TTFTMS, DecodeTokensPerS: lp.DecodeTokensPerS, Jitter: lp.Jitter}
	}
	return health.NewRegistry(priors, health.DefaultAlpha)
}

// buildQueues returns the admission queues of the configured models, keyed by
// configured name and sized from each model's max_concurrency, queue_size and
// queue_timeout.
func buildQueues(cfg *config.Config) *reliability.Queues {
	limits := make(map[string]reliability.QueueLimit, len(cfg.Models))
	for _, m := range cfg.Models {
		limits[m.Name] = reliability.QueueLimit{
			MaxConcurrency: m.MaxConcurrency,
			QueueSize:      m.QueueSize,
			Timeout:        m.QueueTimeout,
		}
	}
	return reliability.NewQueues(limits)
}

// buildBreakers returns the circuit breakers of the configured models, set from
// the reliability.breaker section. They are created on first use per model.
func buildBreakers(cfg *config.Config) *reliability.Breakers {
	b := cfg.Reliability.Breaker
	return reliability.NewBreakers(reliability.BreakerConfig{
		ConsecutiveFailures: b.ConsecutiveFailures,
		ErrorRate:           b.ErrorRate,
		MinRequests:         b.MinRequests,
		Window:              b.Window,
		OpenFor:             b.OpenFor,
		HalfOpenProbes:      b.HalfOpenProbes,
	}, nil)
}

// buildExecutor returns the executor over the configured models: retries and
// failover as reliability sets them, guarded by the breakers and queues and
// feeding the latency estimators with every successful call.
func buildExecutor(cfg *config.Config, providers map[string]provider.Provider, breakers *reliability.Breakers, queues *reliability.Queues, latency *health.Registry) *executor.Executor {
	r := cfg.Reliability
	return executor.NewWithOptions(executor.Options{
		Targets:     buildTargets(cfg, providers),
		Breakers:    breakers,
		Queues:      queues,
		RetryBudget: reliability.NewRetryBudget(r.RetryBudgetRatio, r.RetryBudgetWindow, r.RetryBudgetMin, nil),
		Backoff:     reliability.Backoff{Base: r.BackoffBase, Cap: r.BackoffCap},
		Health:      latency,
		MaxAttempts: r.MaxAttempts,
	})
}

// buildAccountant returns the accountant of cfg over in-memory budgets, rate
// limits and the given usage store, priced by prices. It admits and meters
// requests as accounting.fail_mode says.
func buildAccountant(cfg *config.Config, prices map[string]accounting.Price, usage accounting.UsageStore, logger *slog.Logger) *accounting.Accountant {
	return accounting.NewAccountant(accounting.AccountantOptions{
		Tenants:  cfg.Tenants,
		Prices:   prices,
		Budgets:  accounting.NewMemoryBudgets(),
		Limiter:  accounting.NewMemoryLimiter(time.Now),
		Usage:    usage,
		FailMode: cfg.Accounting.FailMode,
		Logger:   logger,
	})
}

// buildTargets binds each configured model to its provider's instance.
func buildTargets(cfg *config.Config, providers map[string]provider.Provider) map[string]executor.Target {
	targets := make(map[string]executor.Target, len(cfg.Models))
	for _, m := range cfg.Models {
		targets[m.Name] = executor.Target{Model: m.Name, Upstream: m.UpstreamModel, Provider: providers[m.Provider]}
	}
	return targets
}

// modelPrices prices every configured model, keyed by its configured name. A
// model that sets both input_per_mtok and output_per_mtok uses them; any
// other model takes the price file entry for its policy_name. A model with
// neither is an error naming it, and so are all the other problems found. A
// pricing.file that is set is always loaded, even when every model carries
// its own prices.
func modelPrices(cfg *config.Config) (map[string]accounting.Price, error) {
	var table *accounting.PriceTable
	if cfg.Pricing.File != "" {
		t, err := accounting.LoadPrices(cfg.Pricing.File)
		if err != nil {
			return nil, err
		}
		table = t
	}

	prices := make(map[string]accounting.Price, len(cfg.Models))
	var errs []error
	for _, m := range cfg.Models {
		if m.InputPerMTok != nil && m.OutputPerMTok != nil {
			p, err := accounting.NewPrice(m.Name, *m.InputPerMTok, *m.OutputPerMTok)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			prices[m.Name] = p
			continue
		}
		if table == nil {
			errs = append(errs, fmt.Errorf("model %q: no price: set input_per_mtok and output_per_mtok or pricing.file", m.Name))
			continue
		}
		p, ok := table.Price(m.PolicyName)
		if !ok {
			errs = append(errs, fmt.Errorf("model %q: policy_name %q is not in the price file", m.Name, m.PolicyName))
			continue
		}
		p.Name = m.Name
		prices[m.Name] = p
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return prices, nil
}
