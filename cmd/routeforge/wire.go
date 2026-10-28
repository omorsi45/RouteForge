package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	neturl "net/url"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
	"go.opentelemetry.io/otel/trace"

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
	"github.com/omorsi45/routeforge/internal/store"
	"github.com/omorsi45/routeforge/internal/store/postgres"
	"github.com/omorsi45/routeforge/internal/store/redisstore"
	"github.com/omorsi45/routeforge/internal/telemetry"
)

// startupActor is recorded as the actor of the activation at startup.
const startupActor = "startup"

// buildRegistry returns the policy registry for cfg. A policy may only use
// models that some configured model names as its policy_name. When
// policy.path is set the policy is loaded, added and activated; otherwise the
// registry starts empty and the router behaves like the basic router until a
// policy is uploaded.
func buildRegistry(cfg *config.Config) (*policy.Registry, error) {
	reg := newRegistry(cfg)
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

// newRegistry returns an empty registry that accepts the policies cfg can map
// onto its models.
func newRegistry(cfg *config.Config) *policy.Registry {
	names := make([]string, len(cfg.Models))
	for i, m := range cfg.Models {
		names[i] = m.PolicyName
	}
	return policy.NewRegistry(names, nil)
}

// buildPersistedRegistry returns the policy registry for a gateway whose
// policies live in p. The store decides what runs: when it has an active
// policy, that one is loaded from its stored bytes and activated, and the file
// policy, if there is one, is only saved (inactive) for later; when it has none,
// the file policy is saved and activated with actor startup. A stored policy
// that cannot be used stops startup rather than letting another one serve
// silently. The registry writes through to p from here on.
func buildPersistedRegistry(ctx context.Context, cfg *config.Config, p policy.Persister) (*policy.Registry, error) {
	reg := newRegistry(cfg)
	var file *policy.Policy
	if cfg.Policy.Path != "" {
		pol, err := policy.LoadFile(cfg.Policy.Path)
		if err != nil {
			return nil, err
		}
		file = pol
	}

	stored, ok, err := p.ActiveVersion(ctx)
	if err != nil {
		return nil, fmt.Errorf("read the active policy from storage: %w", err)
	}
	if ok {
		// Added and activated before the persister is attached: the store already
		// holds both.
		raw, err := p.LoadRaw(ctx, stored)
		if err != nil {
			return nil, fmt.Errorf("load policy %s from storage: %w", stored, err)
		}
		pol, err := policy.Load(raw)
		if err != nil {
			return nil, fmt.Errorf("stored policy %s: %w", stored, err)
		}
		if pol.Version != stored {
			return nil, fmt.Errorf("stored policy %s holds version %s", stored, pol.Version)
		}
		if err := reg.Add(pol); err != nil {
			return nil, fmt.Errorf("stored policy %s: %w", stored, err)
		}
		if _, err := reg.Activate(stored, startupActor); err != nil {
			return nil, fmt.Errorf("activate stored policy %s: %w", stored, err)
		}
	}

	reg.SetPersister(p)
	if file == nil {
		return reg, nil
	}
	if err := reg.Add(file); err != nil { // saved inactive unless the store has it
		return nil, fmt.Errorf("%s: %w", cfg.Policy.Path, err)
	}
	if !ok {
		if _, err := reg.Activate(file.Version, startupActor); err != nil {
			return nil, fmt.Errorf("activate policy %s: %w", file.Version, err)
		}
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
// feeding the latency estimators with every successful call. Its attempts are
// reported to metrics and traced by tracer, either of which may be nil.
func buildExecutor(cfg *config.Config, providers map[string]provider.Provider, breakers *reliability.Breakers, queues *reliability.Queues, latency *health.Registry, metrics *telemetry.Metrics, tracer trace.Tracer) *executor.Executor {
	r := cfg.Reliability
	return executor.NewWithOptions(executor.Options{
		Targets:     buildTargets(cfg, providers),
		Breakers:    breakers,
		Queues:      queues,
		RetryBudget: reliability.NewRetryBudget(r.RetryBudgetRatio, r.RetryBudgetWindow, r.RetryBudgetMin, nil),
		Backoff:     reliability.Backoff{Base: r.BackoffBase, Cap: r.BackoffCap},
		Health:      latency,
		MaxAttempts: r.MaxAttempts,
		Metrics:     metrics,
		Tracer:      tracer,
	})
}

// buildAccountant returns the accountant of cfg over the budgets, rate limits
// and usage store of bk, priced by prices. It admits and meters requests as
// accounting.fail_mode says and reports to metrics, which may be nil.
func buildAccountant(cfg *config.Config, prices map[string]accounting.Price, bk *backends, logger *slog.Logger, metrics *telemetry.Metrics) *accounting.Accountant {
	return accounting.NewAccountant(accounting.AccountantOptions{
		Tenants:  cfg.Tenants,
		Prices:   prices,
		Budgets:  bk.budgets,
		Limiter:  bk.limiter,
		Usage:    bk.usage,
		FailMode: cfg.Accounting.FailMode,
		Logger:   logger,
		Metrics:  metrics,
	})
}

// watchPolicy makes metrics follow the active policy of reg: policy_info names
// the active version now and after every activation and rollback.
func watchPolicy(reg *policy.Registry, metrics *telemetry.Metrics) {
	if active := reg.Active(); active != nil {
		metrics.PolicyActivated(active.Version, "")
	}
	reg.OnActivate(func(a policy.Activation) { metrics.PolicyActivated(a.Version, a.Previous) })
}

// stateSources tells the state gauges where to read the breakers, queues and
// in-flight count of the configured models.
func stateSources(cfg *config.Config, breakers *reliability.Breakers, queues *reliability.Queues, inflight *reliability.InFlight) telemetry.StateSources {
	models := make([]string, len(cfg.Models))
	for i, m := range cfg.Models {
		models[i] = m.Name
	}
	return telemetry.StateSources{
		Models:       models,
		BreakerState: func(model string) int { return int(breakers.State(model)) },
		QueueDepth:   queues.Depth,
		InFlight:     inflight.Count,
	}
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

// Settings of the Redis client. The budget and limiter scripts are not
// idempotent, so the client must not retry a command whose reply was lost
// (MaxRetries of -1): a retried reservation could be counted twice. The short
// timeouts keep a stuck Redis from holding a request; the accountant's fail
// mode decides what its failure means.
const (
	redisDialTimeout  = 2 * time.Second
	redisReadTimeout  = time.Second
	redisWriteTimeout = time.Second
)

// backendOpenTimeout bounds connecting to PostgreSQL, applying its migrations
// and reading the policy it holds at startup.
const backendOpenTimeout = 30 * time.Second

// policyPollInterval is how often a gateway on PostgreSQL looks for a policy
// activated elsewhere. It is a variable so that an integration test can shorten
// it; nothing else changes it.
var policyPollInterval = 5 * time.Second

// The fixed texts /readyz shows when the storage or limiter check fails. The
// error itself goes to the log.
const (
	storageUnavailable = "storage unavailable"
	limiterUnavailable = "limiter unavailable"
)

// backends are the stores a gateway runs on, chosen by the storage and limiter
// sections of the configuration.
type backends struct {
	usage     accounting.UsageStore
	memUsage  *accounting.MemoryUsage // usage with memory storage, otherwise nil; tests read the records from it
	decisions store.DecisionStore
	budgets   accounting.BudgetStore
	limiter   accounting.RateLimiter
	persister policy.Persister // the policy store; nil with memory storage
	checks    []api.ReadyCheck // readiness checks of the backends that can fail
	closers   []func(context.Context)
}

// close releases the backends in the reverse of the order they were opened.
func (b *backends) close(ctx context.Context) {
	for i := len(b.closers) - 1; i >= 0; i-- {
		b.closers[i](ctx)
	}
}

// buildBackends opens the stores cfg selects. The PostgreSQL connection string
// and the Redis URL are read from the environment variables the configuration
// names, through getenv, and appear in no error and no log: an error names the
// variable. PostgreSQL is opened (and migrated) now, so an unreachable
// database stops startup; the Redis client connects when it is first used, and
// /readyz reports it until it can.
func buildBackends(ctx context.Context, cfg *config.Config, getenv func(string) string, logger *slog.Logger) (*backends, error) {
	b := &backends{}
	if err := b.openStorage(ctx, cfg, getenv, logger); err != nil {
		return nil, err
	}
	if err := b.openLimiter(cfg, getenv, logger); err != nil {
		b.close(ctx)
		return nil, err
	}
	return b, nil
}

func (b *backends) openStorage(ctx context.Context, cfg *config.Config, getenv func(string) string, logger *slog.Logger) error {
	if cfg.Storage.Type != config.StoragePostgres {
		mem := accounting.NewMemoryUsage(usageRecordsKept)
		b.usage, b.memUsage = mem, mem
		b.decisions = store.NewRing(decisionRingSize)
		return nil
	}
	dsn := getenv(cfg.Storage.DSNEnv)
	if dsn == "" {
		return fmt.Errorf("storage.dsn_env: environment variable %s is not set", cfg.Storage.DSNEnv)
	}
	openCtx, cancel := context.WithTimeout(ctx, backendOpenTimeout)
	defer cancel()
	pg, err := postgres.Open(openCtx, dsn, postgres.Options{QueueSize: cfg.Accounting.UsageQueue, Logger: logger})
	if err != nil {
		return fmt.Errorf("storage.dsn_env %s: %w", cfg.Storage.DSNEnv, err)
	}
	b.usage, b.decisions, b.persister = pg, pg, pg
	b.closers = append(b.closers, func(context.Context) { pg.Close() })
	b.checks = append(b.checks, api.ReadyCheck{Name: "storage", Fixed: storageUnavailable, Check: pg.Ping})
	return nil
}

func (b *backends) openLimiter(cfg *config.Config, getenv func(string) string, logger *slog.Logger) error {
	if cfg.Limiter.Type != config.LimiterRedis {
		b.budgets = accounting.NewMemoryBudgets()
		b.limiter = accounting.NewMemoryLimiter(time.Now)
		return nil
	}
	url := getenv(cfg.Limiter.URLEnv)
	if url == "" {
		return fmt.Errorf("limiter.url_env: environment variable %s is not set", cfg.Limiter.URLEnv)
	}
	opts, err := redisOptions(url)
	if err != nil {
		return fmt.Errorf("limiter.url_env: %s does not hold a valid Redis URL: %w", cfg.Limiter.URLEnv, err)
	}
	client := redis.NewClient(opts)
	rs := redisstore.New(client)
	b.limiter, b.budgets = rs, rs
	password := opts.Password
	b.closers = append(b.closers, func(context.Context) {
		if err := client.Close(); err != nil {
			logger.Warn("close the redis client failed", "error", redactText(err.Error(), password))
		}
	})
	b.checks = append(b.checks, api.ReadyCheck{
		Name:  "limiter",
		Fixed: limiterUnavailable,
		Check: func(ctx context.Context) error {
			if err := rs.Ping(ctx); err != nil {
				// The error is logged, with the request id, so it must not carry the password.
				return errors.New(redactText(err.Error(), password))
			}
			return nil
		},
	})
	return nil
}

// redisOptions parses a Redis URL and sets the client up as described at
// redisDialTimeout. An error says what is wrong with the URL without quoting
// it, because it may hold a password.
func redisOptions(url string) (*redis.Options, error) {
	opts, err := redis.ParseURL(url)
	if err != nil {
		var urlErr *neturl.Error
		if errors.As(err, &urlErr) {
			err = urlErr.Err // net/url quotes the whole URL
		}
		return nil, errors.New(err.Error())
	}
	opts.MaxRetries = -1
	opts.DialTimeout = redisDialTimeout
	opts.ReadTimeout = redisReadTimeout
	opts.WriteTimeout = redisWriteTimeout
	return opts, nil
}

// redactText blanks every occurrence of secret in text.
func redactText(text, secret string) string {
	if secret == "" {
		return text
	}
	return strings.ReplaceAll(text, secret, "xxxxx")
}
