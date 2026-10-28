package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/api"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/telemetry"
)

// serviceName is the service name the traces carry.
const serviceName = "routeforge"

// tracingFlushTimeout bounds how long shutting down waits for the last spans to
// reach the collector.
const tracingFlushTimeout = 5 * time.Second

// decisionRingSize is how many routing decisions the gateway keeps for
// GET /v1/routing/decisions/{id}.
const decisionRingSize = 10000

// usageRecordsKept is how many of the latest usage records the in-memory usage
// store holds. Totals cover every record.
const usageRecordsKept = 100000

// gateway is a built gateway: its HTTP surface, and the accounting behind it,
// which the invariant tests read to check the books, and the metrics it
// reports to, which GET /metrics serves. usage is the in-memory usage store,
// and nil when the usage records live in PostgreSQL. close stops the work in
// the background, releases the backends and flushes the spans not yet exported;
// serve calls it after the drain.
type gateway struct {
	*api.Server
	accountant      *accounting.Accountant
	usage           *accounting.MemoryUsage
	metrics         *telemetry.Metrics
	backends        *backends
	stopBackground  context.CancelFunc
	shutdownTracing func(context.Context) error
}

// buildGateway constructs the HTTP surface over cfg: providers, prices, the
// storage and limiter backends, the policy registry, the latency estimators,
// the model queues, the circuit breakers, the accountant, the routing engine
// and the executor that retries and fails over. Without a policy the engine
// routes like the basic router. With storage.type postgres the registry writes
// through to the database and a poller makes it follow activations made by other
// replicas. Every component reports to one set of metrics and one tracer. With
// OTEL_EXPORTER_OTLP_ENDPOINT set (read through getenv) the tracer exports spans
// to that collector; without it, tracing costs nothing.
func buildGateway(cfg *config.Config, getenv func(string) string, logger *slog.Logger) (gw *gateway, err error) {
	// No client Timeout: streams outlive any fixed value, so deadlines come
	// from request contexts.
	client := &http.Client{Transport: http.DefaultTransport.(*http.Transport).Clone()}
	providers, err := buildProviders(cfg, getenv, client)
	if err != nil {
		return nil, err
	}
	prices, err := modelPrices(cfg)
	if err != nil {
		return nil, err
	}
	bk, err := buildBackends(context.Background(), cfg, getenv, logger)
	if err != nil {
		return nil, err
	}
	background, stopBackground := context.WithCancel(context.Background())
	defer func() {
		if err != nil { // nothing owns the backends yet
			stopBackground()
			bk.close(context.Background())
		}
	}()
	reg, err := buildGatewayRegistry(cfg, bk)
	if err != nil {
		return nil, err
	}
	tp, shutdownTracing, err := telemetry.NewTracerProvider(context.Background(), serviceName, getenv)
	if err != nil {
		return nil, err
	}
	tracer := tp.Tracer(serviceName)
	metrics := telemetry.NewMetrics()
	watchPolicy(reg, metrics)
	latency := buildHealth(cfg)
	queues := buildQueues(cfg)
	breakers := buildBreakers(cfg)
	inflight := reliability.NewInFlight(cfg.Server.MaxInFlight)
	metrics.SetStateSources(stateSources(cfg, breakers, queues, inflight))
	acct := buildAccountant(cfg, prices, bk, logger, metrics)
	if bk.persister != nil {
		policy.StartPoller(background, reg, bk.persister, policyPollInterval, logger)
	}

	opts := api.Options{
		Config: cfg,
		Keys:   auth.NewStore(cfg.Tenants),
		Router: router.NewEngine(router.EngineOptions{
			Models:   cfg.Models,
			Prices:   prices,
			Policies: reg,
			Signals: router.Signals{
				BreakerOpen:          breakers.IsOpen,
				QueueFull:            queues.Full,
				PredictedP95MS:       latency.PredictedP95MS,
				BudgetRemainingNanos: acct.RemainingNanos,
			},
			Metrics: metrics,
			Tracer:  tracer,
		}),
		Executor:   buildExecutor(cfg, providers, breakers, queues, latency, metrics, tracer),
		Accountant: acct,
		Decisions:  bk.decisions,
		Policies:   reg,
		InFlight:   inflight,
		Metrics:    metrics,
		Tracer:     tracer,
		Logger:     logger,
	}
	if cfg.Policy.Path != "" {
		opts.ReadyChecks = append(opts.ReadyChecks, policyReadyCheck(reg))
	}
	opts.ReadyChecks = append(opts.ReadyChecks, bk.checks...)
	return &gateway{
		Server:          api.New(opts),
		accountant:      acct,
		usage:           bk.memUsage,
		metrics:         metrics,
		backends:        bk,
		stopBackground:  stopBackground,
		shutdownTracing: shutdownTracing,
	}, nil
}

// buildGatewayRegistry returns the policy registry for the backends bk: the
// file policy alone with memory storage, and the policies of the database,
// written through, with PostgreSQL.
func buildGatewayRegistry(cfg *config.Config, bk *backends) (*policy.Registry, error) {
	if bk.persister == nil {
		return buildRegistry(cfg)
	}
	ctx, cancel := context.WithTimeout(context.Background(), backendOpenTimeout)
	defer cancel()
	return buildPersistedRegistry(ctx, cfg, bk.persister)
}

// close stops the policy poller, releases the backends (writing the usage
// records and decisions still queued) and then exports the spans still waiting.
// Call it after the drain, so that the requests that finished while draining
// are written and traced too.
func (g *gateway) close(ctx context.Context, logger *slog.Logger) {
	g.stopBackground()
	g.backends.close(context.WithoutCancel(ctx))
	g.closeTracing(ctx, logger)
}

// closeTracing exports the spans still waiting and stops the tracer provider,
// within tracingFlushTimeout. A failure is logged: the gateway is going away
// either way.
func (g *gateway) closeTracing(ctx context.Context, logger *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), tracingFlushTimeout)
	defer cancel()
	if err := g.shutdownTracing(ctx); err != nil {
		logger.Warn("flush traces failed", "error", err)
	}
}

// serve runs the gateway on ln until ctx ends, then drains: /readyz turns
// 503 and in-flight requests get up to ShutdownTimeout to finish. serve owns
// ln and closes it on every path.
func serve(ctx context.Context, cfg *config.Config, ln net.Listener, getenv func(string) string, logger *slog.Logger) error {
	gw, err := buildGateway(cfg, getenv, logger)
	if err != nil {
		_ = ln.Close()
		return err
	}
	// Runs after the drain below, so the usage records and spans of the requests
	// that finished while draining are written and exported too.
	defer gw.close(ctx, logger)
	srv := api.NewHTTPServer(cfg.Server, gw.Handler())

	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()
	logger.Info("serving", "addr", ln.Addr().String())

	select {
	case err := <-serveErr:
		// Serve returns before any shutdown only on a listener failure.
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
	}

	gw.SetDraining()
	// The parent ctx is already canceled, so the drain gets its own bound.
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cfg.Server.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Warn("shutdown did not drain cleanly", "error", err)
	}
	if err := <-serveErr; !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	logger.Info("stopped")
	return nil
}

// cmdServe loads --config and runs the gateway until ctx ends. It returns the
// process exit code.
func cmdServe(ctx context.Context, args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("config", "", "path to the gateway configuration file (required)")
	listen := fs.String("listen", "", "listen address, overriding server.listen")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *path == "" {
		fmt.Fprintln(stderr, "serve: --config is required")
		return 2
	}

	cfg, err := config.Load(*path)
	if err != nil {
		fmt.Fprintf(stderr, "serve: %v\n", err)
		return 1
	}
	ln, err := net.Listen("tcp", cmp.Or(*listen, cfg.Server.Listen))
	if err != nil {
		fmt.Fprintf(stderr, "serve: %v\n", err)
		return 1
	}

	logger := slog.New(slog.NewJSONHandler(stderr, nil))
	if err := serve(ctx, cfg, ln, getenv, logger); err != nil {
		fmt.Fprintf(stderr, "serve: %v\n", err)
		return 1
	}
	return 0
}
