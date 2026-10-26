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

	"github.com/omorsi45/routeforge/internal/api"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/router"
)

// serve runs the gateway on ln until ctx ends, then drains: /readyz turns
// 503 and in-flight requests get up to ShutdownTimeout to finish. serve owns
// ln and closes it on every path.
func serve(ctx context.Context, cfg *config.Config, ln net.Listener, getenv func(string) string, logger *slog.Logger) error {
	// No client Timeout: streams outlive any fixed value, so deadlines come
	// from request contexts.
	client := &http.Client{Transport: http.DefaultTransport.(*http.Transport).Clone()}
	providers, err := buildProviders(cfg, getenv, client)
	if err != nil {
		_ = ln.Close()
		return err
	}

	gw := api.New(api.Options{
		Config:   cfg,
		Keys:     auth.NewStore(cfg.Tenants),
		Router:   router.NewBasic(cfg.Models, router.BasicOptions{}),
		Executor: executor.New(buildTargets(cfg, providers), nil),
		Logger:   logger,
	})
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
	shutdownErr := srv.Shutdown(shutdownCtx)
	if err := <-serveErr; !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	logger.Info("stopped")
	return shutdownErr
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
