// Command routeforge-bench runs the RouteForge benchmarks against a gateway and
// writes their results under bench/results.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"syscall"
	"time"

	"github.com/omorsi45/routeforge/internal/bench"
	"github.com/omorsi45/routeforge/internal/config"
)

const usageText = `Usage: routeforge-bench <command> [flags]

Commands:
  replay  Check that the gateway makes the routing decisions the trainer made offline
  load    Measure throughput, latency and gateway overhead under open and closed loop load
  faults  Inject provider faults and check the accounting invariants
  help    Show this help
`

const (
	defaultBaseURL = "http://127.0.0.1:18080"
	// The keys of bench/configs, public and meant for loopback runs only.
	defaultKey      = "test-key-bench"
	defaultOverKey  = "test-key-bench-over"
	defaultAdminKey = "test-admin-key"

	// loadSeed and faultsSeed fix the prompts and the arrival times of the runs.
	loadSeed   = 20261026
	faultsSeed = 20261026
)

// runIDRE bounds a run id to what is safe as a directory name.
var runIDRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// loadConfigRE matches the load configs, whose names give the result kinds.
var loadConfigRE = regexp.MustCompile(`^load-[a-z0-9-]+\.yaml$`)

// Seams for the tests of the load command, which would otherwise start real
// gateway processes and run for minutes.
var (
	startGatewayProcess = func(ctx context.Context, o bench.ProcessOptions) (bench.Gateway, error) {
		g, err := bench.StartGatewayProcess(ctx, o)
		if err != nil {
			return nil, err // not g: a nil *GatewayProcess is not a nil Gateway
		}
		return g, nil
	}
	loadScenarios = bench.DefaultScenarios
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := run(ctx, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}

// run dispatches a subcommand and returns the process exit code. args excludes
// the program name.
func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usageText)
		return 2
	}
	switch args[0] {
	case "replay":
		return cmdReplay(ctx, args[1:], stdout, stderr)
	case "load":
		return cmdLoad(ctx, args[1:], stdout, stderr)
	case "faults":
		return cmdFaults(ctx, args[1:], stdout, stderr)
	case "help":
		fmt.Fprint(stdout, usageText)
		return 0
	default:
		fmt.Fprintf(stderr, "routeforge-bench: unknown command %q\n\n%s", args[0], usageText)
		return 2
	}
}

// checkRunID reports why id cannot name a run.
func checkRunID(id string) error {
	if id == "" {
		return fmt.Errorf("--run-id is required")
	}
	if !runIDRE.MatchString(id) {
		return fmt.Errorf("--run-id %q is not valid: use letters, digits, '.', '_' and '-', starting with a letter or digit", id)
	}
	return nil
}

// resultDir is where the results of run id live.
func resultDir(root, id string) string {
	return filepath.Join(root, "bench", "results", id)
}

// configFile is the path of a config named on the command line: relative paths
// resolve against the repository root, as bench.CollectEnv resolves them.
func configFile(root, configPath string) string {
	if filepath.IsAbs(configPath) {
		return configPath
	}
	return filepath.Join(root, configPath)
}

// newResult builds the envelope of a result file for a run that just finished.
func newResult(kind, runID, configPath string, env bench.Env, params, results any) bench.Result {
	return bench.Result{
		SchemaVersion: bench.SchemaVersion,
		Kind:          kind,
		RunID:         runID,
		CreatedAt:     time.Now().UTC().Format(time.RFC3339),
		Env:           env,
		ConfigPath:    filepath.ToSlash(configPath),
		Params:        params,
		Results:       results,
	}
}

func passFail(ok bool) string {
	if ok {
		return "pass"
	}
	return "fail"
}

// cmdReplay sends the offline decisions of the trainer's test split through a
// running gateway and writes bench/results/<run-id>/replay.json. A gateway that
// disagrees with the trainer on a deterministic strategy is a result, not a
// failure of the command: it exits 0 and the result says parity_pass false. It
// exits 1 on an error that leaves no result, such as a gateway that cannot be
// reached.
func cmdReplay(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", defaultBaseURL, "URL of the running gateway")
	key := fs.String("key", defaultKey, "API key of the bench tenant")
	configPath := fs.String("config", "bench/configs/replay.yaml", "config the gateway runs (relative paths resolve against --root); recorded in the result")
	root := fs.String("root", ".", "repository root")
	runID := fs.String("run-id", "", "name of the run and of its directory under bench/results (required)")
	workers := fs.Int("workers", 16, "requests in flight at once")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if err := checkRunID(*runID); err != nil {
		fmt.Fprintf(stderr, "replay: %v\n", err)
		return 2
	}
	if *workers < 1 {
		fmt.Fprintln(stderr, "replay: --workers must be at least 1")
		return 2
	}

	env, err := bench.CollectEnv(*root, *configPath)
	if err != nil {
		fmt.Fprintf(stderr, "replay: %v\n", err)
		return 1
	}
	results, err := bench.RunReplay(ctx, bench.ReplayOptions{
		Client:  bench.NewClient(*baseURL, *key),
		Root:    *root,
		Workers: *workers,
	})
	if err != nil {
		fmt.Fprintf(stderr, "replay: %v\n", err)
		return 1
	}
	params := map[string]any{
		"base_url":         *baseURL,
		"workers":          *workers,
		"parity_threshold": bench.ParityThreshold,
	}
	r := newResult("replay", *runID, *configPath, env, params, results)
	if _, err := bench.WriteResult(resultDir(*root, *runID), r); err != nil {
		fmt.Fprintf(stderr, "replay: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "replay OK: %d requests, deterministic agreement %.4f, overall agreement %.4f, parity %s\n",
		results.Requests, results.DeterministicAgreement, results.OverallAgreement, passFail(results.ParityPass))
	return 0
}

// scenarioParams lists scenarios for a result file.
func scenarioParams(scenarios []bench.Scenario) []map[string]any {
	out := make([]map[string]any, len(scenarios))
	for i, sc := range scenarios {
		p := map[string]any{"name": sc.Name, "mode": sc.Mode}
		if sc.Mode == bench.ModeOpen {
			p["rps"], p["duration_s"] = sc.RPS, sc.Duration.Seconds()
		} else {
			p["concurrency"], p["requests"] = sc.Concurrency, sc.Requests
		}
		out[i] = p
	}
	return out
}

// cmdLoad runs the load scenarios against gateways it starts from the given
// config, one fresh gateway per scenario, and writes
// bench/results/<run-id>/<kind>.json, where the kind is the name of the config
// (load-zero or load-profiled). Requests that fail are results; it exits 1 when
// a gateway cannot be run or measured.
func cmdLoad(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("load", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayBin := fs.String("gateway-bin", "bin/routeforge", "gateway executable, started once per scenario")
	configPath := fs.String("config", "", "gateway config of the run: bench/configs/load-<name>.yaml (required; relative paths resolve against --root)")
	key := fs.String("key", defaultKey, "API key of the bench tenant")
	root := fs.String("root", ".", "repository root")
	runID := fs.String("run-id", "", "name of the run and of its directory under bench/results (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if err := checkRunID(*runID); err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 2
	}
	if *configPath == "" {
		fmt.Fprintln(stderr, "load: --config is required")
		return 2
	}
	if !loadConfigRE.MatchString(filepath.Base(*configPath)) {
		fmt.Fprintf(stderr, "load: --config %q must be a load config, named load-<name>.yaml\n", *configPath)
		return 2
	}
	if info, err := os.Stat(*gatewayBin); err != nil || info.IsDir() {
		fmt.Fprintf(stderr, "load: --gateway-bin %q is not a file; build it with make build-bench\n", *gatewayBin)
		return 2
	}
	kind := filepath.Base(*configPath)
	kind = kind[:len(kind)-len(filepath.Ext(kind))]

	env, err := bench.CollectEnv(*root, *configPath)
	if err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 1
	}
	binary, err := filepath.Abs(*gatewayBin)
	if err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 1
	}
	configAbs, err := filepath.Abs(configFile(*root, *configPath))
	if err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 1
	}
	dir := resultDir(*root, *runID)
	scenarios := loadScenarios()
	results, err := bench.RunLoad(ctx, bench.LoadOptions{
		Kind: kind,
		Start: func(ctx context.Context, scenario string) (bench.Gateway, error) {
			return startGatewayProcess(ctx, bench.ProcessOptions{
				Binary:     binary,
				ConfigPath: configAbs,
				Addr:       bench.DefaultGatewayAddr,
				LogPath:    filepath.Join(dir, "raw", "gateway-"+kind+"-"+scenario+".log"),
			})
		},
		Key:       *key,
		Scenarios: scenarios,
		Seed:      loadSeed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 1
	}

	params := map[string]any{
		"scenarios": scenarioParams(scenarios),
		"strategy":  "quality",
		"profile":   "balanced",
		"seed":      loadSeed,
		"listen":    bench.DefaultGatewayAddr,
	}
	if _, err := bench.WriteResult(dir, newResult(kind, *runID, *configPath, env, params, results)); err != nil {
		fmt.Fprintf(stderr, "load: %v\n", err)
		return 1
	}
	requests := 0
	for _, sc := range results.Scenarios {
		requests += sc.Requests
	}
	fmt.Fprintf(stdout, "load OK: %s, %d scenarios, %d requests\n", kind, len(results.Scenarios), requests)
	if t := results.OverheadTarget; t != nil {
		fmt.Fprintf(stdout, "overhead target: p95 %.2f ms on %s, below %.0f ms: %s\n", t.P95MS, t.Scenario, t.P95MSBelow, passFail(t.Pass))
	}
	return 0
}

// cmdFaults runs the fault benchmark against a running gateway built from the
// faults config and writes bench/results/<run-id>/faults.json. Violated
// accounting invariants are a result, as a failed parity is for replay: it
// exits 0 and the result says invariants_hold false. It exits 1 on an error that
// leaves no result.
func cmdFaults(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("faults", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", defaultBaseURL, "URL of the running gateway")
	key := fs.String("key", defaultKey, "API key of the bench tenant")
	overKey := fs.String("over-key", defaultOverKey, "API key of the bench-over tenant")
	adminKey := fs.String("admin-key", defaultAdminKey, "admin API key, which reads the accounting")
	configPath := fs.String("config", "bench/configs/faults.yaml", "config the gateway runs (relative paths resolve against --root); gives the budgets")
	root := fs.String("root", ".", "repository root")
	runID := fs.String("run-id", "", "name of the run and of its directory under bench/results (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if err := checkRunID(*runID); err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 2
	}
	for _, k := range []struct{ flag, value string }{{"--key", *key}, {"--over-key", *overKey}, {"--admin-key", *adminKey}} {
		if k.value == "" {
			fmt.Fprintf(stderr, "faults: %s must not be empty\n", k.flag)
			return 2
		}
	}

	env, err := bench.CollectEnv(*root, *configPath)
	if err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 1
	}
	cfg, err := config.Load(configFile(*root, *configPath))
	if err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 1
	}
	setup, err := bench.FaultsSetup(cfg)
	if err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 1
	}
	opts := bench.FaultsOptions{
		Bench:            bench.NewClient(*baseURL, *key),
		Over:             bench.NewClient(*baseURL, *overKey),
		Admin:            bench.NewClient(*baseURL, *adminKey),
		BenchTenant:      bench.BenchTenant,
		OverTenant:       bench.OverTenant,
		OverModel:        setup.OverModel,
		BenchBudgetNanos: setup.BenchBudgetNanos,
		OverBudgetNanos:  setup.OverBudgetNanos,
		Seed:             faultsSeed,
	}
	results, err := bench.RunFaults(ctx, opts)
	if err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 1
	}
	params := opts.Params()
	params["base_url"] = *baseURL
	r := newResult("faults", *runID, *configPath, env, params, results)
	if _, err := bench.WriteResult(resultDir(*root, *runID), r); err != nil {
		fmt.Fprintf(stderr, "faults: %v\n", err)
		return 1
	}
	verdict := "violated"
	if results.Invariants.InvariantsHold {
		verdict = "hold"
	}
	fmt.Fprintf(stdout, "faults OK: %d requests, success rate %.4f, invariants %s\n",
		results.Bench.Requests+results.BenchOver.Requests, results.SuccessRate, verdict)
	return 0
}
