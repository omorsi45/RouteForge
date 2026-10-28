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
)

const usageText = `Usage: routeforge-bench <command> [flags]

Commands:
  replay  Check that the gateway makes the routing decisions the trainer made offline
  help    Show this help
`

const (
	defaultBaseURL = "http://127.0.0.1:18080"
	// defaultKey is the public key of the bench tenant in bench/configs. It is
	// meant for loopback runs only.
	defaultKey = "test-key-bench"
)

// runIDRE bounds a run id to what is safe as a directory name.
var runIDRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

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
// disagrees with the trainer is a result, not a failure of the command: it
// exits 0 and the result says parity_pass false. It exits 1 on an error that
// leaves no result, such as a gateway that cannot be reached.
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
	fmt.Fprintf(stdout, "replay OK: %d requests, overall agreement %.4f, parity %s\n",
		results.Requests, results.OverallAgreement, passFail(results.ParityPass))
	return 0
}
