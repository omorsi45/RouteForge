// Command routeforge is the RouteForge gateway and its operator tooling.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"syscall"
)

// version is overridable with -ldflags "-X main.version=...".
var version = "dev"

const usageText = `Usage: routeforge <command> [flags]

Commands:
  serve            Run the gateway
  validate-config  Validate a gateway configuration file
  keygen           Generate an API key and its stored digest
  version          Print version information
  healthcheck      Probe a running gateway and exit non-zero if unhealthy
  help             Show this help
`

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := run(ctx, os.Args[1:], os.Stdout, os.Stderr, os.Getenv)
	stop()
	os.Exit(code)
}

// run dispatches a subcommand and returns the process exit code. args
// excludes the program name.
func run(ctx context.Context, args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	_ = ctx
	_ = getenv

	if len(args) == 0 {
		fmt.Fprint(stderr, usageText)
		return 2
	}

	switch args[0] {
	case "version":
		return cmdVersion(stdout)
	case "validate-config":
		return cmdValidateConfig(args[1:], stdout, stderr)
	case "help":
		fmt.Fprint(stdout, usageText)
		return 0
	default:
		fmt.Fprintf(stderr, "routeforge: unknown command %q\n\n%s", args[0], usageText)
		return 2
	}
}

func cmdVersion(stdout io.Writer) int {
	line := fmt.Sprintf("routeforge %s (%s, %s/%s)", version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				rev := s.Value
				if len(rev) > 12 {
					rev = rev[:12]
				}
				line += " commit " + rev
				break
			}
		}
	}
	fmt.Fprintln(stdout, line)
	return 0
}
