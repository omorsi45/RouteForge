package main

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/omorsi45/routeforge/internal/config"
)

// cmdValidateConfig loads the file named by --config and reports whether it
// is a valid gateway configuration. It returns the process exit code.
func cmdValidateConfig(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("validate-config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	path := fs.String("config", "", "path to the gateway configuration file (required)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *path == "" {
		fmt.Fprintln(stderr, "validate-config: --config is required")
		return 2
	}

	c, err := config.Load(*path)
	if err != nil {
		fmt.Fprintf(stderr, "invalid config %s:\n", *path)
		for _, line := range strings.Split(err.Error(), "\n") {
			fmt.Fprintf(stderr, "  %s\n", line)
		}
		return 1
	}
	fmt.Fprintf(stdout, "config OK: %d tenants, %d providers, %d models\n", len(c.Tenants), len(c.Providers), len(c.Models))
	return 0
}
