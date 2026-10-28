package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
)

// cmdValidateConfig loads the file named by --config and reports whether it
// is a valid gateway configuration. Beyond the file itself it checks that
// every model is priced and, when policy.path is set, that the policy
// artifact loads. It returns the process exit code.
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
		reportInvalid(stderr, *path, err)
		return 1
	}

	var errs []error
	if _, err := modelPrices(c); err != nil {
		errs = append(errs, err)
	}
	var pol *policy.Policy
	if c.Policy.Path != "" {
		if pol, err = policy.LoadFile(c.Policy.Path); err != nil {
			errs = append(errs, err)
		}
	}
	if err := errors.Join(errs...); err != nil {
		reportInvalid(stderr, *path, err)
		return 1
	}

	fmt.Fprintf(stdout, "config OK: %d tenants, %d providers, %d models", len(c.Tenants), len(c.Providers), len(c.Models))
	if pol != nil {
		fmt.Fprintf(stdout, ", policy %s", pol.Version)
	}
	fmt.Fprintln(stdout)
	return 0
}

// reportInvalid prints err under an "invalid config" heading, one indented
// line per problem.
func reportInvalid(stderr io.Writer, path string, err error) {
	fmt.Fprintf(stderr, "invalid config %s:\n", path)
	for _, line := range strings.Split(err.Error(), "\n") {
		fmt.Fprintf(stderr, "  %s\n", line)
	}
}
