package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"time"
)

// cmdHealthcheck probes a running gateway so a container image without curl
// can declare a HEALTHCHECK. It returns the process exit code.
func cmdHealthcheck(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("healthcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	url := fs.String("url", "http://127.0.0.1:8080/healthz", "URL to probe")
	timeout := fs.Duration("timeout", 2*time.Second, "time allowed for the probe")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, *url, nil)
	if err != nil {
		fmt.Fprintf(stderr, "healthcheck: %v\n", err)
		return 1
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Fprintf(stderr, "healthcheck: %v\n", err)
		return 1
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(stderr, "healthcheck: %s returned %s\n", *url, resp.Status)
		return 1
	}
	fmt.Fprintln(stdout, "ok")
	return 0
}
