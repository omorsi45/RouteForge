package main

import (
	"fmt"
	"io"

	"github.com/omorsi45/routeforge/internal/auth"
)

// cmdKeygen prints a new API key and its SHA-256 digest to stdout and reports
// failures on stderr. The key is shown only once, so a failed write is
// reported as a failure.
func cmdKeygen(stdout, stderr io.Writer, rand io.Reader) int {
	key, err := auth.GenerateKey(rand)
	if err != nil {
		fmt.Fprintf(stderr, "routeforge: keygen: %v\n", err)
		return 1
	}
	if _, err := fmt.Fprintf(stdout, "key: %s\nsha256: %s\n", key, auth.HashKey(key)); err != nil {
		fmt.Fprintf(stderr, "routeforge: keygen: write output: %v\n", err)
		return 1
	}
	return 0
}
