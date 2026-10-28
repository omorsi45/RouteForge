//go:build integration

// Package pgtest starts a throwaway PostgreSQL server for integration tests.
// It is compiled only with the integration build tag, and the first use
// downloads the pinned PostgreSQL binary into the library's default cache
// (see docs/operations.md); later uses reuse it.
package pgtest

import (
	"fmt"
	"io"
	"net"
	"path/filepath"
	"testing"
	"time"

	embeddedpostgres "github.com/fergusstrange/embedded-postgres"
)

const (
	// User and Password are the credentials of the test server. They are fake:
	// the server listens on the loopback interface for the length of one test.
	User     = "routeforge"
	Password = "routeforge-local-only"
	// Database is the database every test server creates.
	Database = "routeforge"
)

// Start runs a PostgreSQL server for the test, stops it when the test ends and
// returns the DSN to reach it. The runtime, data and binary directories live
// under t.TempDir(), which must be on a filesystem that keeps Unix permissions
// (PostgreSQL insists on a private data directory); the downloaded archive
// stays in the library's cache.
func Start(t testing.TB) string {
	t.Helper()
	port := freePort(t)
	dir := t.TempDir()
	pg := embeddedpostgres.NewDatabase(embeddedpostgres.DefaultConfig().
		Version(embeddedpostgres.V17).
		Port(port).
		Username(User).
		Password(Password).
		Database(Database).
		RuntimePath(filepath.Join(dir, "runtime")).
		DataPath(filepath.Join(dir, "data")).
		BinariesPath(filepath.Join(dir, "bin")).
		StartTimeout(90 * time.Second).
		Logger(io.Discard))
	if err := pg.Start(); err != nil {
		t.Fatalf("start embedded postgres: %v", err)
	}
	t.Cleanup(func() {
		if err := pg.Stop(); err != nil {
			t.Errorf("stop embedded postgres: %v", err)
		}
	})
	return fmt.Sprintf("postgres://%s:%s@localhost:%d/%s?sslmode=disable", User, Password, port, Database)
}

// freePort returns a TCP port that nothing is listening on right now.
func freePort(t testing.TB) uint32 {
	t.Helper()
	ln, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("find a free port: %v", err)
	}
	defer ln.Close()
	return uint32(ln.Addr().(*net.TCPAddr).Port)
}
