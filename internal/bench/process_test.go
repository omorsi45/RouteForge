//go:build unix

package bench

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// fakeGatewayEnv makes the test binary behave as a gateway: StartGatewayProcess
// is tested against a real child process that answers like `routeforge serve`
// without the gateway behind it. Its value picks the behavior.
const fakeGatewayEnv = "BENCH_FAKE_GATEWAY"

func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeGatewayEnv); mode != "" {
		fakeGateway(mode)
		return
	}
	os.Exit(m.Run())
}

// fakeGateway serves /healthz and /readyz on the --listen address of its
// arguments until it gets SIGTERM, then exits 0 like the gateway does. Modes:
// "ready" is the normal gateway, "never-ready" answers /readyz with 503,
// "exit" exits with status 3 at once and "ignore-term" ignores SIGTERM.
func fakeGateway(mode string) {
	var addr string
	for i, a := range os.Args {
		if a == "--listen" && i+1 < len(os.Args) {
			addr = os.Args[i+1]
		}
	}
	if mode == "exit" {
		fmt.Fprintln(os.Stderr, "fake gateway: failing on purpose")
		os.Exit(3)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(http.ResponseWriter, *http.Request) {})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if mode == "never-ready" {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	srv := &http.Server{Addr: addr, Handler: mux}
	if mode == "ignore-term" {
		signal.Ignore(syscall.SIGTERM)
	} else {
		term := make(chan os.Signal, 1)
		signal.Notify(term, syscall.SIGTERM)
		go func() {
			<-term
			_ = srv.Close()
		}()
	}
	fmt.Println("fake gateway serving", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "fake gateway:", err)
		os.Exit(1)
	}
}

func freeAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func fakeGatewayOptions(t *testing.T, mode string) ProcessOptions {
	t.Helper()
	return ProcessOptions{
		Binary:       os.Args[0],
		ConfigPath:   "unused.yaml",
		Addr:         freeAddr(t),
		LogPath:      filepath.Join(t.TempDir(), "raw", "gateway.log"),
		Env:          []string{fakeGatewayEnv + "=" + mode},
		ReadyTimeout: 10 * time.Second,
		StopTimeout:  5 * time.Second,
	}
}

func get(url string) (int, error) {
	c := &http.Client{Timeout: time.Second}
	resp, err := c.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	return resp.StatusCode, nil
}

func TestGatewayProcessStartsAndStops(t *testing.T) {
	o := fakeGatewayOptions(t, "ready")
	gw, err := StartGatewayProcess(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	stopped := false
	defer func() {
		if !stopped {
			_ = gw.Stop()
		}
	}()

	if want := "http://" + o.Addr; gw.BaseURL() != want {
		t.Errorf("base url = %q, want %q", gw.BaseURL(), want)
	}
	if gw.PID() <= 0 || gw.PID() == os.Getpid() {
		t.Errorf("pid = %d, want the child's", gw.PID())
	}
	if status, err := get(gw.BaseURL() + "/readyz"); err != nil || status != http.StatusOK {
		t.Fatalf("readyz = %d, %v, want 200", status, err)
	}

	if err := gw.Stop(); err != nil {
		t.Fatalf("stop: %v", err)
	}
	stopped = true
	if _, err := get(gw.BaseURL() + "/healthz"); err == nil {
		t.Error("the gateway still answers after Stop")
	}
	if err := gw.Stop(); err != nil {
		t.Errorf("a second stop: %v", err)
	}

	log, err := os.ReadFile(o.LogPath) // the log directory was created
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(log), "fake gateway serving") {
		t.Errorf("gateway log = %q, want the gateway's output", log)
	}
}

func TestGatewayProcessExitsBeforeReady(t *testing.T) {
	o := fakeGatewayOptions(t, "exit")
	_, err := StartGatewayProcess(context.Background(), o)
	if err == nil {
		t.Fatal("want an error when the gateway exits at once")
	}
	if !strings.Contains(err.Error(), "exited") || !strings.Contains(err.Error(), o.LogPath) {
		t.Errorf("error %q should say the gateway exited and name its log", err)
	}
}

func TestGatewayProcessNeverReady(t *testing.T) {
	o := fakeGatewayOptions(t, "never-ready")
	o.ReadyTimeout = 600 * time.Millisecond
	start := time.Now()
	_, err := StartGatewayProcess(context.Background(), o)
	if err == nil {
		t.Fatal("want an error when /readyz never answers 200")
	}
	if !strings.Contains(err.Error(), "ready") {
		t.Errorf("error %q does not say the gateway was not ready", err)
	}
	if time.Since(start) > 5*time.Second {
		t.Errorf("gave up after %s, want about the ready timeout", time.Since(start))
	}
	// The half-started gateway was stopped: its port answers nothing.
	if _, err := get("http://" + o.Addr + "/healthz"); err == nil {
		t.Error("the gateway that never became ready is still running")
	}
}

func TestGatewayProcessKilledWhenItIgnoresSIGTERM(t *testing.T) {
	o := fakeGatewayOptions(t, "ignore-term")
	o.StopTimeout = 300 * time.Millisecond
	gw, err := StartGatewayProcess(context.Background(), o)
	if err != nil {
		t.Fatal(err)
	}
	if err := gw.Stop(); err == nil {
		t.Error("want an error reporting that the gateway had to be killed")
	}
	if _, err := get(gw.BaseURL() + "/healthz"); err == nil {
		t.Error("the gateway is still running after being killed")
	}
}

func TestGatewayProcessRefusesABusyPort(t *testing.T) {
	busy := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer busy.Close()

	o := fakeGatewayOptions(t, "ready")
	o.Addr = strings.TrimPrefix(busy.URL, "http://")
	_, err := StartGatewayProcess(context.Background(), o)
	if err == nil || !strings.Contains(err.Error(), "already") {
		t.Fatalf("err = %v, want a refusal because something already serves the port", err)
	}
}

func TestGatewayProcessCanceledWhileWaiting(t *testing.T) {
	o := fakeGatewayOptions(t, "never-ready")
	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()
	if _, err := StartGatewayProcess(ctx, o); err == nil {
		t.Fatal("want an error when the context ends before the gateway is ready")
	}
	if _, err := get("http://" + o.Addr + "/healthz"); err == nil {
		t.Error("the gateway is still running after the start was canceled")
	}
}

func TestGatewayProcessNeedsABinary(t *testing.T) {
	o := fakeGatewayOptions(t, "ready")
	o.Binary = filepath.Join(t.TempDir(), "none")
	if _, err := StartGatewayProcess(context.Background(), o); err == nil {
		t.Fatal("want an error for a binary that does not exist")
	}
}

func TestPSSampleReadsARunningProcess(t *testing.T) {
	smp, err := psSample(os.Getpid())
	if err != nil {
		t.Fatalf("ps: %v (the load benchmark samples the gateway with ps)", err)
	}
	if smp.RSSBytes <= 0 || smp.CPUPercent < 0 {
		t.Errorf("sample of this process = %+v, want a positive resident size", smp)
	}
	if _, err := psSample(1 << 30); err == nil {
		t.Error("want an error for a process that does not exist")
	}
}
