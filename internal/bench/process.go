package bench

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// DefaultGatewayAddr is where the benchmarks run the gateway: a loopback port
// no other service of the project uses.
const DefaultGatewayAddr = "127.0.0.1:18080"

// Gateway is a running gateway that a benchmark sends requests to.
type Gateway interface {
	// BaseURL is the address clients send requests to.
	BaseURL() string
	// PID is the process to sample for memory and CPU.
	PID() int
	// Stop ends the gateway and waits for it. It reports a gateway that had
	// to be killed or that was no longer running.
	Stop() error
}

// ProcessOptions says how to start a gateway process.
type ProcessOptions struct {
	// Binary is the gateway executable, started as `<binary> serve --config
	// <config> --listen <addr>`.
	Binary     string
	ConfigPath string
	// Addr is the host:port to listen on; the default is DefaultGatewayAddr.
	Addr string
	// LogPath receives the gateway's output. Its directory is created. The
	// gateway logs no prompt or completion text.
	LogPath string
	// Env is added to the environment of the process.
	Env []string
	// ReadyTimeout is how long to wait for /readyz to answer 200 (default
	// 30 s) and StopTimeout how long a stop waits before it kills the process
	// (default 35 s: the gateway's own drain is bounded by 30 s).
	ReadyTimeout time.Duration
	StopTimeout  time.Duration
}

// GatewayProcess is a gateway process started by StartGatewayProcess.
type GatewayProcess struct {
	cmd         *exec.Cmd
	addr        string
	logPath     string
	log         *os.File
	stopTimeout time.Duration

	exited  chan struct{} // closed once the process has been waited for
	waitErr error         // valid once exited is closed

	stopOnce sync.Once
	stopErr  error
}

var _ Gateway = (*GatewayProcess)(nil)

// probeClient asks a gateway a quick question. It keeps no connection, so a
// probe never holds a port open.
var probeClient = &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}

// probe GETs path of the gateway at addr and returns the status.
func probe(ctx context.Context, addr, path string) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+addr+path, nil)
	if err != nil {
		return 0, err
	}
	resp, err := probeClient.Do(req)
	if err != nil {
		return 0, err
	}
	_ = resp.Body.Close()
	return resp.StatusCode, nil
}

// StartGatewayProcess starts a gateway and waits until it is ready. It refuses
// to start when something already answers on the address, since a stale
// gateway would be measured in place of the new one. On any failure the process
// it started is stopped.
func StartGatewayProcess(ctx context.Context, o ProcessOptions) (*GatewayProcess, error) {
	if o.Addr == "" {
		o.Addr = DefaultGatewayAddr
	}
	if o.ReadyTimeout <= 0 {
		o.ReadyTimeout = 30 * time.Second
	}
	if o.StopTimeout <= 0 {
		o.StopTimeout = 35 * time.Second
	}
	if status, err := probe(ctx, o.Addr, "/healthz"); err == nil {
		return nil, fmt.Errorf("start gateway: something is already serving on %s (status %d)", o.Addr, status)
	}
	if err := os.MkdirAll(filepath.Dir(o.LogPath), 0o755); err != nil {
		return nil, fmt.Errorf("start gateway: %w", err)
	}
	logFile, err := os.OpenFile(o.LogPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, fmt.Errorf("start gateway: %w", err)
	}

	cmd := exec.Command(o.Binary, "serve", "--config", o.ConfigPath, "--listen", o.Addr)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	cmd.Env = append(os.Environ(), o.Env...)
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return nil, fmt.Errorf("start gateway: %w", err)
	}
	g := &GatewayProcess{
		cmd:         cmd,
		addr:        o.Addr,
		logPath:     o.LogPath,
		log:         logFile,
		stopTimeout: o.StopTimeout,
		exited:      make(chan struct{}),
	}
	go func() {
		g.waitErr = cmd.Wait()
		close(g.exited)
	}()

	if err := g.waitReady(ctx, o.ReadyTimeout); err != nil {
		if stopErr := g.stop(false); stopErr != nil {
			err = errors.Join(err, stopErr)
		}
		return nil, err
	}
	return g, nil
}

// waitReady polls /readyz until it answers 200.
func (g *GatewayProcess) waitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-g.exited:
			return fmt.Errorf("start gateway: it exited before it was ready (%v); see %s", g.waitErr, g.logPath)
		case <-ctx.Done():
			return fmt.Errorf("start gateway: waiting for it to be ready: %w", ctx.Err())
		case <-deadline.C:
			return fmt.Errorf("start gateway: it was not ready within %s; see %s", timeout, g.logPath)
		case <-tick.C:
		}
		if status, err := probe(ctx, g.addr, "/readyz"); err == nil && status == http.StatusOK {
			return nil
		}
	}
}

// BaseURL implements Gateway.
func (g *GatewayProcess) BaseURL() string { return "http://" + g.addr }

// PID implements Gateway.
func (g *GatewayProcess) PID() int { return g.cmd.Process.Pid }

// Stop sends the gateway SIGTERM, which makes it drain and exit, and waits for
// it. A gateway that does not exit within StopTimeout is killed, and a gateway
// that was no longer running is reported: either way the numbers measured
// against it need a second look. Stop is safe to call more than once.
func (g *GatewayProcess) Stop() error {
	g.stopOnce.Do(func() { g.stopErr = g.stop(true) })
	return g.stopErr
}

// stop ends the process and closes its log. It treats a process that has
// already exited as an error when reportEarlyExit is set.
func (g *GatewayProcess) stop(reportEarlyExit bool) error {
	defer g.log.Close()
	select {
	case <-g.exited:
		if reportEarlyExit {
			return fmt.Errorf("the gateway exited before it was stopped (%v); see %s", g.waitErr, g.logPath)
		}
		return nil
	default:
	}
	if err := g.cmd.Process.Signal(syscall.SIGTERM); err != nil && !errors.Is(err, os.ErrProcessDone) {
		_ = g.cmd.Process.Kill()
		<-g.exited
		return fmt.Errorf("signal the gateway: %w", err)
	}
	timer := time.NewTimer(g.stopTimeout)
	defer timer.Stop()
	select {
	case <-g.exited:
		if g.waitErr != nil {
			return fmt.Errorf("the gateway did not exit cleanly after SIGTERM: %w; see %s", g.waitErr, g.logPath)
		}
		return nil
	case <-timer.C:
		_ = g.cmd.Process.Kill()
		<-g.exited
		return fmt.Errorf("the gateway did not stop within %s of SIGTERM and was killed; see %s", g.stopTimeout, g.logPath)
	}
}

// ResourceSample is the memory and CPU of a process at one moment.
type ResourceSample struct {
	RSSBytes   int64
	CPUPercent float64 // as ps reports it: CPU time over the whole life of the process
}

// psSample reads the resident size and CPU use of process pid with ps.
func psSample(pid int) (ResourceSample, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ps", "-o", "rss=,pcpu=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return ResourceSample{}, fmt.Errorf("ps -p %d: %w", pid, err)
	}
	return parsePS(out)
}

// parsePS reads the output of `ps -o rss=,pcpu=`: the resident size in KiB and
// the CPU percentage, on one line.
func parsePS(out []byte) (ResourceSample, error) {
	fields := strings.Fields(string(out))
	if len(fields) != 2 {
		return ResourceSample{}, fmt.Errorf("unexpected ps output %q, want the resident size and the CPU percentage", strings.TrimSpace(string(out)))
	}
	kib, err := strconv.ParseInt(fields[0], 10, 64)
	if err != nil || kib < 0 {
		return ResourceSample{}, fmt.Errorf("unexpected resident size %q in ps output", fields[0])
	}
	cpu, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return ResourceSample{}, fmt.Errorf("unexpected CPU percentage %q in ps output", fields[1])
	}
	return ResourceSample{RSSBytes: kib * 1024, CPUPercent: cpu}, nil
}
