package bench

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
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
	RSSBytes int64
	// CPUPercent is the CPU use as a percentage of one CPU, so a process that
	// keeps two CPUs busy reads 200. What it covers depends on CPUMethod: the
	// interval since the previous sample, or the whole life of the process.
	CPUPercent float64
	// NoCPU says the sample has no CPU figure, and CPUPercent means nothing:
	// the first reading of a process under CPUMethodProcStat has no interval
	// to measure yet, and only sets the baseline for the next.
	NoCPU bool
	// CPUMethod is how CPUPercent was measured: CPUMethodProcStat or
	// CPUMethodPSLifetime.
	CPUMethod string
}

// The ways CPU use is measured, as ScenarioResult.CPUMethod names them.
const (
	// CPUMethodProcStat is the CPU time (utime and stime) of
	// /proc/<pid>/stat, as a rate over the time since the previous reading.
	CPUMethodProcStat = "proc_stat_interval"
	// CPUMethodPSLifetime is the pcpu of ps, which is the CPU time of the
	// process divided by its age. Sampled every second it gives a running
	// average over the whole life of the process, not the use in the last
	// second.
	CPUMethodPSLifetime = "ps_lifetime_average"
)

// clockTicksPerSecond is the unit of the utime and stime fields of
// /proc/<pid>/stat, USER_HZ, which sysconf(_SC_CLK_TCK) returns. It is 100 on
// Linux on x86_64 and arm64, the hosts the benchmarks run on.
const clockTicksPerSecond = 100

// The fields of /proc/<pid>/stat the sampler reads, numbered from 1 as proc(5)
// numbers them.
const (
	statFieldState = 3  // the first field after the command name
	statFieldUTime = 14 // CPU time in user mode, in clock ticks
	statFieldSTime = 15 // CPU time in kernel mode, in clock ticks
	statFieldRSS   = 24 // resident set size, in pages
)

// procStat is what the sampler reads of one /proc/<pid>/stat line.
type procStat struct {
	cpuTicks int64 // utime + stime
	rssPages int64
}

// parseProcStat reads a /proc/<pid>/stat line. The command name (field 2) is in
// parentheses and may hold spaces and parentheses of its own, so the fields
// after it are found by splitting what follows the last ')': no later field
// holds one.
func parseProcStat(line []byte) (procStat, error) {
	end := bytes.LastIndexByte(line, ')')
	if end < 0 {
		return procStat{}, errors.New("no command name in parentheses")
	}
	fields := strings.Fields(string(line[end+1:]))
	field := func(n int) (int64, error) {
		i := n - statFieldState
		if i >= len(fields) {
			return 0, fmt.Errorf("field %d is missing: %d fields follow the command name", n, len(fields))
		}
		v, err := strconv.ParseInt(fields[i], 10, 64)
		if err != nil || v < 0 {
			return 0, fmt.Errorf("field %d is %q, want a non-negative integer", n, fields[i])
		}
		return v, nil
	}
	utime, err := field(statFieldUTime)
	if err != nil {
		return procStat{}, err
	}
	stime, err := field(statFieldSTime)
	if err != nil {
		return procStat{}, err
	}
	rss, err := field(statFieldRSS)
	if err != nil {
		return procStat{}, err
	}
	return procStat{cpuTicks: utime + stime, rssPages: rss}, nil
}

// intervalCPUPercent is the CPU use over an interval: the CPU time used in it,
// as a percentage of the time that passed.
func intervalCPUPercent(deltaTicks int64, interval time.Duration) float64 {
	return 100 * float64(deltaTicks) / (clockTicksPerSecond * interval.Seconds())
}

// procSampler reads the memory and CPU use of a process from
// /proc/<pid>/stat, in the same read, so that both belong to the same moment.
// Where the process has no such file, as on a host without /proc, it uses ps.
//
// CPU use is a rate over the time since the previous reading, so a procSampler
// keeps the previous reading of the process it follows, and starts over when it
// is asked about another. It is for one goroutine at a time.
type procSampler struct {
	procDir  string // where /proc is
	pageSize int64
	now      func() time.Time
	ps       func(pid int) (ResourceSample, error)

	pid       int
	method    string // CPUMethodProcStat or CPUMethodPSLifetime, once known for pid
	havePrev  bool
	prevTicks int64
	prevAt    time.Time
}

func newProcSampler() *procSampler {
	return &procSampler{procDir: "/proc", pageSize: int64(os.Getpagesize()), now: time.Now, ps: psSample}
}

// Sample reads process pid. It is a SampleFunc. The first reading of a process
// has no CPU figure (NoCPU), since a rate needs two. A process is measured one
// way for as long as it is followed: one whose /proc file was there and goes
// away is an error, not a reason to switch to ps.
func (s *procSampler) Sample(pid int) (ResourceSample, error) {
	if pid != s.pid {
		s.pid, s.method, s.havePrev = pid, "", false
	}
	if s.method == CPUMethodPSLifetime {
		return s.samplePS(pid)
	}
	data, err := os.ReadFile(filepath.Join(s.procDir, strconv.Itoa(pid), "stat"))
	if errors.Is(err, fs.ErrNotExist) && s.method == "" {
		s.method = CPUMethodPSLifetime
		return s.samplePS(pid)
	}
	if err != nil {
		return ResourceSample{}, fmt.Errorf("read the stat of process %d: %w", pid, err)
	}
	s.method = CPUMethodProcStat
	at := s.now()
	st, err := parseProcStat(data)
	if err != nil {
		return ResourceSample{}, fmt.Errorf("read the stat of process %d: %w", pid, err)
	}

	smp := ResourceSample{RSSBytes: st.rssPages * s.pageSize, CPUMethod: CPUMethodProcStat}
	switch interval := at.Sub(s.prevAt); {
	case !s.havePrev || interval <= 0:
		smp.NoCPU = true
	case st.cpuTicks < s.prevTicks:
		return ResourceSample{}, fmt.Errorf("the CPU time of process %d went down from %d to %d ticks", pid, s.prevTicks, st.cpuTicks)
	default:
		smp.CPUPercent = intervalCPUPercent(st.cpuTicks-s.prevTicks, interval)
	}
	s.havePrev, s.prevTicks, s.prevAt = true, st.cpuTicks, at
	return smp, nil
}

func (s *procSampler) samplePS(pid int) (ResourceSample, error) {
	smp, err := s.ps(pid)
	if err != nil {
		return ResourceSample{}, err
	}
	smp.CPUMethod = CPUMethodPSLifetime
	return smp, nil
}

// psSample reads the resident size and CPU use of process pid with ps. The CPU
// figure is ps's pcpu, an average over the life of the process.
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
