//go:build unix

package bench

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
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

// statLine builds the line /proc/<pid>/stat holds: 52 fields, of which the
// tests set the command name (field 2), utime (14), stime (15) and rss (24).
func statLine(pid int, comm string, utime, stime, rss int64) string {
	fields := []string{strconv.Itoa(pid), "(" + comm + ")"}
	for n := 3; n <= 52; n++ {
		value := "0"
		switch n {
		case 3:
			value = "S"
		case 14:
			value = strconv.FormatInt(utime, 10)
		case 15:
			value = strconv.FormatInt(stime, 10)
		case 24:
			value = strconv.FormatInt(rss, 10)
		}
		fields = append(fields, value)
	}
	return strings.Join(fields, " ") + "\n"
}

func TestParseProcStat(t *testing.T) {
	for _, comm := range []string{
		"routeforge bench", // spaces in the name
		"routeforge",
		"a) b (c", // unbalanced parentheses
		"(x) y (z)",
		") 1 2 3 (",
		"",
	} {
		t.Run(comm, func(t *testing.T) {
			got, err := parseProcStat([]byte(statLine(77, comm, 120, 30, 456)))
			if err != nil {
				t.Fatal(err)
			}
			if got.cpuTicks != 150 || got.rssPages != 456 {
				t.Errorf("parsed %+v, want 150 CPU ticks (utime + stime) and 456 pages", got)
			}
		})
	}
}

func TestParseProcStatRejectsBadLines(t *testing.T) {
	good := statLine(77, "x", 1, 2, 3)
	fields := strings.Fields(good)
	withField := func(n int, value string) string { // field n counts from 1; the name has no spaces here
		f := append([]string(nil), fields...)
		f[n-1] = value
		return strings.Join(f, " ")
	}
	for _, c := range []struct{ name, line string }{
		{"empty", ""},
		{"no command name", "77 S 1 2 3"},
		{"too few fields", strings.Join(fields[:20], " ")},
		{"utime not a number", withField(14, "many")},
		{"stime not a number", withField(15, "-")},
		{"rss not a number", withField(24, "1.5")},
		{"negative utime", withField(14, "-4")},
		{"negative rss", withField(24, "-4")},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got, err := parseProcStat([]byte(c.line)); err == nil {
				t.Errorf("parsed %q as %+v, want an error", c.line, got)
			}
		})
	}
}

// fakeClock is a clock that moves only when a test says so.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func writeStat(t *testing.T, procDir string, pid int, content string) {
	t.Helper()
	dir := filepath.Join(procDir, strconv.Itoa(pid))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "stat"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// noPS is a ps that fails the test when it is asked.
func noPS(t *testing.T) func(int) (ResourceSample, error) {
	return func(pid int) (ResourceSample, error) {
		t.Errorf("ps was asked about process %d, want /proc/<pid>/stat only", pid)
		return ResourceSample{}, errors.New("ps not expected")
	}
}

func TestProcSamplerComputesCPUPerInterval(t *testing.T) {
	const pid = 4242
	dir := t.TempDir()
	clock := &fakeClock{t: time.Unix(1_000, 0)}
	s := &procSampler{procDir: dir, pageSize: 4096, now: clock.now, ps: noPS(t)}

	// The first reading only sets the baseline: an interval needs two.
	writeStat(t, dir, pid, statLine(pid, "routeforge bench", 100, 40, 300))
	first, err := s.Sample(pid)
	if err != nil {
		t.Fatal(err)
	}
	if first.RSSBytes != 300*4096 || !first.NoCPU || first.CPUMethod != CPUMethodProcStat {
		t.Errorf("first sample = %+v, want 300 pages of 4096 bytes, no CPU yet, method %s", first, CPUMethodProcStat)
	}

	// 50 ticks of CPU time in one second: half of one CPU at 100 ticks/s.
	clock.advance(time.Second)
	writeStat(t, dir, pid, statLine(pid, "routeforge bench", 130, 60, 310))
	second, err := s.Sample(pid)
	if err != nil {
		t.Fatal(err)
	}
	if second.NoCPU || second.CPUMethod != CPUMethodProcStat || second.RSSBytes != 310*4096 {
		t.Errorf("second sample = %+v, want a CPU reading, method %s and 310 pages", second, CPUMethodProcStat)
	}
	requireClose(t, "cpu over 1 s", second.CPUPercent, 50)

	// The next interval is measured from the previous reading, not from the first.
	clock.advance(2 * time.Second)
	writeStat(t, dir, pid, statLine(pid, "routeforge bench", 150, 90, 310)) // +50 ticks in 2 s
	third, err := s.Sample(pid)
	if err != nil {
		t.Fatal(err)
	}
	requireClose(t, "cpu over 2 s", third.CPUPercent, 25)

	// More than one CPU is more than 100 percent, and an idle interval is 0.
	clock.advance(time.Second)
	writeStat(t, dir, pid, statLine(pid, "routeforge bench", 250, 90, 310)) // +100 ticks in 1 s
	fourth, err := s.Sample(pid)
	if err != nil {
		t.Fatal(err)
	}
	requireClose(t, "cpu of a full CPU", fourth.CPUPercent, 100)
	clock.advance(time.Second)
	idle, err := s.Sample(pid) // the file is unchanged
	if err != nil {
		t.Fatal(err)
	}
	requireClose(t, "cpu of an idle interval", idle.CPUPercent, 0)
	if idle.NoCPU {
		t.Error("an idle interval is a reading of 0, not a missing one")
	}
}

func TestProcSamplerFallsBackToPS(t *testing.T) {
	const pid = 4243
	dir := t.TempDir() // no <pid>/stat in it, as on a host without /proc
	var asked []int
	s := &procSampler{
		procDir: dir, pageSize: 4096, now: (&fakeClock{t: time.Unix(1_000, 0)}).now,
		ps: func(pid int) (ResourceSample, error) {
			asked = append(asked, pid)
			return ResourceSample{RSSBytes: 5 << 20, CPUPercent: 7}, nil
		},
	}

	for i := 0; i < 2; i++ {
		got, err := s.Sample(pid)
		if err != nil {
			t.Fatal(err)
		}
		if got.CPUMethod != CPUMethodPSLifetime || got.NoCPU || got.CPUPercent != 7 || got.RSSBytes != 5<<20 {
			t.Errorf("sample %d = %+v, want the ps reading labeled %s", i, got, CPUMethodPSLifetime)
		}
	}
	if len(asked) != 2 || asked[0] != pid || asked[1] != pid {
		t.Errorf("ps was asked about %v, want process %d twice", asked, pid)
	}
}

func TestProcSamplerStartsOverForANewProcess(t *testing.T) {
	dir := t.TempDir()
	clock := &fakeClock{t: time.Unix(1_000, 0)}
	s := &procSampler{procDir: dir, pageSize: 4096, now: clock.now, ps: noPS(t)}

	writeStat(t, dir, 10, statLine(10, "gw", 500, 500, 1))
	writeStat(t, dir, 11, statLine(11, "gw", 5, 5, 1))
	if _, err := s.Sample(10); err != nil {
		t.Fatal(err)
	}
	clock.advance(time.Second)
	// Process 11 has used far less CPU time than process 10, which would read
	// as a counter running backwards if it were measured against it.
	got, err := s.Sample(11)
	if err != nil {
		t.Fatal(err)
	}
	if !got.NoCPU {
		t.Errorf("first sample of a new process = %+v, want only a baseline", got)
	}
}

func TestProcSamplerReportsWhatItCannotRead(t *testing.T) {
	const pid = 4244
	for _, c := range []struct {
		name  string
		setup func(t *testing.T, dir string)
		want  string
	}{
		{"stat is a directory", func(t *testing.T, dir string) {
			if err := os.MkdirAll(filepath.Join(dir, strconv.Itoa(pid), "stat"), 0o755); err != nil {
				t.Fatal(err)
			}
		}, "4244"},
		{"stat is garbage", func(t *testing.T, dir string) { writeStat(t, dir, pid, "garbage\n") }, "4244"},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			c.setup(t, dir)
			s := &procSampler{procDir: dir, pageSize: 4096, now: time.Now, ps: noPS(t)}
			_, err := s.Sample(pid)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err = %v, want one that names process %s", err, c.want)
			}
		})
	}

	t.Run("process vanishes after the first reading", func(t *testing.T) {
		dir := t.TempDir()
		s := &procSampler{procDir: dir, pageSize: 4096, now: time.Now, ps: noPS(t)}
		writeStat(t, dir, pid, statLine(pid, "gw", 1, 1, 1))
		if _, err := s.Sample(pid); err != nil {
			t.Fatal(err)
		}
		if err := os.RemoveAll(filepath.Join(dir, strconv.Itoa(pid))); err != nil {
			t.Fatal(err)
		}
		// A gone process is an error, not a switch to ps: one run is measured one way.
		if _, err := s.Sample(pid); err == nil {
			t.Error("want an error when /proc/<pid>/stat goes away mid-run")
		}
	})

	t.Run("CPU time runs backwards", func(t *testing.T) {
		dir := t.TempDir()
		clock := &fakeClock{t: time.Unix(1_000, 0)}
		s := &procSampler{procDir: dir, pageSize: 4096, now: clock.now, ps: noPS(t)}
		writeStat(t, dir, pid, statLine(pid, "gw", 100, 100, 1))
		if _, err := s.Sample(pid); err != nil {
			t.Fatal(err)
		}
		clock.advance(time.Second)
		writeStat(t, dir, pid, statLine(pid, "gw", 10, 10, 1))
		if _, err := s.Sample(pid); err == nil {
			t.Error("want an error for a CPU time that went down")
		}
	})
}

func TestDefaultSamplerReadsThisProcess(t *testing.T) {
	// On Linux the sampler reads /proc; elsewhere it falls back to ps. Either
	// way it must read the resident size and, from the second reading, the CPU.
	want := CPUMethodPSLifetime
	if _, err := os.Stat(fmt.Sprintf("/proc/%d/stat", os.Getpid())); err == nil {
		want = CPUMethodProcStat
	}
	s := newProcSampler()
	first, err := s.Sample(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	// Burn a little CPU time so that the second interval has something in it.
	deadline := time.Now().Add(60 * time.Millisecond)
	for x := 0; time.Now().Before(deadline); x++ {
		_ = x * x
	}
	second, err := s.Sample(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	for i, smp := range []ResourceSample{first, second} {
		if smp.CPUMethod != want || smp.RSSBytes <= 0 {
			t.Errorf("sample %d = %+v, want method %s and a positive resident size", i, smp, want)
		}
	}
	if want == CPUMethodProcStat && (!first.NoCPU || second.NoCPU || second.CPUPercent < 0) {
		t.Errorf("first = %+v, second = %+v, want a baseline and then an interval reading", first, second)
	}
}
