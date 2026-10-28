package policy

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

const pollEvery = 5 * time.Second

// pollLog is a goroutine-safe log sink for the poller.
type pollLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *pollLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *pollLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

func (l *pollLog) logger() *slog.Logger { return slog.New(slog.NewJSONHandler(l, nil)) }

// startedPoller runs a poller of r over fp until the returned stop is called.
// It must run inside a synctest bubble.
func startedPoller(t *testing.T, r *Registry, fp *fakePersister, logs *pollLog) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	StartPoller(ctx, r, fp, pollEvery, logs.logger())
	return func() {
		cancel()
		synctest.Wait()
	}
}

func TestPollerFollowsRemoteActivation(t *testing.T) {
	a, b := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[a.Version], fp.raw[b.Version] = a.Raw, b.Raw
		mustAdd(t, r, a)
		mustActivate(t, r, a, "startup")
		fp.setActive(a.Version) // the database agrees with this replica
		// A registry on a database has its persister attached; following another
		// replica must write nothing back to it.
		r.SetPersister(fp)
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(3 * pollEvery)
		synctest.Wait()
		if got := r.Active(); got != a {
			t.Fatalf("Active() = %s while the database agrees, want %s", versionOrEmpty(got), a.Version)
		}
		requireCalls(t, "polling an agreeing database", fp)

		// Another replica activates b.
		fp.setActive(b.Version)
		time.Sleep(2 * pollEvery) // within two ticks
		synctest.Wait()
		if got := r.Active(); got == nil || got.Version != b.Version {
			t.Fatalf("Active() = %s two ticks after a remote activation, want %s", versionOrEmpty(got), b.Version)
		}
		// The policy is loaded from its stored bytes and not saved again.
		requireCalls(t, "following", fp, "load "+b.Version)
		if got := r.Active().Raw; !bytes.Equal(got, b.Raw) {
			t.Error("the followed policy does not carry the stored bytes")
		}
		if got := statuses(r); got[a.Version] != StatusRetired || got[b.Version] != StatusActive {
			t.Errorf("statuses = %v, want a retired and b active", got)
		}
		acts := r.Activations()
		last := acts[len(acts)-1]
		if last.Version != b.Version || last.Previous != a.Version || last.Actor != "poller" {
			t.Errorf("last activation = %+v, want b after a by the poller", last)
		}
		if strings.Contains(logs.String(), `"level":"WARN"`) || strings.Contains(logs.String(), `"level":"ERROR"`) {
			t.Errorf("unexpected warnings: %s", logs.String())
		}

		// Nothing changes while the database stays where it is.
		time.Sleep(3 * pollEvery)
		synctest.Wait()
		requireCalls(t, "polling again", fp, "load "+b.Version)
		if n := len(r.Activations()); n != len(acts) {
			t.Errorf("%d activations after quiet polls, want %d", n, len(acts))
		}
	})
}

func TestPollerStartsFromAnEmptyRegistry(t *testing.T) {
	a, _ := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[a.Version] = a.Raw
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(pollEvery)
		synctest.Wait()
		if r.Active() != nil {
			t.Fatal("a policy became active though the database has none")
		}

		fp.setActive(a.Version)
		time.Sleep(pollEvery)
		synctest.Wait()
		if got := r.Active(); got == nil || got.Version != a.Version {
			t.Fatalf("Active() = %s, want %s", versionOrEmpty(got), a.Version)
		}
		if acts := r.Activations(); len(acts) != 1 || acts[0].Previous != "" {
			t.Errorf("activations = %+v, want one without a previous", acts)
		}
	})
}

func TestPollerUsesAPolicyItAlreadyHolds(t *testing.T) {
	a, b := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		mustAdd(t, r, a, b) // both known here, neither needs loading
		mustActivate(t, r, a, "startup")
		fp.setActive(b.Version)
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(pollEvery)
		synctest.Wait()
		if got := r.Active(); got != b {
			t.Fatalf("Active() = %s, want %s", versionOrEmpty(got), b.Version)
		}
		requireCalls(t, "following a known policy", fp)
	})
}

func TestPollerSurvivesAndReportsFailures(t *testing.T) {
	a, b := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[a.Version], fp.raw[b.Version] = a.Raw, b.Raw
		mustAdd(t, r, a)
		mustActivate(t, r, a, "startup")
		fp.setActive(b.Version)
		fp.activeErr = errors.New("connection refused")
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(4 * pollEvery)
		synctest.Wait()
		if got := r.Active(); got != a {
			t.Fatalf("Active() = %s during an outage, want %s", versionOrEmpty(got), a.Version)
		}
		// Four failed polls name the problem once, not four times.
		if n := strings.Count(logs.String(), "connection refused"); n != 1 {
			t.Errorf("the outage was logged %d times, want 1: %s", n, logs.String())
		}
		if !strings.Contains(logs.String(), `"level":"WARN"`) {
			t.Errorf("the outage was not logged as a warning: %s", logs.String())
		}

		fp.mu.Lock()
		fp.activeErr = nil
		fp.mu.Unlock()
		time.Sleep(pollEvery)
		synctest.Wait()
		if got := r.Active(); got == nil || got.Version != b.Version {
			t.Fatalf("Active() = %s after the outage, want %s", versionOrEmpty(got), b.Version)
		}
		if !strings.Contains(logs.String(), "recovered") {
			t.Errorf("the recovery was not logged: %s", logs.String())
		}
	})
}

func TestPollerKeepsThePolicyWhenTheStoredOneIsBad(t *testing.T) {
	a, b := policyPair(t)
	tests := []struct {
		name  string
		setup func(fp *fakePersister)
		want  string // in the log
	}{
		{"cannot be read", func(fp *fakePersister) { fp.loadErr = errors.New("permission denied") }, "permission denied"},
		{"is not a policy", func(fp *fakePersister) { fp.raw[b.Version] = []byte("{") }, "stored policy "},
		{"is another version", func(fp *fakePersister) { fp.raw[b.Version] = a.Raw }, "holds version " + a.Version},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				r := NewRegistry(poolOrder, testClock())
				fp := newFakePersister()
				tc.setup(fp)
				mustAdd(t, r, a)
				mustActivate(t, r, a, "startup")
				fp.setActive(b.Version)
				logs := &pollLog{}
				stop := startedPoller(t, r, fp, logs)
				defer stop()

				time.Sleep(2 * pollEvery)
				synctest.Wait()
				if got := r.Active(); got != a {
					t.Fatalf("Active() = %s, want %s kept", versionOrEmpty(got), a.Version)
				}
				if len(r.List()) != 1 {
					t.Errorf("a bad stored policy was added: %v", r.List())
				}
				if !strings.Contains(logs.String(), tc.want) {
					t.Errorf("log %s does not contain %q", logs.String(), tc.want)
				}
			})
		})
	}
}

func TestPollerRejectsAPolicyWithAnUnmappedModel(t *testing.T) {
	a, _ := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder[:5], testClock()) // nothing is configured for the sixth model
		fp := newFakePersister()
		fp.raw[a.Version] = a.Raw
		fp.setActive(a.Version)
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(2 * pollEvery)
		synctest.Wait()
		if r.Active() != nil || len(r.List()) != 0 {
			t.Errorf("a policy with an unmapped model was taken: %v", r.List())
		}
		if !strings.Contains(logs.String(), "qwen3-235b-a22b-2507") {
			t.Errorf("the unmapped model is not named in the log: %s", logs.String())
		}
	})
}

// A local activation that lands while a poll is in flight wins: the poll does
// not undo it with a database answer that was read before it. The next poll
// looks again.
func TestPollerYieldsToALocalActivationDuringAPoll(t *testing.T) {
	a, b := policyPair(t)
	c := buildPolicy(t, func(m map[string]any) {
		setAt(t, m, jn("0.456"), "models", 0, "predictor", "init")
	})
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[b.Version] = b.Raw
		mustAdd(t, r, a)
		mustActivate(t, r, a, "startup")
		fp.setActive(b.Version)

		var once sync.Once
		fp.atLoad = func() {
			once.Do(func() { // an admin activates c here, while the poll loads b
				mustAdd(t, r, c)
				mustActivate(t, r, c, "admin")
			})
		}
		logs := &pollLog{}
		stop := startedPoller(t, r, fp, logs)
		defer stop()

		time.Sleep(pollEvery)
		synctest.Wait()
		if got := r.Active(); got != c {
			t.Fatalf("Active() = %s after the racing poll, want the admin's %s", versionOrEmpty(got), c.Version)
		}

		// The database does not know c (this registry has no persister), so the
		// next poll brings b back; what matters is that it looked again.
		time.Sleep(pollEvery)
		synctest.Wait()
		if got := r.Active(); got == nil || got.Version != b.Version {
			t.Fatalf("Active() = %s after the next poll, want %s", versionOrEmpty(got), b.Version)
		}
		versions := []string{}
		for _, act := range r.Activations() {
			versions = append(versions, act.Version)
		}
		if want := []string{a.Version, c.Version, b.Version}; !slices.Equal(versions, want) {
			t.Errorf("activation order = %v, want %v", versions, want)
		}
	})
}

func TestPollerStopsWithItsContext(t *testing.T) {
	a, b := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[a.Version], fp.raw[b.Version] = a.Raw, b.Raw
		mustAdd(t, r, a)
		mustActivate(t, r, a, "startup")
		fp.setActive(a.Version)
		stop := startedPoller(t, r, fp, &pollLog{})
		time.Sleep(pollEvery)
		synctest.Wait()

		stop() // the bubble would deadlock if the goroutine did not exit
		fp.setActive(b.Version)
		time.Sleep(3 * pollEvery)
		synctest.Wait()
		if got := r.Active(); got != a {
			t.Errorf("Active() = %s, want %s: a stopped poller moved the registry", versionOrEmpty(got), a.Version)
		}
	})
}

func TestPollerNilLogger(t *testing.T) {
	a, b := policyPair(t)
	synctest.Test(t, func(t *testing.T) {
		r := NewRegistry(poolOrder, testClock())
		fp := newFakePersister()
		fp.raw[a.Version], fp.raw[b.Version] = a.Raw, b.Raw
		fp.activeErr = errors.New("down")
		ctx, cancel := context.WithCancel(context.Background())
		StartPoller(ctx, r, fp, pollEvery, nil)
		time.Sleep(2 * pollEvery)
		synctest.Wait()
		cancel()
		synctest.Wait()
	})
}
