package reliability

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestInFlightLimit(t *testing.T) {
	f := NewInFlight(2)
	if f.Count() != 0 {
		t.Fatalf("Count = %d on a new limiter, want 0", f.Count())
	}

	r1, ok := f.TryAcquire()
	if !ok {
		t.Fatal("first TryAcquire refused")
	}
	r2, ok := f.TryAcquire()
	if !ok {
		t.Fatal("second TryAcquire refused")
	}
	if f.Count() != 2 {
		t.Fatalf("Count = %d, want 2", f.Count())
	}
	if release, ok := f.TryAcquire(); ok || release != nil {
		t.Fatalf("third TryAcquire = release set: %v, ok: %v; want a refusal", release != nil, ok)
	}
	if f.Count() != 2 {
		t.Fatalf("Count = %d after a refusal, want 2", f.Count())
	}

	r1()
	if f.Count() != 1 {
		t.Fatalf("Count = %d after a release, want 1", f.Count())
	}
	r3, ok := f.TryAcquire()
	if !ok {
		t.Fatal("TryAcquire refused after a release")
	}
	r2()
	r3()
	if f.Count() != 0 {
		t.Fatalf("Count = %d after every release, want 0", f.Count())
	}
}

func TestInFlightReleaseIdempotent(t *testing.T) {
	f := NewInFlight(1)
	release, ok := f.TryAcquire()
	if !ok {
		t.Fatal("TryAcquire refused")
	}
	release()
	release()
	if f.Count() != 0 {
		t.Fatalf("Count = %d after a repeated release, want 0", f.Count())
	}

	// Had the repeat freed a second slot, two requests would now fit.
	held, ok := f.TryAcquire()
	if !ok {
		t.Fatal("TryAcquire refused after a release")
	}
	if _, ok := f.TryAcquire(); ok {
		t.Fatal("a repeated release freed an extra slot")
	}
	held()
}

func TestInFlightConcurrent(t *testing.T) {
	const limit = 10
	f := NewInFlight(limit)
	var (
		granted atomic.Int32
		start   = make(chan struct{})
		wg      sync.WaitGroup
	)
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, ok := f.TryAcquire(); ok {
				granted.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	if got := granted.Load(); got != limit {
		t.Fatalf("%d of 100 simultaneous callers admitted, want exactly %d", got, limit)
	}
	if f.Count() != limit {
		t.Fatalf("Count = %d, want %d", f.Count(), limit)
	}
}
