package reliability

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

// The timing tests run in a synctest bubble, whose clock is fake: a Sleep
// returns as soon as every goroutine in the bubble is blocked, taking no real
// time, so a queue timeout of seconds is tested without waiting.

func oneModel(limit QueueLimit) *Queues {
	return NewQueues(map[string]QueueLimit{"m": limit})
}

func mustAcquire(t *testing.T, q *Queues, model string) func() {
	t.Helper()
	release, err := q.Acquire(context.Background(), model)
	if err != nil {
		t.Fatalf("Acquire(%q) = %v, want a slot", model, err)
	}
	return release
}

// result is what a goroutine blocked in Acquire eventually gets.
type result struct {
	release func()
	err     error
}

// acquireAsync starts Acquire in a goroutine and returns the channel its
// result arrives on.
func acquireAsync(ctx context.Context, q *Queues, model string) <-chan result {
	out := make(chan result, 1)
	go func() {
		release, err := q.Acquire(ctx, model)
		out <- result{release, err}
	}()
	return out
}

func TestQueueRunsBelowConcurrency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 2, QueueSize: 1, Timeout: time.Second})
		start := time.Now()
		a := mustAcquire(t, q, "m")
		b := mustAcquire(t, q, "m")
		if got := time.Since(start); got != 0 {
			t.Errorf("Acquire below the concurrency limit took %v, want no wait", got)
		}
		if q.Depth("m") != 0 || q.Full("m") {
			t.Errorf("Depth, Full = %d, %v with two busy slots and an empty line, want 0, false", q.Depth("m"), q.Full("m"))
		}
		a()
		b()
	})
}

func TestQueueWaitsThenRuns(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 2, Timeout: time.Second})
		first := mustAcquire(t, q, "m")

		second := acquireAsync(context.Background(), q, "m")
		synctest.Wait()
		third := acquireAsync(context.Background(), q, "m")
		synctest.Wait()
		if got := q.Depth("m"); got != 2 {
			t.Fatalf("Depth = %d, want 2 waiting", got)
		}
		select {
		case <-second:
			t.Fatal("a waiter ran while the slot was busy")
		default:
		}

		// The line is served first in, first out, one slot at a time.
		first()
		synctest.Wait()
		r2 := <-second
		if r2.err != nil {
			t.Fatalf("second waiter: %v", r2.err)
		}
		select {
		case <-third:
			t.Fatal("the third request ran before the second released")
		default:
		}
		if got := q.Depth("m"); got != 1 {
			t.Fatalf("Depth = %d, want 1 waiting", got)
		}

		r2.release()
		synctest.Wait()
		r3 := <-third
		if r3.err != nil {
			t.Fatalf("third waiter: %v", r3.err)
		}
		r3.release()
		if got := q.Depth("m"); got != 0 {
			t.Fatalf("Depth = %d, want 0", got)
		}
	})
}

func TestQueueFullRejects(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 1, Timeout: time.Hour})
		first := mustAcquire(t, q, "m")
		waiter := acquireAsync(context.Background(), q, "m")
		synctest.Wait()

		start := time.Now()
		release, err := q.Acquire(context.Background(), "m")
		if !errors.Is(err, ErrQueueFull) || release != nil {
			t.Fatalf("Acquire with slots and line full = %v, release set: %v; want ErrQueueFull", err, release != nil)
		}
		if got := time.Since(start); got != 0 {
			t.Errorf("a full queue made the caller wait %v, want an immediate refusal", got)
		}

		first()
		synctest.Wait()
		(<-waiter).release()
	})

	t.Run("no waiting room", func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 0, Timeout: time.Hour})
		held := mustAcquire(t, q, "m")
		if _, err := q.Acquire(context.Background(), "m"); !errors.Is(err, ErrQueueFull) {
			t.Fatalf("Acquire with QueueSize 0 = %v, want ErrQueueFull", err)
		}
		held()
		mustAcquire(t, q, "m")()
	})
}

func TestQueueTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 1, Timeout: 2 * time.Second})
		held := mustAcquire(t, q, "m")
		start := time.Now()
		waiter := acquireAsync(context.Background(), q, "m")

		time.Sleep(2*time.Second - time.Nanosecond)
		synctest.Wait()
		select {
		case r := <-waiter:
			t.Fatalf("waiter gave up early: %v", r.err)
		default:
		}
		if got := q.Depth("m"); got != 1 {
			t.Fatalf("Depth = %d, want 1 while the waiter is within its timeout", got)
		}

		time.Sleep(time.Nanosecond)
		synctest.Wait()
		r := <-waiter
		if !errors.Is(r.err, ErrQueueTimeout) || r.release != nil {
			t.Fatalf("waiter = %v, release set: %v; want ErrQueueTimeout", r.err, r.release != nil)
		}
		if got := time.Since(start); got != 2*time.Second {
			t.Errorf("waited %v, want the 2s timeout", got)
		}
		if got := q.Depth("m"); got != 0 {
			t.Fatalf("Depth = %d after the timeout, want 0", got)
		}

		// The timed-out waiter took no slot with it.
		held()
		mustAcquire(t, q, "m")()
	})
}

func TestQueueContextCancelLeavesLine(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 2, Timeout: time.Hour})
		held := mustAcquire(t, q, "m")

		ctx, cancel := context.WithCancel(context.Background())
		leaving := acquireAsync(ctx, q, "m")
		staying := acquireAsync(context.Background(), q, "m")
		synctest.Wait()
		if got := q.Depth("m"); got != 2 {
			t.Fatalf("Depth = %d, want 2", got)
		}

		cancel()
		synctest.Wait()
		r := <-leaving
		if !errors.Is(r.err, context.Canceled) || r.release != nil {
			t.Fatalf("canceled waiter = %v, release set: %v; want context.Canceled", r.err, r.release != nil)
		}
		if got := q.Depth("m"); got != 1 {
			t.Fatalf("Depth = %d after the cancel, want 1", got)
		}

		// The slot goes to the waiter that stayed, not to the one that left.
		held()
		synctest.Wait()
		stayed := <-staying
		if stayed.err != nil {
			t.Fatalf("remaining waiter: %v", stayed.err)
		}
		stayed.release()
		if got := q.Depth("m"); got != 0 {
			t.Fatalf("Depth = %d, want 0", got)
		}
	})

	t.Run("deadline", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 1, Timeout: time.Hour})
			held := mustAcquire(t, q, "m")
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if _, err := q.Acquire(ctx, "m"); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("Acquire = %v, want context.DeadlineExceeded", err)
			}
			if got := q.Depth("m"); got != 0 {
				t.Fatalf("Depth = %d, want 0", got)
			}
			held()
		})
	})
}

func TestQueueDeadContextTakesNoSlot(t *testing.T) {
	q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	release, err := q.Acquire(ctx, "m")
	if !errors.Is(err, context.Canceled) || release != nil {
		t.Fatalf("Acquire with a canceled context = %v, release set: %v; want context.Canceled", err, release != nil)
	}
	// The slot is still free.
	mustAcquire(t, q, "m")()
}

func TestQueueReleaseIdempotent(t *testing.T) {
	q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second})
	release := mustAcquire(t, q, "m")
	release()
	release()
	release()

	// Had the extra calls freed more slots, two requests would now run.
	held := mustAcquire(t, q, "m")
	if _, err := q.Acquire(context.Background(), "m"); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("second Acquire = %v, want ErrQueueFull: a repeated release freed an extra slot", err)
	}
	held()
}

func TestQueueFullReport(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 1, Timeout: time.Hour})
		if q.Full("m") {
			t.Fatal("Full on an idle queue")
		}
		first := mustAcquire(t, q, "m")
		if q.Full("m") {
			t.Fatal("Full with all slots busy but a free place in line")
		}

		waiter := acquireAsync(context.Background(), q, "m")
		synctest.Wait()
		if !q.Full("m") {
			t.Fatal("Full = false with all slots busy and the line full")
		}

		first() // the waiter takes the slot; the line is empty again
		synctest.Wait()
		r := <-waiter
		if q.Full("m") {
			t.Fatal("Full after the line moved up")
		}
		r.release()
		if q.Full("m") || q.Depth("m") != 0 {
			t.Fatalf("Full, Depth = %v, %d on an idle queue", q.Full("m"), q.Depth("m"))
		}
	})

	t.Run("no waiting room is full as soon as the slots are", func(t *testing.T) {
		q := oneModel(QueueLimit{MaxConcurrency: 2, QueueSize: 0, Timeout: time.Second})
		a := mustAcquire(t, q, "m")
		if q.Full("m") {
			t.Fatal("Full with a free slot")
		}
		b := mustAcquire(t, q, "m")
		if !q.Full("m") {
			t.Fatal("Full = false with every slot busy and no waiting room")
		}
		a()
		b()
	})
}

func TestQueueModelsAreIndependent(t *testing.T) {
	q := NewQueues(map[string]QueueLimit{
		"a": {MaxConcurrency: 1, QueueSize: 0, Timeout: time.Second},
		"b": {MaxConcurrency: 3, QueueSize: 0, Timeout: time.Second},
	})
	heldA := mustAcquire(t, q, "a")
	if _, err := q.Acquire(context.Background(), "a"); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("second Acquire on a = %v, want ErrQueueFull", err)
	}
	if !q.Full("a") || q.Full("b") {
		t.Fatalf("Full(a), Full(b) = %v, %v; want true, false", q.Full("a"), q.Full("b"))
	}
	for range 3 {
		defer mustAcquire(t, q, "b")()
	}
	heldA()
}

func TestQueueUnknownModel(t *testing.T) {
	q := oneModel(QueueLimit{MaxConcurrency: 1, Timeout: time.Second})
	release, err := q.Acquire(context.Background(), "other")
	if err == nil || release != nil {
		t.Fatalf("Acquire for an unconfigured model = %v, release set: %v; want an error", err, release != nil)
	}
	if q.Full("other") || q.Depth("other") != 0 {
		t.Fatalf("Full, Depth = %v, %d for an unconfigured model, want false, 0", q.Full("other"), q.Depth("other"))
	}
}

func TestQueueNeverExceedsConcurrency(t *testing.T) {
	const limit = 3
	q := oneModel(QueueLimit{MaxConcurrency: limit, QueueSize: 1000, Timeout: time.Hour})
	var (
		running, peak, done atomic.Int32
		wg                  sync.WaitGroup
	)
	for range 40 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 25 {
				release, err := q.Acquire(context.Background(), "m")
				if err != nil {
					t.Errorf("Acquire: %v", err)
					return
				}
				n := running.Add(1)
				for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
				}
				runtime.Gosched() // let other goroutines overlap with this one
				running.Add(-1)
				release()
				done.Add(1)
			}
		}()
	}
	wg.Wait()
	if got := done.Load(); got != 40*25 {
		t.Fatalf("%d requests completed, want %d", got, 40*25)
	}
	if got := peak.Load(); got > limit {
		t.Fatalf("%d requests ran at once, limit %d", got, limit)
	}
	if q.Depth("m") != 0 || q.Full("m") {
		t.Fatalf("Depth, Full = %d, %v after the run, want 0, false", q.Depth("m"), q.Full("m"))
	}
}

// A slot handed to a waiter at the moment the waiter gives up must be neither
// lost nor counted twice. Either outcome is legal; what matters is that the
// slot count is right afterwards.
func TestQueueGrantRacingCancelKeepsSlotCount(t *testing.T) {
	for range 300 {
		q := oneModel(QueueLimit{MaxConcurrency: 1, QueueSize: 1, Timeout: time.Hour})
		held := mustAcquire(t, q, "m")
		ctx, cancel := context.WithCancel(context.Background())
		waiter := acquireAsync(ctx, q, "m")
		for q.Depth("m") != 1 {
			runtime.Gosched() // wait for the waiter to join the line
		}

		cancel()
		held()
		r := <-waiter
		if r.err == nil {
			r.release()
		} else if !errors.Is(r.err, context.Canceled) {
			t.Fatalf("waiter: %v", r.err)
		}

		if got := q.Depth("m"); got != 0 {
			t.Fatalf("Depth = %d, want 0", got)
		}
		if got := slotsInUse(q, "m"); got != 0 {
			t.Fatalf("%d slots in use after everything was released, want 0", got)
		}
	}
}

// slotsInUse returns how many slots of model are taken.
func slotsInUse(q *Queues, model string) int {
	mq := q.byModel[model]
	mq.mu.Lock()
	defer mq.mu.Unlock()
	return mq.running
}
