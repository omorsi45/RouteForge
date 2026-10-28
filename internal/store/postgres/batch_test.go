package postgres

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

// recorder is a flush function that remembers its batches, and when each ran.
// fail decides, by call number from 1, whether a call fails.
type recorder struct {
	mu      sync.Mutex
	batches [][]int
	at      []time.Time
	fail    func(call int) error
	gate    chan struct{} // when set, each call waits for it to be closed
}

func (r *recorder) flush(ctx context.Context, rows []int) error {
	r.mu.Lock()
	call := len(r.at) + 1
	r.at = append(r.at, time.Now())
	gate, fail := r.gate, r.fail
	r.mu.Unlock()

	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if fail != nil {
		if err := fail(call); err != nil {
			return err
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.batches = append(r.batches, slices.Clone(rows))
	return nil
}

func (r *recorder) written() [][]int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.batches)
}

func (r *recorder) calls() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.at)
}

func (r *recorder) callTimes() []time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.at)
}

// syncBuf is a goroutine-safe log sink.
type syncBuf struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func mustAdd(t *testing.T, b *batcher[int], rows ...int) {
	t.Helper()
	for _, r := range rows {
		if err := b.Add(r); err != nil {
			t.Fatalf("Add(%d): %v", r, err)
		}
	}
}

func assertBatches(t *testing.T, step string, rec *recorder, want ...[]int) {
	t.Helper()
	got := rec.written()
	if len(got) != len(want) {
		t.Fatalf("after %s: batches = %v, want %v", step, got, want)
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("after %s: batches = %v, want %v", step, got, want)
		}
	}
}

func TestBatchFlushesAtSize(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{}
		b := newBatcher(batcherOptions{BatchSize: 3, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2)
		synctest.Wait()
		assertBatches(t, "two of three rows", rec)

		mustAdd(t, b, 3)
		synctest.Wait()
		assertBatches(t, "the third row", rec, []int{1, 2, 3})

		// Rows that arrive while a flush runs wait for the next one, and a flush
		// takes at most one batch.
		mustAdd(t, b, 4, 5, 6, 7)
		synctest.Wait()
		assertBatches(t, "four more rows", rec, []int{1, 2, 3}, []int{4, 5, 6})
	})
}

func TestBatchFlushesAfterInterval(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{}
		b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: 200 * time.Millisecond, QueueSize: 1000}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		start := time.Now()
		mustAdd(t, b, 1)
		time.Sleep(100 * time.Millisecond)
		mustAdd(t, b, 2) // does not move the deadline: it counts from the first row
		time.Sleep(99 * time.Millisecond)
		synctest.Wait()
		assertBatches(t, "199 ms", rec)

		time.Sleep(time.Millisecond)
		synctest.Wait()
		assertBatches(t, "200 ms", rec, []int{1, 2})
		if got := rec.callTimes()[0].Sub(start); got != 200*time.Millisecond {
			t.Errorf("flushed %v after the first row, want 200ms", got)
		}

		// The next row starts a new interval of its own.
		second := time.Now()
		mustAdd(t, b, 3)
		time.Sleep(199 * time.Millisecond)
		synctest.Wait()
		assertBatches(t, "199 ms into the second interval", rec, []int{1, 2})
		time.Sleep(time.Millisecond)
		synctest.Wait()
		assertBatches(t, "200 ms into the second interval", rec, []int{1, 2}, []int{3})
		if got := rec.callTimes()[1].Sub(second); got != 200*time.Millisecond {
			t.Errorf("second flush %v after its first row, want 200ms", got)
		}
	})
}

func TestBatchCloseFlushes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{}
		b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: time.Hour, QueueSize: 1000}, rec.flush)

		mustAdd(t, b, 1, 2)
		synctest.Wait()
		assertBatches(t, "adding", rec)

		if err := b.Close(context.Background()); err != nil {
			t.Fatalf("Close: %v", err)
		}
		assertBatches(t, "Close", rec, []int{1, 2})

		if err := b.Add(3); !errors.Is(err, errBatcherClosed) {
			t.Errorf("Add after Close = %v, want errBatcherClosed", err)
		}
		if err := b.TryAdd(3); !errors.Is(err, errBatcherClosed) {
			t.Errorf("TryAdd after Close = %v, want errBatcherClosed", err)
		}
		if err := b.Close(context.Background()); err != nil {
			t.Errorf("second Close: %v", err)
		}
		assertBatches(t, "the second Close", rec, []int{1, 2})
	})
}

func TestBatchCloseWaitsForARunningFlush(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{gate: make(chan struct{})}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)

		mustAdd(t, b, 1, 2, 3) // [1 2] is flushing and blocked; 3 is pending
		synctest.Wait()
		closed := make(chan error, 1)
		go func() { closed <- b.Close(context.Background()) }()
		synctest.Wait()
		select {
		case err := <-closed:
			t.Fatalf("Close returned %v while a flush was running", err)
		default:
		}

		close(rec.gate)
		synctest.Wait()
		if err := <-closed; err != nil {
			t.Fatalf("Close: %v", err)
		}
		assertBatches(t, "Close", rec, []int{1, 2}, []int{3})
	})
}

func TestBatchCloseGivesUpWhenItsContextEnds(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		logs := &syncBuf{}
		rec := &recorder{gate: make(chan struct{})} // never opened
		b := newBatcher(batcherOptions{
			BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.NewJSONHandler(logs, nil)),
		}, rec.flush)
		mustAdd(t, b, 1, 2, 3)
		synctest.Wait()

		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		err := b.Close(ctx)
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Close = %v, want the context's deadline error", err)
		}
		if got := b.Dropped(); got != 3 {
			t.Errorf("Dropped = %d, want the 3 rows that were never written", got)
		}
		if !strings.Contains(logs.String(), `"rows":`) {
			t.Errorf("the dropped rows were not logged: %s", logs.String())
		}
	})
}

func TestBatchFullReports(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{gate: make(chan struct{})}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 3}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		if b.Full() {
			t.Fatal("an empty queue is full")
		}
		mustAdd(t, b, 1)
		if b.Full() {
			t.Fatal("full with 1 of 3 rows")
		}
		mustAdd(t, b, 2) // a batch: it flushes, and blocks on the gate
		synctest.Wait()
		if b.Full() {
			t.Fatal("full with 2 of 3 rows in flight")
		}
		// Rows being flushed still count: they are not written yet.
		mustAdd(t, b, 3)
		if !b.Full() {
			t.Fatal("not full with 3 of 3 rows waiting or in flight")
		}
		if err := b.TryAdd(4); !errors.Is(err, errQueueFull) {
			t.Fatalf("TryAdd on a full queue = %v, want errQueueFull", err)
		}
		// Add never refuses a bill: a request admitted before the queue filled up
		// can still write its record.
		mustAdd(t, b, 4)
		if !b.Full() {
			t.Fatal("not full after the extra row")
		}

		close(rec.gate)
		synctest.Wait()
		assertBatches(t, "the gate opened", rec, []int{1, 2}, []int{3, 4})
		if b.Full() {
			t.Fatal("still full after everything was written")
		}
		if err := b.TryAdd(5); err != nil {
			t.Fatalf("TryAdd with room = %v", err)
		}
	})
}

func TestBatchRetriesThenLogs(t *testing.T) {
	t.Run("a flush that keeps failing is retried three times, then dropped and logged", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			logs := &syncBuf{}
			boom := errors.New("connection refused")
			rec := &recorder{fail: func(int) error { return boom }}
			b := newBatcher(batcherOptions{
				BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
				Logger: slog.New(slog.NewJSONHandler(logs, nil)),
			}, rec.flush)
			defer func() { _ = b.Close(context.Background()) }()

			start := time.Now()
			mustAdd(t, b, 1, 2)
			time.Sleep(time.Minute)
			synctest.Wait()

			if got := rec.calls(); got != 4 {
				t.Fatalf("flush ran %d times, want 1 try and 3 retries", got)
			}
			var offsets []time.Duration
			for _, at := range rec.callTimes() {
				offsets = append(offsets, at.Sub(start))
			}
			want := []time.Duration{0, 100 * time.Millisecond, 300 * time.Millisecond, 700 * time.Millisecond}
			if !slices.Equal(offsets, want) {
				t.Errorf("tries at %v, want %v (backoff 100ms, 200ms, 400ms)", offsets, want)
			}
			if got := b.Dropped(); got != 2 {
				t.Errorf("Dropped = %d, want 2", got)
			}
			out := logs.String()
			for _, want := range []string{`"level":"ERROR"`, `"rows":2`, "connection refused"} {
				if !strings.Contains(out, want) {
					t.Errorf("log %s does not contain %s", out, want)
				}
			}
			assertBatches(t, "the failures", rec)

			// The queue is free again: the dropped rows no longer count.
			if b.Full() {
				t.Error("Full after the rows were dropped")
			}
		})
	})

	t.Run("a flush that recovers writes the rows once and logs nothing", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			logs := &syncBuf{}
			rec := &recorder{fail: func(call int) error {
				if call <= 2 {
					return errors.New("deadlock detected")
				}
				return nil
			}}
			b := newBatcher(batcherOptions{
				BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
				Logger: slog.New(slog.NewJSONHandler(logs, nil)),
			}, rec.flush)
			defer func() { _ = b.Close(context.Background()) }()

			mustAdd(t, b, 1, 2)
			time.Sleep(time.Minute)
			synctest.Wait()

			assertBatches(t, "recovery", rec, []int{1, 2})
			if got := rec.calls(); got != 3 {
				t.Errorf("flush ran %d times, want 3", got)
			}
			if got := b.Dropped(); got != 0 {
				t.Errorf("Dropped = %d, want 0", got)
			}
			if out := logs.String(); out != "" {
				t.Errorf("unexpected log output: %s", out)
			}
		})
	})
}

func TestBatchFlushWritesEverythingPending(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2, 3, 4, 5)
		synctest.Wait()
		// The worker took two batches; the fifth row waits.
		assertBatches(t, "adding", rec, []int{1, 2}, []int{3, 4})

		if err := b.Flush(context.Background()); err != nil {
			t.Fatalf("Flush: %v", err)
		}
		assertBatches(t, "Flush", rec, []int{1, 2}, []int{3, 4}, []int{5})
		if err := b.Flush(context.Background()); err != nil {
			t.Fatalf("Flush of nothing: %v", err)
		}
		if got := rec.calls(); got != 3 {
			t.Errorf("flush ran %d times, want 3: an empty Flush writes nothing", got)
		}
	})
}

func TestBatchFlushReturnsTheFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("server closed the connection")
		rec := &recorder{fail: func(int) error { return boom }}
		b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2)
		if err := b.Flush(context.Background()); !errors.Is(err, boom) {
			t.Fatalf("Flush = %v, want the flush error", err)
		}
		if got := b.Dropped(); got != 2 {
			t.Errorf("Dropped = %d, want 2", got)
		}
	})
}

func TestBatchKeepsRowsThatWaitAfterAFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{fail: func(call int) error {
			if call <= 4 { // the whole first batch: one try and three retries
				return errors.New("down")
			}
			return nil
		}}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: 200 * time.Millisecond, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2, 3) // [1 2] fails and is dropped; 3 was never part of it
		time.Sleep(10 * time.Second)
		synctest.Wait()
		assertBatches(t, "the outage", rec, []int{3})
		if got := b.Dropped(); got != 2 {
			t.Errorf("Dropped = %d, want only the 2 rows of the failed batch", got)
		}
	})
}

func TestBatcherDefaults(t *testing.T) {
	b := newBatcher(batcherOptions{}, func(context.Context, []int) error { return nil })
	defer func() { _ = b.Close(context.Background()) }()
	if b.size != defaultBatchSize || b.interval != defaultFlushInterval || b.capacity != defaultQueueSize {
		t.Errorf("defaults = %d, %v, %d; want %d, %v, %d",
			b.size, b.interval, b.capacity, defaultBatchSize, defaultFlushInterval, defaultQueueSize)
	}
	if defaultBatchSize != 500 || defaultFlushInterval != 200*time.Millisecond || defaultQueueSize != 10000 {
		t.Errorf("default constants = %d, %v, %d; want 500, 200ms, 10000", defaultBatchSize, defaultFlushInterval, defaultQueueSize)
	}
}
