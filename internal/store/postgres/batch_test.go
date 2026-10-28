package postgres

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
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

// offsets returns when each call ran, counted from start.
func (r *recorder) offsets(start time.Time) []time.Duration {
	var out []time.Duration
	for _, at := range r.callTimes() {
		out = append(out, at.Sub(start))
	}
	return out
}

// gaps returns the waits between consecutive calls.
func (r *recorder) gaps() []time.Duration {
	times := r.callTimes()
	var out []time.Duration
	for i := 1; i < len(times); i++ {
		out = append(out, times[i].Sub(times[i-1]))
	}
	return out
}

// failUntil makes the first n calls fail.
func failUntil(n int) func(call int) error {
	return func(call int) error {
		if call <= n {
			return errors.New("connection refused")
		}
		return nil
	}
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

// queuedRows is how many rows are pending or being written.
func queuedRows(b *batcher[int]) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pending) + b.inflight
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
		if !strings.Contains(logs.String(), `"rows":3`) {
			t.Errorf("the dropped rows were not logged with their count: %s", logs.String())
		}
	})
}

func TestBatchCloseWithAnExpiredContextDropsTheQueuedRows(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		logs := &syncBuf{}
		rec := &recorder{fail: failUntil(1 << 30)}
		b := newBatcher(batcherOptions{
			BatchSize: 500, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.NewJSONHandler(logs, nil)),
		}, rec.flush)
		mustAdd(t, b, 1, 2, 3)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := b.Close(ctx)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Close = %v, want the context's error", err)
		}
		if got := b.Dropped(); got != 3 {
			t.Errorf("Dropped = %d, want the 3 queued rows", got)
		}
		if got := rec.calls(); got != 0 {
			t.Errorf("flush ran %d times for a context that had ended, want 0", got)
		}
		if !strings.Contains(logs.String(), `"rows":3`) {
			t.Errorf("the dropped rows were not logged with their count: %s", logs.String())
		}
		if b.Full() {
			t.Error("Full after the queued rows were dropped")
		}
	})
}

func TestBatchCloseKeepsTryingUntilItsContextEnds(t *testing.T) {
	t.Run("the rows are written when the database comes back in time", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			rec := &recorder{fail: failUntil(3)}
			b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)
			mustAdd(t, b, 1, 2, 3)

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := b.Close(ctx); err != nil {
				t.Fatalf("Close: %v", err)
			}
			assertBatches(t, "Close", rec, []int{1, 2, 3})
			if got := b.Dropped(); got != 0 {
				t.Errorf("Dropped = %d, want 0", got)
			}
		})
	})

	t.Run("the rows left when it ends are dropped and counted", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			rec := &recorder{fail: failUntil(1 << 30)}
			b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: time.Hour, QueueSize: 100}, rec.flush)
			mustAdd(t, b, 1, 2, 3)

			start := time.Now()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			err := b.Close(ctx)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("Close = %v, want the context's deadline error", err)
			}
			want := []time.Duration{0, 100 * time.Millisecond, 300 * time.Millisecond, 700 * time.Millisecond}
			if got := rec.offsets(start); !slices.Equal(got, want) {
				t.Errorf("tries at %v, want %v (backoff 100ms, 200ms, 400ms, then 800ms would pass the deadline)", got, want)
			}
			if got := b.Dropped(); got != 3 {
				t.Errorf("Dropped = %d, want 3", got)
			}
			assertBatches(t, "the failed Close", rec)
		})
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

func TestBatchFailedBatchStaysQueuedUntilWritten(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{fail: failUntil(6)}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		start := time.Now()
		mustAdd(t, b, 1, 2, 3, 4, 5)
		time.Sleep(time.Minute)
		synctest.Wait()

		// Every row is written exactly once and in the order it was added; the
		// failed batch went ahead of the rows behind it.
		assertBatches(t, "the recovery", rec, []int{1, 2}, []int{3, 4}, []int{5})
		if got := b.Dropped(); got != 0 {
			t.Errorf("Dropped = %d, want 0: a batcher that is open never drops rows", got)
		}
		if got := rec.calls(); got != 9 {
			t.Errorf("flush ran %d times, want 6 failed tries and 3 batches", got)
		}
		want := []time.Duration{0, 100 * time.Millisecond, 300 * time.Millisecond, 700 * time.Millisecond,
			1500 * time.Millisecond, 3100 * time.Millisecond, 6300 * time.Millisecond}
		if got := rec.offsets(start)[:len(want)]; !slices.Equal(got, want) {
			t.Errorf("tries at %v, want %v (backoff 100ms doubling)", got, want)
		}
		if b.Full() {
			t.Error("Full after everything was written")
		}
	})
}

func TestBatchRowsAddedDuringAnOutageStayBehindTheFailedBatch(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{fail: failUntil(2)}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2) // fails at 0 and at 100 ms, then is written at 300 ms
		time.Sleep(50 * time.Millisecond)
		mustAdd(t, b, 3, 4)
		time.Sleep(time.Minute)
		synctest.Wait()

		assertBatches(t, "the recovery", rec, []int{1, 2}, []int{3, 4})
		if got := b.Dropped(); got != 0 {
			t.Errorf("Dropped = %d, want 0", got)
		}
	})
}

func TestBatchBackoffDoublesUpToFiveSeconds(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{fail: failUntil(10)}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2)
		time.Sleep(time.Minute)
		synctest.Wait()

		want := []time.Duration{
			100 * time.Millisecond, 200 * time.Millisecond, 400 * time.Millisecond, 800 * time.Millisecond,
			1600 * time.Millisecond, 3200 * time.Millisecond,
			5 * time.Second, 5 * time.Second, 5 * time.Second, 5 * time.Second,
		}
		if got := rec.gaps(); !slices.Equal(got, want) {
			t.Errorf("waits between tries = %v, want %v", got, want)
		}
		assertBatches(t, "the recovery", rec, []int{1, 2})
	})
}

func TestBatchBackoffStartsOverAfterASuccess(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rec := &recorder{fail: func(call int) error {
			if call <= 3 || call == 5 {
				return errors.New("connection refused")
			}
			return nil
		}}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2) // tries at 0, 100 and 300 ms fail, the one at 700 ms writes
		time.Sleep(time.Second)
		synctest.Wait()
		assertBatches(t, "the first outage", rec, []int{1, 2})

		mustAdd(t, b, 3, 4) // fails once, a long time after the first outage
		time.Sleep(time.Minute)
		synctest.Wait()
		assertBatches(t, "the second outage", rec, []int{1, 2}, []int{3, 4})

		gaps := rec.gaps()
		if len(gaps) != 5 {
			t.Fatalf("flush ran %d times, want 6", len(gaps)+1)
		}
		if gaps[4] != 100*time.Millisecond {
			t.Errorf("the wait after the first failure of the second outage = %v, want 100ms: a success starts the backoff over", gaps[4])
		}
	})
}

func TestBatchLogsEachFailedTryOnce(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		logs := &syncBuf{}
		rec := &recorder{fail: failUntil(1 << 30)}
		b := newBatcher(batcherOptions{
			BatchSize: 2, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.NewJSONHandler(logs, nil)),
		}, rec.flush)

		mustAdd(t, b, 1, 2)
		time.Sleep(time.Minute)
		synctest.Wait()

		out := logs.String()
		// One line per try, and the tries are spaced by the backoff: 17 in the
		// first minute, not one per row or per signal.
		if got, want := strings.Count(out, "\n"), rec.calls(); got != want {
			t.Errorf("%d log lines for %d failed tries, want one for each:\n%s", got, want, out)
		}
		if got := rec.calls(); got != 17 {
			t.Errorf("flush ran %d times in a minute, want 17", got)
		}
		for _, want := range []string{`"level":"ERROR"`, `"queued":2`, `"retry_in"`, "connection refused"} {
			if !strings.Contains(out, want) {
				t.Errorf("log %s does not contain %s", out, want)
			}
		}

		// Closing with an expired context ends the retries and drops the rows.
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = b.Close(ctx)
		if got := b.Dropped(); got != 2 {
			t.Errorf("Dropped = %d, want 2", got)
		}
	})
}

func TestBatchOutageFillsTheQueueWithoutLosingRows(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var down atomic.Bool
		down.Store(true)
		rec := &recorder{fail: func(int) error {
			if down.Load() {
				return errors.New("connection refused")
			}
			return nil
		}}
		b := newBatcher(batcherOptions{BatchSize: 2, FlushInterval: 200 * time.Millisecond, QueueSize: 4,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2, 3)
		time.Sleep(time.Minute)
		synctest.Wait()
		if b.Full() {
			t.Fatal("full with 3 of 4 rows")
		}

		mustAdd(t, b, 4)
		if !b.Full() {
			t.Fatal("not full with 4 of 4 rows queued")
		}
		time.Sleep(10 * time.Minute)
		synctest.Wait()
		if !b.Full() {
			t.Fatal("no longer full after a long outage: rows were lost")
		}
		if got := b.Dropped(); got != 0 {
			t.Errorf("Dropped = %d, want 0", got)
		}
		if err := b.TryAdd(5); !errors.Is(err, errQueueFull) {
			t.Errorf("TryAdd on a full queue = %v, want errQueueFull", err)
		}
		assertBatches(t, "the outage", rec)

		down.Store(false)
		time.Sleep(10 * time.Second)
		synctest.Wait()
		assertBatches(t, "the recovery", rec, []int{1, 2}, []int{3, 4})
		if b.Full() {
			t.Error("still full after everything was written")
		}
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

func TestBatchFlushReturnsTheFailureAndKeepsTheRows(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		boom := errors.New("server closed the connection")
		var down atomic.Bool
		down.Store(true)
		rec := &recorder{fail: func(int) error {
			if down.Load() {
				return boom
			}
			return nil
		}}
		b := newBatcher(batcherOptions{BatchSize: 500, FlushInterval: time.Hour, QueueSize: 100,
			Logger: slog.New(slog.DiscardHandler)}, rec.flush)
		defer func() { _ = b.Close(context.Background()) }()

		mustAdd(t, b, 1, 2)
		if err := b.Flush(context.Background()); !errors.Is(err, boom) {
			t.Fatalf("Flush = %v, want the flush error", err)
		}
		if got := b.Dropped(); got != 0 {
			t.Errorf("Dropped = %d, want 0: a failed Flush drops nothing", got)
		}
		if got := queuedRows(b); got != 2 {
			t.Errorf("%d rows are queued after the failed Flush, want its 2 rows", got)
		}

		// The database is back: the worker writes the rows, and a Flush after that
		// has nothing left to write twice.
		down.Store(false)
		time.Sleep(time.Minute)
		synctest.Wait()
		assertBatches(t, "the recovery", rec, []int{1, 2})
		if err := b.Flush(context.Background()); err != nil {
			t.Fatalf("Flush after the recovery: %v", err)
		}
		assertBatches(t, "the Flush after the recovery", rec, []int{1, 2})
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
