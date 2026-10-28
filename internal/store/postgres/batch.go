package postgres

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// Defaults of the batching writer (spec 12).
const (
	defaultBatchSize     = 500
	defaultFlushInterval = 200 * time.Millisecond
	defaultQueueSize     = 10000

	// flushRetries is how many times a failed flush is tried again. The waits
	// before the retries are retryBase, twice that, and four times that.
	flushRetries = 3
	retryBase    = 100 * time.Millisecond
	// flushTimeout bounds one try to write a batch.
	flushTimeout = 10 * time.Second
)

var (
	errBatcherClosed = errors.New("postgres writer is closed")
	errQueueFull     = errors.New("postgres write queue is full")
)

// batcherOptions configures a batcher. A zero field takes its default.
type batcherOptions struct {
	// BatchSize is how many pending rows make a batch: it is written at once.
	BatchSize int
	// FlushInterval is how long the first pending row waits for company before
	// whatever is pending is written anyway.
	FlushInterval time.Duration
	// QueueSize is how many rows may be pending or being written before Full
	// reports true.
	QueueSize int
	Logger    *slog.Logger
}

// batcher collects rows and writes them in batches through flush, in the
// order they were added, one batch at a time. A batch is written when
// BatchSize rows are pending or FlushInterval after the first pending row,
// whichever comes first. It is generic over the row type and the flush
// function so that it can be tested without a database. It is safe for
// concurrent use.
//
// A batch that cannot be written is tried again flushRetries times with a
// doubling wait; after that its rows are dropped, logged and counted, because
// holding them back would only fill the queue behind an outage.
type batcher[T any] struct {
	flush    func(ctx context.Context, rows []T) error
	size     int
	interval time.Duration
	capacity int
	log      *slog.Logger

	// base is the context of the batches the worker writes; abort cancels it.
	base  context.Context
	abort context.CancelFunc

	mu       sync.Mutex
	pending  []T
	inflight int // rows taken for a batch that is not finished
	closed   bool
	timer    *time.Timer // runs when the first pending row has waited too long

	flushMu sync.Mutex    // held while a batch is written, so batches stay in order
	wake    chan struct{} // tells the worker to look at the queue; capacity 1
	due     atomic.Bool   // the timer ran: write whatever is pending
	stop    chan struct{}
	done    chan struct{}
	dropped atomic.Int64
}

// newBatcher starts a batcher whose batches are written by flush. Close it to
// stop its worker.
func newBatcher[T any](o batcherOptions, flush func(ctx context.Context, rows []T) error) *batcher[T] {
	b := &batcher[T]{
		flush:    flush,
		size:     o.BatchSize,
		interval: o.FlushInterval,
		capacity: o.QueueSize,
		log:      o.Logger,
		wake:     make(chan struct{}, 1),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	if b.size < 1 {
		b.size = defaultBatchSize
	}
	if b.interval <= 0 {
		b.interval = defaultFlushInterval
	}
	if b.capacity < 1 {
		b.capacity = defaultQueueSize
	}
	if b.log == nil {
		b.log = slog.New(slog.DiscardHandler)
	}
	b.base, b.abort = context.WithCancel(context.Background())
	go b.run()
	return b
}

// Add queues a row. It never refuses a row because the queue is full: a bill
// must not be lost when the request that owns it was admitted before the queue
// filled up. It fails only after Close.
func (b *batcher[T]) Add(row T) error { return b.add(row, false) }

// TryAdd queues a row unless the queue is full (errQueueFull) or the batcher is
// closed. It is for rows that can be lost better than a bill can.
func (b *batcher[T]) TryAdd(row T) error { return b.add(row, true) }

func (b *batcher[T]) add(row T, bounded bool) error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return errBatcherClosed
	}
	if bounded && len(b.pending)+b.inflight >= b.capacity {
		b.mu.Unlock()
		return errQueueFull
	}
	b.pending = append(b.pending, row)
	n := len(b.pending)
	if n == 1 {
		b.arm()
	}
	b.mu.Unlock()
	if n >= b.size {
		b.signal()
	}
	return nil
}

// Full reports that as many rows are pending or being written as the queue
// holds. Rows count until they are written or dropped.
func (b *batcher[T]) Full() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pending)+b.inflight >= b.capacity
}

// Dropped is how many rows were given up after their batch kept failing.
func (b *batcher[T]) Dropped() int64 { return b.dropped.Load() }

// Flush writes every row pending now, one batch at a time, and returns the
// error of the first batch that failed for good; the rows after it stay
// pending.
func (b *batcher[T]) Flush(ctx context.Context) error {
	return b.flushPending(ctx, true)
}

// Close stops accepting rows, waits for the batch being written, then writes
// what is left. When ctx ends first it stops waiting and cancels the writes in
// progress; the rows not written are dropped and counted, and the error says
// so. Close is safe to call more than once.
func (b *batcher[T]) Close(ctx context.Context) error {
	b.mu.Lock()
	first := !b.closed
	b.closed = true
	if b.timer != nil {
		b.timer.Stop()
	}
	b.mu.Unlock()
	if !first {
		return nil
	}

	close(b.stop)
	var errs []error
	select {
	case <-b.done:
	case <-ctx.Done():
		b.abort()
		errs = append(errs, ctx.Err())
		<-b.done
	}
	errs = append(errs, b.flushPending(ctx, true))
	b.abort()
	return errors.Join(errs...)
}

// run is the worker: it writes batches when it is woken.
func (b *batcher[T]) run() {
	defer close(b.done)
	for {
		select {
		case <-b.stop:
			return
		case <-b.wake:
			// A failed batch has been logged and counted; the next wake takes up
			// the rows behind it.
			_ = b.flushPending(b.base, b.due.Swap(false))
		}
	}
}

// signal wakes the worker unless a wake is already waiting.
func (b *batcher[T]) signal() {
	select {
	case b.wake <- struct{}{}:
	default:
	}
}

// arm starts the wait of the first pending row. The caller holds b.mu.
func (b *batcher[T]) arm() {
	if b.timer == nil {
		b.timer = time.AfterFunc(b.interval, b.onTimer)
		return
	}
	b.timer.Reset(b.interval)
}

func (b *batcher[T]) onTimer() {
	b.due.Store(true)
	b.signal()
}

// flushPending writes pending rows in batches under flushMu. With all it writes
// every pending row; without, it writes only while a full batch is pending and
// leaves the rest to the timer.
func (b *batcher[T]) flushPending(ctx context.Context, all bool) error {
	b.flushMu.Lock()
	defer b.flushMu.Unlock()
	for {
		rows := b.take(all)
		if len(rows) == 0 {
			return nil
		}
		err := b.write(ctx, rows)
		b.mu.Lock()
		b.inflight -= len(rows)
		b.mu.Unlock()
		if err != nil {
			return err
		}
	}
}

// take removes up to one batch from the queue and counts it as in flight. It
// returns nil when there is nothing to write: nothing is pending, or, without
// all, less than a full batch is.
func (b *batcher[T]) take(all bool) []T {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(b.pending)
	if n == 0 || (!all && n < b.size) {
		return nil
	}
	n = min(n, b.size)
	rows := b.pending[:n:n]
	if n == len(b.pending) {
		b.pending = nil
	} else {
		b.pending = b.pending[n:]
	}
	b.inflight += n

	if b.timer != nil {
		b.timer.Stop()
	}
	if len(b.pending) > 0 {
		b.arm() // the rows left behind wait for their own interval
	}
	return rows
}

// write calls flush for rows, trying again after a failure up to flushRetries
// times. When the last try fails, or ctx ends, the rows are dropped.
func (b *batcher[T]) write(ctx context.Context, rows []T) error {
	var err error
	tries := 0
	for {
		tries++
		if err = b.try(ctx, rows); err == nil {
			return nil
		}
		if tries > flushRetries || ctx.Err() != nil || !wait(ctx, retryBase<<(tries-1)) {
			break
		}
	}
	b.dropped.Add(int64(len(rows)))
	b.log.Error("postgres writer dropped rows after repeated failures",
		"rows", len(rows), "tries", tries, "error", err)
	return fmt.Errorf("write %d rows: %w", len(rows), err)
}

func (b *batcher[T]) try(ctx context.Context, rows []T) error {
	ctx, cancel := context.WithTimeout(ctx, flushTimeout)
	defer cancel()
	return b.flush(ctx, rows)
}

// wait sleeps for d and reports whether it did: false means ctx ended first.
func wait(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
