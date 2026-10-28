package postgres

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// Defaults of the batching writer (spec 12).
const (
	defaultBatchSize     = 500
	defaultFlushInterval = 200 * time.Millisecond
	defaultQueueSize     = 10000

	// retryBase is the wait before a failed batch is tried again. Each further
	// failure doubles it, up to retryMax; a successful write starts over at
	// retryBase.
	retryBase = 100 * time.Millisecond
	retryMax  = 5 * time.Second
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
// Rows are never dropped while the batcher is open. A batch that cannot be
// written goes back to the head of the queue, ahead of the rows added after it,
// and the worker tries it again after a wait that starts at retryBase, doubles
// with every failure up to retryMax and starts over after a success. A long
// outage therefore fills the queue until Full reports true, and the caller
// stops admitting work; it does not lose the rows already queued. Only Close
// drops rows: those still queued when its context ends.
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
	failures int // writes that failed since the last one that succeeded
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
// holds. Rows count until they are written, however many times their batch
// failed, or until Close drops them.
func (b *batcher[T]) Full() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pending)+b.inflight >= b.capacity
}

// Dropped is how many rows Close gave up because its context ended before they
// were written. It is 0 while the batcher is open.
func (b *batcher[T]) Dropped() int64 { return b.dropped.Load() }

// Flush writes every row pending now, one batch at a time, and returns the
// error of the first batch that failed. Nothing is dropped: that batch and the
// rows after it stay queued, in order, and the worker tries them again.
func (b *batcher[T]) Flush(ctx context.Context) error {
	return b.flushPending(ctx, true)
}

// Close stops accepting rows, waits for the batch being written, then writes
// what is left, trying again after a failure as the worker does, until ctx
// ends. The rows still queued then are dropped, counted and logged, and the
// error says so. When ctx ends while a batch is being written, that write is
// cancelled. Close is safe to call more than once.
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
	select {
	case <-b.done:
	case <-ctx.Done():
		b.abort()
		<-b.done
	}
	lastErr := b.drain(ctx)
	b.abort()

	n := b.discard()
	if n == 0 {
		return nil
	}
	b.log.Error("postgres writer dropped the rows still queued on close", "rows", n)
	return fmt.Errorf("%d queued rows were not written: %w", n, errors.Join(ctx.Err(), lastErr))
}

// run is the worker: it writes batches when it is woken.
func (b *batcher[T]) run() {
	defer close(b.done)
	for {
		select {
		case <-b.stop:
			return
		case <-b.wake:
			if !b.work() {
				return
			}
		}
	}
}

// work writes the batches that are due. After a failure it waits out the
// backoff and tries again, writing whatever is pending, until the queue is
// written. It reports false when the batcher was stopped meanwhile.
func (b *batcher[T]) work() bool {
	all := b.due.Swap(false)
	for {
		if d := b.delay(); d > 0 {
			// A batch failed and went back to the queue: wait before the retry, and
			// then write all of it, however few rows there are.
			if !wait(b.stop, d) {
				return false
			}
			all = true
		}
		err := b.flushPending(b.base, all)
		if err == nil {
			return true
		}
		if b.base.Err() != nil {
			return false // Close cancelled the write; it takes over the queue
		}
		b.log.Error("postgres writer could not write a batch; it stays queued and is tried again",
			"queued", b.queued(), "retry_in", b.delay().String(), "error", err)
	}
}

// drain writes the queue for Close, waiting out the backoff between tries,
// until it is empty or ctx ends. It returns the error of the last failed try.
func (b *batcher[T]) drain(ctx context.Context) error {
	var err error
	for ctx.Err() == nil {
		if err = b.flushPending(ctx, true); err == nil {
			return nil
		}
		if !wait(ctx.Done(), b.delay()) {
			break
		}
	}
	return err
}

// discard drops every queued row and returns how many there were.
func (b *batcher[T]) discard() int {
	b.flushMu.Lock() // no write is in progress after this
	defer b.flushMu.Unlock()
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(b.pending)
	b.pending = nil
	b.dropped.Add(int64(n))
	return n
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

// queued is how many rows are pending or being written.
func (b *batcher[T]) queued() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pending) + b.inflight
}

// delay is how long to wait before the next try after the writes that failed
// since the last success: 0 with none, then retryBase, doubling up to retryMax.
func (b *batcher[T]) delay() time.Duration {
	b.mu.Lock()
	n := b.failures
	b.mu.Unlock()
	if n == 0 {
		return 0
	}
	d := retryBase
	for i := 1; i < n && d < retryMax; i++ {
		d *= 2
	}
	return min(d, retryMax)
}

// flushPending writes pending rows in batches under flushMu. With all it writes
// every pending row; without, it writes only while a full batch is pending and
// leaves the rest to the timer. It stops at the first batch that fails and
// returns its error; that batch is back at the head of the queue.
func (b *batcher[T]) flushPending(ctx context.Context, all bool) error {
	b.flushMu.Lock()
	defer b.flushMu.Unlock()
	for {
		rows := b.take(all)
		if len(rows) == 0 {
			return nil
		}
		err := b.try(ctx, rows)
		b.settle(rows, err)
		if err != nil {
			return fmt.Errorf("write %d rows: %w", len(rows), err)
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

// settle ends the time rows spent in flight. After a failed write they go back
// to the head of the queue, ahead of the rows added meanwhile, and the worker
// is woken to try again after the backoff. The rows count as queued throughout.
func (b *batcher[T]) settle(rows []T, err error) {
	b.mu.Lock()
	b.inflight -= len(rows)
	if err == nil {
		b.failures = 0
		b.mu.Unlock()
		return
	}
	b.failures++
	b.pending = slices.Concat(rows, b.pending)
	b.mu.Unlock()
	b.signal()
}

func (b *batcher[T]) try(ctx context.Context, rows []T) error {
	ctx, cancel := context.WithTimeout(ctx, flushTimeout)
	defer cancel()
	return b.flush(ctx, rows)
}

// wait sleeps for d and reports whether it did: false means done closed first.
func wait(done <-chan struct{}, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-done:
		return false
	}
}
