package reliability

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Errors returned by Queues.Acquire. Callers match them with errors.Is.
var (
	// ErrQueueFull means every slot is busy and the line is at its limit.
	ErrQueueFull = errors.New("queue full")
	// ErrQueueTimeout means the request waited the model's full timeout
	// without getting a slot.
	ErrQueueTimeout = errors.New("queue timeout")
)

// QueueLimit bounds the requests one model serves at a time.
type QueueLimit struct {
	// MaxConcurrency requests run at once; it must be at least 1.
	MaxConcurrency int
	// QueueSize more requests may wait for a slot; 0 means none may.
	QueueSize int
	// Timeout is the longest a request waits for a slot.
	Timeout time.Duration
}

// Queues admits requests to models: each model has MaxConcurrency slots and a
// bounded first-in, first-out line for requests that find them all busy. A full
// line refuses at once instead of letting latency grow without bound. It is
// safe for concurrent use.
type Queues struct {
	byModel map[string]*modelQueue // fixed at construction; each guards itself
}

// NewQueues returns a Queues with one queue per limits entry, keyed by model.
func NewQueues(limits map[string]QueueLimit) *Queues {
	q := &Queues{byModel: make(map[string]*modelQueue, len(limits))}
	for model, limit := range limits {
		q.byModel[model] = &modelQueue{limit: limit, waiting: list.New()}
	}
	return q
}

// Acquire takes a slot of model. It returns at once when one is free. Else it
// joins the line if there is room, and waits for a slot, for at most the
// model's Timeout and until ctx ends; with no room it returns ErrQueueFull. A
// timeout returns ErrQueueTimeout and a ctx that ends returns its error.
//
// The caller must call release exactly when the request is done; calling it
// again does nothing. A model without a configured limit is an error.
func (q *Queues) Acquire(ctx context.Context, model string) (release func(), err error) {
	mq, ok := q.byModel[model]
	if !ok {
		return nil, fmt.Errorf("no queue configured for model %q", model)
	}
	return mq.acquire(ctx)
}

// Full reports whether a request for model would be refused right now: every
// slot is busy and the line is at its limit. The routing engine treats a full
// model as ineligible. An unconfigured model is never full.
func (q *Queues) Full(model string) bool {
	mq, ok := q.byModel[model]
	if !ok {
		return false
	}
	mq.mu.Lock()
	defer mq.mu.Unlock()
	return mq.running >= mq.limit.MaxConcurrency && mq.waiting.Len() >= mq.limit.QueueSize
}

// Depth returns how many requests are waiting for a slot of model.
func (q *Queues) Depth(model string) int {
	mq, ok := q.byModel[model]
	if !ok {
		return 0
	}
	mq.mu.Lock()
	defer mq.mu.Unlock()
	return mq.waiting.Len()
}

// modelQueue is the slots and the line of one model.
type modelQueue struct {
	limit QueueLimit

	mu      sync.Mutex
	running int        // slots in use
	waiting *list.List // of *waiter, oldest first
}

// waiter is a request in line. ready is closed, under the queue's lock, when
// the request is handed a slot.
type waiter struct {
	ready chan struct{}
	elem  *list.Element
}

func (m *modelQueue) acquire(ctx context.Context) (func(), error) {
	// A request that is already over must not take a slot.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.Lock()
	if m.running < m.limit.MaxConcurrency {
		m.running++
		m.mu.Unlock()
		return m.releaser(), nil
	}
	if m.waiting.Len() >= m.limit.QueueSize {
		m.mu.Unlock()
		return nil, ErrQueueFull
	}
	w := &waiter{ready: make(chan struct{})}
	w.elem = m.waiting.PushBack(w)
	m.mu.Unlock()

	timer := time.NewTimer(m.limit.Timeout)
	defer timer.Stop()
	var cause error
	select {
	case <-w.ready:
		return m.releaser(), nil
	case <-timer.C:
		cause = ErrQueueTimeout
	case <-ctx.Done():
		cause = ctx.Err()
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	select {
	case <-w.ready:
		// A release handed over the slot just as the wait ended. The slot is
		// ours and a release only frees it, so take it rather than pass it
		// on; the caller finds out the request is over when it uses it.
		return m.releaser(), nil
	default:
	}
	m.waiting.Remove(w.elem)
	return nil, cause
}

// releaser returns the function that gives back one slot, once.
func (m *modelQueue) releaser() func() {
	var once sync.Once
	return func() { once.Do(m.release) }
}

// release hands the slot to the oldest waiter, or frees it when none waits.
func (m *modelQueue) release() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if front := m.waiting.Front(); front != nil {
		close(m.waiting.Remove(front).(*waiter).ready)
		return // the slot stays in use, now by the waiter
	}
	m.running--
}
