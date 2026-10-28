// Package fake is a deterministic, fault-injecting provider used by tests and
// load runs. Output and fault sequences depend only on the configured seed,
// the request content and the order of calls.
package fake

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"math/rand/v2"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider"
)

// Vocabulary is the word list generated output is drawn from.
var Vocabulary = [16]string{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
	"india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"}

// Provider is the fake provider. It is safe for concurrent use.
type Provider struct {
	name  string
	cfg   config.Fake
	calls atomic.Int64
}

var _ provider.Provider = (*Provider)(nil)

// New returns a fake provider. Defaults must already be applied to c.
func New(name string, c config.Fake) *Provider {
	return &Provider{name: name, cfg: c}
}

// Calls reports how many attempts have started, Complete and Stream combined.
func (p *Provider) Calls() int64 { return p.calls.Load() }

// call holds the per-attempt state derived at the start of a call.
type call struct {
	words      []string
	finish     string
	usage      *provider.Usage // nil when the omit-usage fault fired
	errFault   bool
	disconnect bool
	rng        *rand.Rand // fault RNG, continues with one draw per sleep
}

func (p *Provider) begin(req provider.Request) *call {
	idx := uint64(p.calls.Add(1) - 1)
	rng := rand.New(rand.NewPCG(uint64(p.cfg.Seed), idx))
	// Fixed draw order: error, disconnect, omit-usage, then jitter per sleep.
	errFault := rng.Float64() < p.cfg.ErrorRate
	disconnect := rng.Float64() < p.cfg.DisconnectRate
	omit := rng.Float64() < p.cfg.OmitUsageRate

	n, finish := p.cfg.OutputTokens, "stop"
	if req.MaxTokens > 0 && req.MaxTokens < n {
		n, finish = req.MaxTokens, "length"
	}
	if n < 0 {
		n = 0
	}

	c := &call{
		words:      words(p.cfg.Seed, req, n),
		finish:     finish,
		errFault:   errFault,
		disconnect: disconnect,
		rng:        rng,
	}
	if !omit {
		c.usage = &provider.Usage{PromptTokens: promptTokens(req), CompletionTokens: n}
	}
	return c
}

func words(seed int64, req provider.Request, n int) []string {
	h := fnv.New64a()
	_, _ = h.Write([]byte(req.Model))
	for _, m := range req.Messages {
		_, _ = h.Write([]byte(m.Role + "\x00" + m.Content + "\x00"))
	}
	rng := rand.New(rand.NewPCG(uint64(seed), h.Sum64()))
	out := make([]string, n)
	for i := range out {
		out[i] = Vocabulary[rng.IntN(len(Vocabulary))]
	}
	return out
}

func promptTokens(req provider.Request) int {
	total := 0
	for _, m := range req.Messages {
		total += len(m.Content)
	}
	return (total + 3) / 4
}

// sleep waits d scaled by jitter. It returns early with a transport error when
// ctx ends, or with a closed error when done is closed (nil done never closes).
func (p *Provider) sleep(ctx context.Context, done <-chan struct{}, c *call, d time.Duration) error {
	u := c.rng.Float64()
	d = time.Duration(float64(d) * (1 + p.cfg.Jitter*(2*u-1)))
	if err := ctx.Err(); err != nil {
		return provider.TransportError(ctx, p.name, err)
	}
	select {
	case <-done:
		return errClosed(p.name)
	default:
	}
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return provider.TransportError(ctx, p.name, ctx.Err())
	case <-done:
		return errClosed(p.name)
	}
}

func errClosed(name string) error {
	return &provider.Error{Kind: provider.KindInternal, Provider: name, Message: "fake: stream closed"}
}

func (p *Provider) injectedError() error {
	e := &provider.Error{
		Kind:     p.cfg.ErrorKind,
		Provider: p.name,
		Message:  fmt.Sprintf("fake: injected %s error", p.cfg.ErrorKind),
	}
	if p.cfg.ErrorKind == provider.KindRateLimited {
		e.RetryAfter = p.cfg.RetryAfter
	}
	return e
}

// Complete returns the full generated response after simulated latency.
func (p *Provider) Complete(ctx context.Context, req provider.Request) (*provider.Response, error) {
	c := p.begin(req)
	if err := p.sleep(ctx, nil, c, p.cfg.TTFT); err != nil {
		return nil, err
	}
	if c.errFault {
		return nil, p.injectedError()
	}
	if err := p.sleep(ctx, nil, c, time.Duration(len(c.words))*p.cfg.PerToken); err != nil {
		return nil, err
	}
	return &provider.Response{
		Content:      strings.Join(c.words, " "),
		FinishReason: c.finish,
		Usage:        c.usage,
	}, nil
}

// Stream sleeps TTFT as connection setup, then returns a stream of word deltas.
func (p *Provider) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	c := p.begin(req)
	if err := p.sleep(ctx, nil, c, p.cfg.TTFT); err != nil {
		return nil, err
	}
	if c.errFault {
		return nil, p.injectedError()
	}
	after := p.cfg.DisconnectAfter
	if after > len(c.words) {
		after = len(c.words)
	}
	return &stream{p: p, ctx: ctx, c: c, disconnectAfter: after, done: make(chan struct{})}, nil
}

// stream is one in-flight streaming response. Recv must be called from a
// single goroutine; Close may be called concurrently.
type stream struct {
	p               *Provider
	ctx             context.Context
	c               *call
	disconnectAfter int
	next            int   // index of the next word to send
	err             error // sticky terminal state
	done            chan struct{}
	once            sync.Once
}

func (s *stream) Recv() (provider.StreamEvent, error) {
	if s.err != nil {
		return provider.StreamEvent{}, s.err
	}
	ev, err := s.recv()
	if err != nil {
		s.err = err
	}
	return ev, err
}

func (s *stream) recv() (provider.StreamEvent, error) {
	select {
	case <-s.done:
		return provider.StreamEvent{}, errClosed(s.p.name)
	default:
	}
	if s.c.disconnect && s.next >= s.disconnectAfter {
		// A canceled caller sees its own cancellation, not the injected failure.
		if err := s.canceled(); err != nil {
			return provider.StreamEvent{}, err
		}
		return provider.StreamEvent{}, &provider.Error{
			Kind:     provider.KindUnavailable,
			Provider: s.p.name,
			Message:  "fake: injected mid-stream disconnect",
		}
	}
	if s.next < len(s.c.words) {
		if s.next > 0 {
			if err := s.p.sleep(s.ctx, s.done, s.c, s.p.cfg.PerToken); err != nil {
				return provider.StreamEvent{}, err
			}
		} else if err := s.canceled(); err != nil {
			// The first event waits for nothing, so nothing else would notice.
			return provider.StreamEvent{}, err
		}
		delta := s.c.words[s.next]
		if s.next > 0 {
			delta = " " + delta
		}
		s.next++
		return provider.StreamEvent{Delta: delta}, nil
	}
	if err := s.canceled(); err != nil {
		return provider.StreamEvent{}, err
	}
	// The final event is followed by io.EOF through the sticky error.
	s.err = io.EOF
	return provider.StreamEvent{FinishReason: s.c.finish, Usage: s.c.usage}, nil
}

// canceled returns the transport error for a caller whose context has ended,
// or nil while it is live.
func (s *stream) canceled() error {
	if err := s.ctx.Err(); err != nil {
		return provider.TransportError(s.ctx, s.p.name, err)
	}
	return nil
}

// Close releases the stream and wakes a blocked Recv. It is idempotent.
func (s *stream) Close() error {
	s.once.Do(func() { close(s.done) })
	return nil
}
