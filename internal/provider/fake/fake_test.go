package fake_test

import (
	"context"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/provider/fake"
)

var ctx = context.Background()

func req() provider.Request {
	return provider.Request{Model: "m", MaxTokens: 100, Messages: []provider.Message{{Role: "user", Content: "hello world!"}}}
}

func baseCfg() config.Fake {
	return config.Fake{Seed: 7, OutputTokens: 5, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1}
}

// drain reads a stream until it ends, returning the events and the final error.
func drain(t *testing.T, s provider.Stream) ([]provider.StreamEvent, error) {
	t.Helper()
	var evs []provider.StreamEvent
	for {
		ev, err := s.Recv()
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
}

func TestCompleteDeterministic(t *testing.T) {
	a, err := fake.New("f", baseCfg()).Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	b, err := fake.New("f", baseCfg()).Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	if a.Content != b.Content {
		t.Fatalf("same seed gave different content: %q vs %q", a.Content, b.Content)
	}
	words := strings.Split(a.Content, " ")
	if len(words) != 5 {
		t.Fatalf("got %d words, want 5: %q", len(words), a.Content)
	}
	for _, w := range words {
		if !slices.Contains(fake.Vocabulary[:], w) {
			t.Fatalf("word %q not in vocabulary", w)
		}
	}
	if a.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", a.FinishReason)
	}
	if a.Usage == nil || *a.Usage != (provider.Usage{PromptTokens: 3, CompletionTokens: 5}) {
		t.Fatalf("Usage = %+v, want {3 5}", a.Usage)
	}

	other := baseCfg()
	other.Seed = 8
	c, err := fake.New("f", other).Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	if c.Content == a.Content {
		t.Fatalf("seed 8 gave the same content as seed 7: %q", c.Content)
	}
}

func TestStreamMatchesComplete(t *testing.T) {
	p := fake.New("f", baseCfg())
	full, err := p.Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	s, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, err := drain(t, s)
	if err != io.EOF {
		t.Fatalf("stream ended with %v, want io.EOF", err)
	}
	if len(evs) != 6 {
		t.Fatalf("got %d events, want 5 content + 1 final", len(evs))
	}
	var sb strings.Builder
	for i, ev := range evs[:5] {
		if ev.Delta == "" || ev.FinishReason != "" || ev.Usage != nil {
			t.Fatalf("event %d is not a pure content event: %+v", i, ev)
		}
		if i > 0 && !strings.HasPrefix(ev.Delta, " ") {
			t.Fatalf("event %d delta %q lacks leading space", i, ev.Delta)
		}
		sb.WriteString(ev.Delta)
	}
	if sb.String() != full.Content {
		t.Fatalf("stream content %q != complete content %q", sb.String(), full.Content)
	}
	last := evs[5]
	if last.Delta != "" || last.FinishReason != "stop" || last.Usage == nil || *last.Usage != (provider.Usage{PromptTokens: 3, CompletionTokens: 5}) {
		t.Fatalf("final event = %+v", last)
	}
}

func TestMaxTokensCapsOutput(t *testing.T) {
	r := req()
	r.MaxTokens = 2
	resp, err := fake.New("f", baseCfg()).Complete(ctx, r)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(strings.Split(resp.Content, " ")); n != 2 {
		t.Fatalf("got %d words, want 2", n)
	}
	if resp.FinishReason != "length" {
		t.Fatalf("FinishReason = %q, want length", resp.FinishReason)
	}
	if resp.Usage == nil || resp.Usage.CompletionTokens != 2 {
		t.Fatalf("Usage = %+v, want CompletionTokens 2", resp.Usage)
	}
}

func TestInjectedError(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, ErrorRate: 1, ErrorKind: provider.KindRateLimited, RetryAfter: 2 * time.Second, DisconnectAfter: 1})
	check := func(err error) {
		t.Helper()
		var pe *provider.Error
		if !errors.As(err, &pe) {
			t.Fatalf("err = %v, want *provider.Error", err)
		}
		if pe.Kind != provider.KindRateLimited || pe.Provider != "f" || pe.RetryAfter != 2*time.Second {
			t.Fatalf("got %+v", pe)
		}
		if !strings.Contains(pe.Message, "injected rate_limited error") {
			t.Fatalf("message = %q", pe.Message)
		}
	}
	_, err := p.Complete(ctx, req())
	check(err)
	_, err = p.Stream(ctx, req())
	check(err)
	if got := p.Calls(); got != 2 {
		t.Fatalf("Calls() = %d, want 2", got)
	}
}

func TestInjectedErrorOmitsRetryAfterForOtherKinds(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, ErrorRate: 1, ErrorKind: provider.KindUnavailable, RetryAfter: 2 * time.Second, DisconnectAfter: 1})
	_, err := p.Complete(ctx, req())
	var pe *provider.Error
	if !errors.As(err, &pe) {
		t.Fatalf("err = %v, want *provider.Error", err)
	}
	if pe.Kind != provider.KindUnavailable || pe.RetryAfter != 0 {
		t.Fatalf("got %+v", pe)
	}
}

func TestMidStreamDisconnect(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 5, DisconnectRate: 1, DisconnectAfter: 2, ErrorKind: provider.KindUnavailable})
	s, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, err := drain(t, s)
	if len(evs) != 2 {
		t.Fatalf("got %d content events, want 2", len(evs))
	}
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Kind != provider.KindUnavailable || pe.Provider != "f" {
		t.Fatalf("err = %v, want unavailable *provider.Error", err)
	}
	if !strings.Contains(pe.Message, "mid-stream disconnect") {
		t.Fatalf("message = %q", pe.Message)
	}
}

func TestMidStreamDisconnectWithFewerWords(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 2, DisconnectRate: 1, DisconnectAfter: 10, ErrorKind: provider.KindUnavailable})
	s, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, err := drain(t, s)
	if len(evs) != 2 {
		t.Fatalf("got %d content events, want all 2 words", len(evs))
	}
	var pe *provider.Error
	if !errors.As(err, &pe) || !strings.Contains(pe.Message, "mid-stream disconnect") {
		t.Fatalf("err = %v, want mid-stream disconnect", err)
	}
}

func TestOmitUsage(t *testing.T) {
	cfg := baseCfg()
	cfg.OmitUsageRate = 1
	p := fake.New("f", cfg)
	resp, err := p.Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage != nil {
		t.Fatalf("Complete Usage = %+v, want nil", resp.Usage)
	}
	s, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, err := drain(t, s)
	if err != io.EOF {
		t.Fatalf("stream ended with %v, want io.EOF", err)
	}
	last := evs[len(evs)-1]
	if last.FinishReason != "stop" || last.Usage != nil {
		t.Fatalf("final event = %+v, want stop with nil usage", last)
	}
}

func TestFakeOverReport(t *testing.T) {
	clean, err := fake.New("f", baseCfg()).Complete(ctx, req())
	if err != nil {
		t.Fatal(err)
	}

	cfg := baseCfg()
	cfg.OverReportTokens = 50
	p := fake.New("f", cfg)

	t.Run("complete", func(t *testing.T) {
		resp, err := p.Complete(ctx, req())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Content != clean.Content {
			t.Fatalf("Content = %q, want the delivered text unchanged: %q", resp.Content, clean.Content)
		}
		if resp.Usage == nil || *resp.Usage != (provider.Usage{PromptTokens: 3, CompletionTokens: 55}) {
			t.Fatalf("Usage = %+v, want {3 55}: the 5 delivered tokens plus 50", resp.Usage)
		}
	})

	t.Run("stream", func(t *testing.T) {
		s, err := p.Stream(ctx, req())
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		evs, err := drain(t, s)
		if err != io.EOF {
			t.Fatalf("stream ended with %v, want io.EOF", err)
		}
		var sb strings.Builder
		for _, ev := range evs[:len(evs)-1] {
			sb.WriteString(ev.Delta)
		}
		if sb.String() != clean.Content {
			t.Fatalf("streamed %q, want the delivered text unchanged: %q", sb.String(), clean.Content)
		}
		last := evs[len(evs)-1]
		if last.Usage == nil || *last.Usage != (provider.Usage{PromptTokens: 3, CompletionTokens: 55}) {
			t.Fatalf("final Usage = %+v, want {3 55}", last.Usage)
		}
	})

	t.Run("on top of the max_tokens cap", func(t *testing.T) {
		r := req()
		r.MaxTokens = 2
		resp, err := p.Complete(ctx, r)
		if err != nil {
			t.Fatal(err)
		}
		if n := len(strings.Split(resp.Content, " ")); n != 2 {
			t.Fatalf("got %d words, want 2", n)
		}
		if resp.Usage == nil || resp.Usage.CompletionTokens != 52 {
			t.Fatalf("Usage = %+v, want CompletionTokens 52", resp.Usage)
		}
	})

	t.Run("a response without usage stays without", func(t *testing.T) {
		silent := cfg
		silent.OmitUsageRate = 1
		resp, err := fake.New("f", silent).Complete(ctx, req())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage != nil {
			t.Fatalf("Usage = %+v, want nil: there is nothing to over-report", resp.Usage)
		}
	})
}

func TestFaultSequenceIsSeeded(t *testing.T) {
	pattern := func() []bool {
		p := fake.New("f", config.Fake{Seed: 3, OutputTokens: 2, ErrorRate: 0.5, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})
		out := make([]bool, 20)
		for i := range out {
			_, err := p.Complete(ctx, req())
			out[i] = err != nil
		}
		return out
	}
	a, b := pattern(), pattern()
	if !slices.Equal(a, b) {
		t.Fatalf("fault pattern differs between fresh providers:\n%v\n%v", a, b)
	}
	if !slices.Contains(a, true) {
		t.Fatalf("no failure in %v: ErrorRate ignored", a)
	}
	if !slices.Contains(a, false) {
		t.Fatalf("no success in %v: ErrorRate ignored", a)
	}
}

func TestLatencyAndCancel(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, TTFT: 5 * time.Second, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})

	cctx, cancel := context.WithCancel(ctx)
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	_, err := p.Complete(cctx, req())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if d := time.Since(start); d >= time.Second {
		t.Fatalf("cancel took %v", d)
	}

	tctx, tcancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer tcancel()
	_, err = p.Complete(tctx, req())
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Kind != provider.KindTimeout {
		t.Fatalf("err = %v, want timeout *provider.Error", err)
	}
}

func TestStreamCancelDuringTTFT(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, TTFT: 5 * time.Second, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})
	cctx, cancel := context.WithCancel(ctx)
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	_, err := p.Stream(cctx, req())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if d := time.Since(start); d >= time.Second {
		t.Fatalf("cancel took %v", d)
	}
}

func TestStreamRecvHonorsCancelAndClose(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, PerToken: 5 * time.Second, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})

	cctx, cancel := context.WithCancel(ctx)
	s, err := p.Stream(cctx, req())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Recv(); err != nil {
		t.Fatalf("first Recv must return immediately: %v", err)
	}
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv err = %v, want context.Canceled", err)
	}
	if d := time.Since(start); d >= time.Second {
		t.Fatalf("cancel took %v", d)
	}

	s2, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Recv(); err != nil {
		t.Fatal(err)
	}
	time.AfterFunc(50*time.Millisecond, func() { _ = s2.Close() })
	start = time.Now()
	if _, err := s2.Recv(); err == nil {
		t.Fatal("Recv after Close returned nil error")
	}
	if d := time.Since(start); d >= time.Second {
		t.Fatalf("close took %v", d)
	}
	if err := s2.Close(); err != nil {
		t.Fatalf("second Close = %v", err)
	}
}

func TestFakeRecvHonorsCanceledContextOnFirstEvent(t *testing.T) {
	p := fake.New("f", baseCfg())
	cctx, cancel := context.WithCancel(ctx)
	s, err := p.Stream(cctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	cancel()
	// PerToken is zero, so no sleep would notice the cancel: Recv must check.
	if ev, err := s.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("first Recv = %+v, %v; want context.Canceled", ev, err)
	}
	if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv after the failure = %v, want the same sticky error", err)
	}
}

func TestFakeRecvHonorsCanceledContextBeforeFinal(t *testing.T) {
	cfg := baseCfg()
	cfg.OutputTokens = 2
	p := fake.New("f", cfg)
	cctx, cancel := context.WithCancel(ctx)
	s, err := p.Stream(cctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for i := range 2 {
		if _, err := s.Recv(); err != nil {
			t.Fatalf("Recv %d: %v", i, err)
		}
	}
	cancel()
	if ev, err := s.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv for the final event = %+v, %v; want context.Canceled", ev, err)
	}
}

func TestFakeDisconnectPathChecksContext(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 5, DisconnectRate: 1, DisconnectAfter: 1, ErrorKind: provider.KindUnavailable})
	cctx, cancel := context.WithCancel(ctx)
	s, err := p.Stream(cctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.Recv(); err != nil {
		t.Fatalf("Recv before the disconnect: %v", err)
	}
	cancel()
	// The disconnect is due now; a canceled caller sees its own cancellation,
	// not the injected upstream failure.
	_, err = s.Recv()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv at the disconnect = %v, want context.Canceled", err)
	}
	var pe *provider.Error
	if errors.As(err, &pe) {
		t.Fatalf("err = %+v, want the cancellation, not a provider error", pe)
	}
}

func TestFakeRecvHonorsExpiredDeadline(t *testing.T) {
	p := fake.New("f", baseCfg())
	dctx, cancel := context.WithTimeout(ctx, 20*time.Millisecond)
	defer cancel()
	s, err := p.Stream(dctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	<-dctx.Done()
	_, err = s.Recv()
	var pe *provider.Error
	if !errors.As(err, &pe) || pe.Kind != provider.KindTimeout || pe.Provider != "f" {
		t.Fatalf("Recv = %v, want a timeout *provider.Error from f", err)
	}
}

func TestTTFTApplied(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, TTFT: 30 * time.Millisecond, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})
	start := time.Now()
	s, err := p.Stream(ctx, req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if d := time.Since(start); d < 30*time.Millisecond {
		t.Fatalf("Stream returned after %v, want >= 30ms", d)
	}
}

func TestConcurrentCallsAreRaceFree(t *testing.T) {
	p := fake.New("f", config.Fake{Seed: 1, OutputTokens: 3, ErrorRate: 0.3, ErrorKind: provider.KindUnavailable, DisconnectAfter: 1})
	var wg sync.WaitGroup
	for range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = p.Complete(ctx, req())
		}()
	}
	wg.Wait()
	if got := p.Calls(); got != 50 {
		t.Fatalf("Calls() = %d, want 50", got)
	}
}
