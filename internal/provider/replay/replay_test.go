package replay

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/omorsi45/routeforge/internal/provider"
)

var bg = context.Background()

func digest(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// fixture is one recorded output. Prompt is the feature text of the request
// that must find it.
type fixture struct {
	Prompt           string
	Model            string
	Text             string
	PromptTokens     int
	CompletionTokens int
}

// jsonl renders fixtures in the trainer's export format, including the
// fields the provider has no use for.
func jsonl(t *testing.T, fixtures ...fixture) string {
	t.Helper()
	var b strings.Builder
	for i, f := range fixtures {
		line, err := json.Marshal(map[string]any{
			"prompt_sha256":     digest(f.Prompt),
			"model":             f.Model,
			"prompt_id":         fmt.Sprintf("ds:%d", i),
			"dataset":           "ds",
			"text":              f.Text,
			"prompt_tokens":     f.PromptTokens,
			"completion_tokens": f.CompletionTokens,
			"cost_usd":          0.5,
			"score":             1.0,
			"success":           true,
		})
		if err != nil {
			t.Fatal(err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	return b.String()
}

// profiles gives every test model the same 100 ms TTFT and 100 tokens/s, with
// no jitter so timings are exact.
var profiles = Profiles{
	"m-a": {TTFTMS: 100, DecodeTokensPerS: 100},
	"m-b": {TTFTMS: 100, DecodeTokensPerS: 100},
}

func newProvider(t *testing.T, scale float64, fixtures ...fixture) *Provider {
	t.Helper()
	p, err := New("replay", strings.NewReader(jsonl(t, fixtures...)), profiles, scale, 7)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func request(model string, contents ...string) provider.Request {
	msgs := make([]provider.Message, len(contents))
	for i, c := range contents {
		msgs[i] = provider.Message{Role: provider.RoleUser, Content: c}
	}
	return provider.Request{Model: model, MaxTokens: 100, Messages: msgs}
}

func drain(s provider.Stream) ([]provider.StreamEvent, error) {
	var evs []provider.StreamEvent
	for {
		ev, err := s.Recv()
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
}

func TestReplayCompleteReturnsRecorded(t *testing.T) {
	p := newProvider(t, 0.001,
		fixture{"prompt one", "m-a", "answer a1", 11, 22},
		fixture{"prompt one", "m-b", "answer b1", 33, 44},
		fixture{"prompt two", "m-a", "answer a2", 55, 66},
	)
	for _, tc := range []struct {
		prompt, model, text string
		usage               provider.Usage
	}{
		{"prompt one", "m-a", "answer a1", provider.Usage{PromptTokens: 11, CompletionTokens: 22}},
		{"prompt one", "m-b", "answer b1", provider.Usage{PromptTokens: 33, CompletionTokens: 44}},
		{"prompt two", "m-a", "answer a2", provider.Usage{PromptTokens: 55, CompletionTokens: 66}},
	} {
		resp, err := p.Complete(bg, request(tc.model, tc.prompt))
		if err != nil {
			t.Fatalf("%s/%s: %v", tc.model, tc.prompt, err)
		}
		if resp.Content != tc.text || resp.FinishReason != "stop" {
			t.Errorf("%s/%s: got %q finish %q, want %q finish stop", tc.model, tc.prompt, resp.Content, resp.FinishReason, tc.text)
		}
		if resp.Usage == nil || *resp.Usage != tc.usage {
			t.Errorf("%s/%s: usage = %+v, want %+v", tc.model, tc.prompt, resp.Usage, tc.usage)
		}
	}
}

func TestReplayNeverTruncates(t *testing.T) {
	long := strings.Repeat("word ", 400)
	p := newProvider(t, 0.001, fixture{"p", "m-a", long, 1, 400})
	req := request("m-a", "p")
	req.MaxTokens = 1
	resp, err := p.Complete(bg, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != long || resp.FinishReason != "stop" {
		t.Fatalf("content length %d finish %q, want the full %d bytes and stop", len(resp.Content), resp.FinishReason, len(long))
	}
	s, err := p.Stream(bg, req)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, _ := drain(s)
	var sb strings.Builder
	for _, ev := range evs {
		sb.WriteString(ev.Delta)
	}
	if sb.String() != long {
		t.Fatalf("streamed %d bytes, want the full %d", sb.Len(), len(long))
	}
}

func TestReplayUnknownPromptIsBadRequest(t *testing.T) {
	p := newProvider(t, 0.001, fixture{"secret prompt text", "m-a", "answer", 1, 1})
	for name, req := range map[string]provider.Request{
		"unknown prompt": request("m-a", "another secret prompt"),
		"unknown model":  request("m-b", "secret prompt text"),
	} {
		check := func(err error) {
			t.Helper()
			var pe *provider.Error
			if !errors.As(err, &pe) {
				t.Fatalf("%s: err = %v, want *provider.Error", name, err)
			}
			if pe.Kind != provider.KindBadRequest || pe.Provider != "replay" ||
				pe.Message != "replay: no recorded output for this prompt and model" {
				t.Errorf("%s: got %+v", name, pe)
			}
			if strings.Contains(err.Error(), "secret") {
				t.Errorf("%s: error %q leaks prompt text", name, err)
			}
		}
		_, err := p.Complete(bg, req)
		check(err)
		_, err = p.Stream(bg, req)
		check(err)
	}
}

func TestReplayKeyIsFeatureTextSHA(t *testing.T) {
	// The key is sha256 of the message contents joined with a newline; roles
	// do not matter.
	p := newProvider(t, 0.001, fixture{"first\nsecond", "m-a", "joined", 1, 1})
	req := provider.Request{Model: "m-a", MaxTokens: 10, Messages: []provider.Message{
		{Role: provider.RoleSystem, Content: "first"},
		{Role: provider.RoleAssistant, Content: "second"},
	}}
	resp, err := p.Complete(bg, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "joined" {
		t.Fatalf("content = %q", resp.Content)
	}
	if _, err := p.Complete(bg, request("m-a", "firstsecond")); err == nil {
		t.Fatal("messages joined without a newline must not match")
	}
	if _, err := p.Complete(bg, request("m-a", "first\nsecond")); err != nil {
		t.Fatalf("one message holding the joined text must match: %v", err)
	}
}

func TestReplayStreamChunksUTF8(t *testing.T) {
	// Three-byte runes cannot be split at 16 bytes, so every chunk holds five
	// of them (15 bytes).
	cjk := strings.Repeat("日", 30)
	// A four-byte rune straddling the 16-byte mark moves whole to the next chunk.
	mixed := "abcdefghijklmno" + "😀" + "tail 日本語 end"
	for _, text := range []string{cjk, mixed, "short", strings.Repeat("a", 33)} {
		p := newProvider(t, 0.001, fixture{"p", "m-a", text, 9, 8})
		s, err := p.Stream(bg, request("m-a", "p"))
		if err != nil {
			t.Fatal(err)
		}
		evs, err := drain(s)
		s.Close()
		if err != io.EOF {
			t.Fatalf("stream ended with %v, want io.EOF", err)
		}
		if len(evs) < 2 {
			t.Fatalf("got %d events, want chunks plus a final event", len(evs))
		}
		var sb strings.Builder
		for i, ev := range evs[:len(evs)-1] {
			if ev.Delta == "" || ev.FinishReason != "" || ev.Usage != nil {
				t.Fatalf("event %d is not a pure content event: %+v", i, ev)
			}
			if len(ev.Delta) > 16 || !utf8.ValidString(ev.Delta) {
				t.Fatalf("event %d delta %q: %d bytes, valid=%v", i, ev.Delta, len(ev.Delta), utf8.ValidString(ev.Delta))
			}
			sb.WriteString(ev.Delta)
		}
		if sb.String() != text {
			t.Fatalf("concatenated chunks %q != text %q", sb.String(), text)
		}
		last := evs[len(evs)-1]
		if last.Delta != "" || last.FinishReason != "stop" || last.Usage == nil ||
			*last.Usage != (provider.Usage{PromptTokens: 9, CompletionTokens: 8}) {
			t.Fatalf("final event = %+v", last)
		}
		if _, err := s.Recv(); err != io.EOF {
			t.Fatalf("Recv after the end = %v, want io.EOF", err)
		}
	}

	p := newProvider(t, 0.001, fixture{"p", "m-a", cjk, 1, 1})
	s, err := p.Stream(bg, request("m-a", "p"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, _ := drain(s)
	if got := len(evs) - 1; got != 6 {
		t.Fatalf("30 three-byte runes gave %d chunks, want 6", got)
	}
}

func TestReplayEmptyTextStillFinishes(t *testing.T) {
	p := newProvider(t, 0.001, fixture{"p", "m-a", "", 4, 0})
	s, err := p.Stream(bg, request("m-a", "p"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	evs, err := drain(s)
	if err != io.EOF || len(evs) != 1 || evs[0].FinishReason != "stop" || evs[0].Usage == nil {
		t.Fatalf("events %+v, err %v; want one final event then EOF", evs, err)
	}
}

func TestReplayLatencyScaled(t *testing.T) {
	// 100 ms TTFT plus 10 tokens at 100 tokens/s is 200 ms unscaled.
	p := newProvider(t, 0.001, fixture{"p", "m-a", "twenty bytes of text", 1, 10})
	start := time.Now()
	if _, err := p.Complete(bg, request("m-a", "p")); err != nil {
		t.Fatal(err)
	}
	s, err := p.Stream(bg, request("m-a", "p"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := drain(s); err != io.EOF {
		t.Fatal(err)
	}
	if d := time.Since(start); d >= 50*time.Millisecond {
		t.Fatalf("a Complete and a Stream took %v with time_scale 0.001, want well under 50ms", d)
	}
}

func TestReplayLatencyApplied(t *testing.T) {
	// 100 ms TTFT plus 10 tokens at 100 tokens/s: 200 ms in total, at scale 0.5.
	p := newProvider(t, 0.5, fixture{"p", "m-a", "x", 1, 10})
	const total, ttft = 100 * time.Millisecond, 50 * time.Millisecond

	start := time.Now()
	if _, err := p.Complete(bg, request("m-a", "p")); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < total {
		t.Fatalf("Complete took %v, want at least %v", d, total)
	}

	start = time.Now()
	s, err := p.Stream(bg, request("m-a", "p"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if d := time.Since(start); d < ttft {
		t.Fatalf("Stream returned after %v, want at least the %v TTFT", d, ttft)
	}
	if _, err := drain(s); err != io.EOF {
		t.Fatal(err)
	}
	if d := time.Since(start); d < total {
		t.Fatalf("Stream finished after %v, want at least %v", d, total)
	}
}

func TestDrawIsSeeded(t *testing.T) {
	if got, want := draw(7, 3), rand.New(rand.NewPCG(7, 3)).Float64(); got != want {
		t.Fatalf("draw(7, 3) = %v, want the first Float64 of PCG(7, 3) = %v", got, want)
	}
	if draw(7, 3) != draw(7, 3) {
		t.Fatal("draw is not deterministic")
	}
	if draw(7, 3) == draw(7, 4) || draw(7, 3) == draw(8, 3) {
		t.Fatal("draw ignores the call index or the seed")
	}
}

func TestLatencyFormula(t *testing.T) {
	prof := profile{TTFTMS: 200, DecodeTokensPerS: 100, Jitter: 0.1}
	near := func(got, want time.Duration) bool { return got-want < time.Microsecond && want-got < time.Microsecond }
	// 200 ms TTFT plus 50 tokens at 100 tokens/s (500 ms), times jitter and scale.
	for _, tc := range []struct {
		u, factor float64
	}{{0, 0.9}, {0.5, 1}, {0.999999, 1.1}} {
		ttft, decode := latency(prof, 50, 0.5, tc.u)
		wantTTFT := time.Duration(200 * tc.factor * 0.5 * float64(time.Millisecond))
		wantDecode := time.Duration(500 * tc.factor * 0.5 * float64(time.Millisecond))
		if !near(ttft, wantTTFT) || !near(decode, wantDecode) {
			t.Errorf("u=%v: ttft %v decode %v, want about %v and %v", tc.u, ttft, decode, wantTTFT, wantDecode)
		}
	}
}

func TestReplayStreamHonorsContext(t *testing.T) {
	slow := Profiles{"m-a": {TTFTMS: 5000, DecodeTokensPerS: 100}}
	quick := Profiles{"m-a": {TTFTMS: 1, DecodeTokensPerS: 10}} // 100 tokens take 10 s
	build := func(prof Profiles) *Provider {
		p, err := New("replay", strings.NewReader(jsonl(t, fixture{"p", "m-a", strings.Repeat("x", 48), 1, 100})), prof, 1, 7)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	under := func(t *testing.T, start time.Time) {
		t.Helper()
		if d := time.Since(start); d >= time.Second {
			t.Fatalf("took %v", d)
		}
	}

	t.Run("cancel while waiting for the first byte", func(t *testing.T) {
		ctx, cancel := context.WithCancel(bg)
		time.AfterFunc(50*time.Millisecond, cancel)
		start := time.Now()
		if _, err := build(slow).Stream(ctx, request("m-a", "p")); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		under(t, start)
	})
	t.Run("cancel in Complete", func(t *testing.T) {
		ctx, cancel := context.WithCancel(bg)
		time.AfterFunc(50*time.Millisecond, cancel)
		start := time.Now()
		if _, err := build(slow).Complete(ctx, request("m-a", "p")); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		under(t, start)
	})
	t.Run("deadline in Complete", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(bg, 50*time.Millisecond)
		defer cancel()
		_, err := build(slow).Complete(ctx, request("m-a", "p"))
		var pe *provider.Error
		if !errors.As(err, &pe) || pe.Kind != provider.KindTimeout {
			t.Fatalf("err = %v, want a timeout *provider.Error", err)
		}
	})
	t.Run("cancel between chunks", func(t *testing.T) {
		ctx, cancel := context.WithCancel(bg)
		s, err := build(quick).Stream(ctx, request("m-a", "p"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if _, err := s.Recv(); err != nil {
			t.Fatalf("first Recv must not wait: %v", err)
		}
		time.AfterFunc(50*time.Millisecond, cancel)
		start := time.Now()
		if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
			t.Fatalf("Recv err = %v, want context.Canceled", err)
		}
		under(t, start)
	})
	t.Run("canceled before the first event", func(t *testing.T) {
		ctx, cancel := context.WithCancel(bg)
		s, err := build(quick).Stream(ctx, request("m-a", "p"))
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		cancel()
		if _, err := s.Recv(); !errors.Is(err, context.Canceled) {
			t.Fatalf("first Recv err = %v, want context.Canceled", err)
		}
	})
	t.Run("close wakes a blocked Recv", func(t *testing.T) {
		s, err := build(quick).Stream(bg, request("m-a", "p"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Recv(); err != nil {
			t.Fatal(err)
		}
		time.AfterFunc(50*time.Millisecond, func() { _ = s.Close() })
		start := time.Now()
		if _, err := s.Recv(); err == nil {
			t.Fatal("Recv after Close returned nil")
		}
		under(t, start)
		if err := s.Close(); err != nil {
			t.Fatalf("second Close = %v", err)
		}
	})
}

func TestReplayDuplicateRecordFirstWins(t *testing.T) {
	p := newProvider(t, 0.001,
		fixture{"p", "m-a", "first", 1, 1},
		fixture{"p", "m-a", "second", 2, 2},
	)
	resp, err := p.Complete(bg, request("m-a", "p"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "first" {
		t.Fatalf("content = %q, want the first record", resp.Content)
	}
}

func TestReplayConcurrentCalls(t *testing.T) {
	p := newProvider(t, 0.001, fixture{"p", "m-a", strings.Repeat("c", 100), 1, 5})
	var wg sync.WaitGroup
	for range 40 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			if _, err := p.Complete(bg, request("m-a", "p")); err != nil {
				t.Error(err)
			}
		}()
		go func() {
			defer wg.Done()
			s, err := p.Stream(bg, request("m-a", "p"))
			if err != nil {
				t.Error(err)
				return
			}
			defer s.Close()
			if _, err := drain(s); err != io.EOF {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if got := p.calls.Load(); got != 80 {
		t.Fatalf("calls = %d, want 80", got)
	}
}

func TestNewRejects(t *testing.T) {
	ok := digest("p")
	line := func(format string, args ...any) string { return fmt.Sprintf(format, args...) }
	tests := []struct {
		name    string
		input   string
		scale   float64
		profs   Profiles
		wantErr string
	}{
		{"empty input", "", 1, profiles, "replay: no records"},
		{"not json", "{nope", 1, profiles, "replay: record 1:"},
		{"missing digest", line(`{"model":"m-a","text":"t","prompt_tokens":1,"completion_tokens":1}`), 1, profiles, "record 1: prompt_sha256 must be 64 lowercase hex characters"},
		{"short digest", line(`{"prompt_sha256":"abc","model":"m-a","text":"t","prompt_tokens":1,"completion_tokens":1}`), 1, profiles, "record 1: prompt_sha256 must be 64 lowercase hex characters"},
		{"missing model", line(`{"prompt_sha256":%q,"text":"t","prompt_tokens":1,"completion_tokens":1}`, ok), 1, profiles, "record 1: model is required"},
		{"missing text", line(`{"prompt_sha256":%q,"model":"m-a","prompt_tokens":1,"completion_tokens":1}`, ok), 1, profiles, "record 1: text is required"},
		{"missing tokens", line(`{"prompt_sha256":%q,"model":"m-a","text":"t"}`, ok), 1, profiles, "record 1: prompt_tokens and completion_tokens must be present and not negative"},
		{"negative tokens", line(`{"prompt_sha256":%q,"model":"m-a","text":"t","prompt_tokens":-1,"completion_tokens":1}`, ok), 1, profiles, "record 1: prompt_tokens and completion_tokens must be present and not negative"},
		{"no profile for the model", line(`{"prompt_sha256":%q,"model":"m-x","text":"t","prompt_tokens":1,"completion_tokens":1}`, ok), 1, profiles, `record 1: model "m-x" has no latency profile`},
		{"error names the failing line", line(`{"prompt_sha256":%q,"model":"m-a","text":"t","prompt_tokens":1,"completion_tokens":1}`, ok) + "\n" + `{"model":"m-a"}`, 1, profiles, "record 2:"},
		{"time scale zero", jsonl(t, fixture{"p", "m-a", "t", 1, 1}), 0, profiles, "time_scale"},
		{"time scale above one", jsonl(t, fixture{"p", "m-a", "t", 1, 1}), 1.5, profiles, "time_scale"},
		{"time scale negative", jsonl(t, fixture{"p", "m-a", "t", 1, 1}), -1, profiles, "time_scale"},
		{"bad profile", jsonl(t, fixture{"p", "m-a", "t", 1, 1}), 1, Profiles{"m-a": {TTFTMS: 100, DecodeTokensPerS: 0}}, `profile "m-a"`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("replay", strings.NewReader(tc.input), tc.profs, tc.scale, 1)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want one containing %q", err, tc.wantErr)
			}
		})
	}
}

func write(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const profilesYAML = `# SIMULATED test profiles
simulated: true
seed: 1
jitter: 0.1
models:
  - {name: m-a, rank: 0, ttft_ms: 200, decode_tokens_per_s: 120}
  - {name: m-b, rank: 1, ttft_ms: 300, decode_tokens_per_s: 105}
`

func TestLoadProfiles(t *testing.T) {
	dir := t.TempDir()
	got, err := LoadProfiles(write(t, dir, "ok.yaml", profilesYAML))
	if err != nil {
		t.Fatal(err)
	}
	want := Profiles{
		"m-a": {TTFTMS: 200, DecodeTokensPerS: 120, Jitter: 0.1},
		"m-b": {TTFTMS: 300, DecodeTokensPerS: 105, Jitter: 0.1},
	}
	if len(got) != len(want) || got["m-a"] != want["m-a"] || got["m-b"] != want["m-b"] {
		t.Fatalf("profiles = %+v, want %+v", got, want)
	}

	rejects := map[string]string{
		"not simulated":    strings.Replace(profilesYAML, "simulated: true", "simulated: false", 1),
		"simulated absent": strings.Replace(profilesYAML, "simulated: true\n", "", 1),
		"unknown field":    strings.Replace(profilesYAML, "seed: 1\n", "seed: 1\nextra: 1\n", 1),
		"no models":        "simulated: true\njitter: 0.1\nmodels: []\n",
		"duplicate name":   strings.Replace(profilesYAML, "name: m-b", "name: m-a", 1),
		"empty name":       strings.Replace(profilesYAML, "name: m-b", "name: \"\"", 1),
		"zero ttft":        strings.Replace(profilesYAML, "ttft_ms: 200", "ttft_ms: 0", 1),
		"zero decode":      strings.Replace(profilesYAML, "decode_tokens_per_s: 120", "decode_tokens_per_s: 0", 1),
		"jitter one":       strings.Replace(profilesYAML, "jitter: 0.1", "jitter: 1", 1),
		"not yaml":         "simulated: [",
	}
	for name, doc := range rejects {
		if _, err := LoadProfiles(write(t, dir, "bad.yaml", doc)); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
	if _, err := LoadProfiles(filepath.Join(dir, "absent.yaml")); err == nil {
		t.Error("absent file: want an error")
	}
}

func TestLoadProfilesCommitted(t *testing.T) {
	got, err := LoadProfiles("../../../bench/latency_profiles.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 6 {
		t.Fatalf("got %d profiles, want 6", len(got))
	}
	if want := (profile{TTFTMS: 600, DecodeTokensPerS: 60, Jitter: 0.1}); got["gpt-5"] != want {
		t.Fatalf("gpt-5 = %+v, want %+v", got["gpt-5"], want)
	}
}

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	recs := write(t, dir, "test.jsonl", jsonl(t, fixture{"hello", "m-a", "world", 3, 5}))
	profs := write(t, dir, "profiles.yaml", profilesYAML)
	p, err := LoadFile("replay", recs, profs, 0.001, 20261026)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Complete(bg, request("m-a", "hello"))
	if err != nil || resp.Content != "world" {
		t.Fatalf("resp %+v, err %v", resp, err)
	}

	if _, err := LoadFile("replay", filepath.Join(dir, "absent.jsonl"), profs, 1, 1); err == nil {
		t.Error("absent records file: want an error")
	}
	if _, err := LoadFile("replay", recs, filepath.Join(dir, "absent.yaml"), 1, 1); err == nil {
		t.Error("absent profiles file: want an error")
	}
}
