// Package replay is a provider that serves recorded model outputs. Benchmarks
// use it to run the whole gateway over real outputs without calling a model:
// a request is answered with the output recorded for the same prompt and
// model, after a simulated delay drawn from a latency profile.
package replay

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"regexp"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"go.yaml.in/yaml/v3"

	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
)

// maxChunkBytes is the largest streamed delta.
const maxChunkBytes = 16

// finishReason ends every replayed response: recorded outputs are never
// truncated, so a replayed response always ran to its natural end.
const finishReason = "stop"

// profile is the simulated latency of one model.
type profile = struct{ TTFTMS, DecodeTokensPerS, Jitter float64 }

// Profiles maps an upstream model id to its simulated latency: time to first
// byte in ms, decode speed in tokens per second, and the half-width of the
// uniform jitter as a fraction of the total latency.
type Profiles map[string]profile

// checkProfile reports why p is not a usable latency profile.
func checkProfile(model string, p profile) error {
	finitePositive := func(v float64) bool { return v > 0 && !math.IsInf(v, 1) }
	switch {
	case !finitePositive(p.TTFTMS):
		return fmt.Errorf("replay: profile %q: ttft_ms must be finite and > 0", model)
	case !finitePositive(p.DecodeTokensPerS):
		return fmt.Errorf("replay: profile %q: decode_tokens_per_s must be finite and > 0", model)
	case !(p.Jitter >= 0 && p.Jitter < 1):
		return fmt.Errorf("replay: profile %q: jitter must be in [0, 1)", model)
	}
	return nil
}

// LoadProfiles reads a latency profile file such as bench/latency_profiles.yaml.
// The file must be marked `simulated: true`: its numbers are synthetic, and
// the marker keeps measurements from being mistaken for them. The file's
// jitter applies to every model.
func LoadProfiles(path string) (Profiles, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("replay: read latency profiles: %w", err)
	}
	var doc struct {
		Simulated bool    `yaml:"simulated"`
		Seed      int64   `yaml:"seed"` // read by the trainer; replay takes its seed from the config
		Jitter    float64 `yaml:"jitter"`
		Models    []struct {
			Name             string  `yaml:"name"`
			Rank             int     `yaml:"rank"`
			TTFTMS           float64 `yaml:"ttft_ms"`
			DecodeTokensPerS float64 `yaml:"decode_tokens_per_s"`
		} `yaml:"models"`
	}
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("replay: %s: %w", path, err)
	}
	if !doc.Simulated {
		return nil, fmt.Errorf("replay: %s: simulated must be true: these profiles are synthetic", path)
	}
	if len(doc.Models) == 0 {
		return nil, fmt.Errorf("replay: %s: models must not be empty", path)
	}
	out := make(Profiles, len(doc.Models))
	for _, m := range doc.Models {
		if m.Name == "" {
			return nil, fmt.Errorf("replay: %s: a model has no name", path)
		}
		if _, dup := out[m.Name]; dup {
			return nil, fmt.Errorf("replay: %s: duplicate model %q", path, m.Name)
		}
		p := profile{TTFTMS: m.TTFTMS, DecodeTokensPerS: m.DecodeTokensPerS, Jitter: doc.Jitter}
		if err := checkProfile(m.Name, p); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		out[m.Name] = p
	}
	return out, nil
}

// record is one recorded output.
type record struct {
	text             string
	promptTokens     int
	completionTokens int
}

// key finds a record: the SHA-256 of the prompt's feature text and the
// upstream model id.
type key struct{ prompt, model string }

var digestRE = regexp.MustCompile(`^[0-9a-f]{64}$`)

// Provider replays recorded outputs. It is safe for concurrent use.
type Provider struct {
	name     string
	records  map[key]record
	profiles Profiles
	scale    float64
	seed     uint64
	calls    atomic.Uint64
}

var _ provider.Provider = (*Provider)(nil)

// New reads the trainer's replay export (JSON lines) from records. Each line
// needs prompt_sha256, model, text, prompt_tokens and completion_tokens; other
// fields are ignored and only those five are kept. Every recorded model needs
// a profile. When a prompt and model are recorded twice the first record wins.
//
// timeScale multiplies every simulated latency and must be in (0, 1]; seed
// seeds the latency jitter. A call's jitter is drawn from a PCG seeded with
// (seed, index of the call), so a fixed call order gives fixed latencies.
func New(name string, records io.Reader, profiles Profiles, timeScale float64, seed uint64) (*Provider, error) {
	if !(timeScale > 0 && timeScale <= 1) {
		return nil, fmt.Errorf("replay: time_scale %v must be in (0, 1]", timeScale)
	}
	for model, p := range profiles {
		if err := checkProfile(model, p); err != nil {
			return nil, err
		}
	}

	table := make(map[key]record)
	dec := json.NewDecoder(records)
	for n := 1; ; n++ {
		var line struct {
			PromptSHA256     string  `json:"prompt_sha256"`
			Model            string  `json:"model"`
			Text             *string `json:"text"`
			PromptTokens     *int    `json:"prompt_tokens"`
			CompletionTokens *int    `json:"completion_tokens"`
		}
		err := dec.Decode(&line)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("replay: record %d: %w", n, err)
		}
		switch {
		case !digestRE.MatchString(line.PromptSHA256):
			return nil, fmt.Errorf("replay: record %d: prompt_sha256 must be 64 lowercase hex characters", n)
		case line.Model == "":
			return nil, fmt.Errorf("replay: record %d: model is required", n)
		case line.Text == nil:
			return nil, fmt.Errorf("replay: record %d: text is required", n)
		case line.PromptTokens == nil || line.CompletionTokens == nil || *line.PromptTokens < 0 || *line.CompletionTokens < 0:
			return nil, fmt.Errorf("replay: record %d: prompt_tokens and completion_tokens must be present and not negative", n)
		}
		if _, ok := profiles[line.Model]; !ok {
			return nil, fmt.Errorf("replay: record %d: model %q has no latency profile", n, line.Model)
		}
		k := key{line.PromptSHA256, line.Model}
		if _, dup := table[k]; !dup {
			table[k] = record{text: *line.Text, promptTokens: *line.PromptTokens, completionTokens: *line.CompletionTokens}
		}
	}
	if len(table) == 0 {
		return nil, errors.New("replay: no records")
	}
	return &Provider{name: name, records: table, profiles: profiles, scale: timeScale, seed: seed}, nil
}

// LoadFile builds a Provider from a records file and a latency profile file.
func LoadFile(name, path, profilesPath string, timeScale float64, seed uint64) (*Provider, error) {
	profiles, err := LoadProfiles(profilesPath)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("replay: open records: %w", err)
	}
	defer f.Close()
	return New(name, bufio.NewReaderSize(f, 1<<20), profiles, timeScale, seed)
}

// draw returns the jitter draw in [0, 1) of the call with the given index.
func draw(seed, index uint64) float64 {
	return rand.New(rand.NewPCG(seed, index)).Float64()
}

// latency returns the simulated time to first byte and the decode time of a
// response of completionTokens tokens. Both carry the same jitter factor, which
// is uniform in [1-Jitter, 1+Jitter) for u uniform in [0, 1), and the time
// scale:
//
//	(ttft + completionTokens * 1000 / decode) * (1 + jitter * (2u - 1)) * scale
func latency(p profile, completionTokens int, scale, u float64) (ttft, decode time.Duration) {
	factor := (1 + p.Jitter*(2*u-1)) * scale
	toDuration := func(ms float64) time.Duration { return time.Duration(ms * factor * float64(time.Millisecond)) }
	return toDuration(p.TTFTMS), toDuration(float64(completionTokens) * 1000 / p.DecodeTokensPerS)
}

// found is the outcome of looking up a request.
type found struct {
	rec    record
	ttft   time.Duration
	decode time.Duration
}

// lookup finds the record for req and draws the call's latency. Every call
// takes the next call index, found or not.
func (p *Provider) lookup(req provider.Request) (found, error) {
	index := p.calls.Add(1) - 1
	sum := sha256.Sum256([]byte(policy.FeatureText(req.Messages)))
	rec, ok := p.records[key{hex.EncodeToString(sum[:]), req.Model}]
	if !ok {
		return found{}, &provider.Error{
			Kind:     provider.KindBadRequest,
			Provider: p.name,
			Message:  "replay: no recorded output for this prompt and model",
		}
	}
	ttft, decode := latency(p.profiles[req.Model], rec.completionTokens, p.scale, draw(p.seed, index))
	return found{rec: rec, ttft: ttft, decode: decode}, nil
}

func (r record) usage() *provider.Usage {
	return &provider.Usage{PromptTokens: r.promptTokens, CompletionTokens: r.completionTokens}
}

// waitUntil sleeps until due. It returns a transport error as soon as ctx ends
// and a closed error when done is closed (a nil done never closes). It checks
// both before it sleeps, so a call that is already over never proceeds.
func (p *Provider) waitUntil(ctx context.Context, done <-chan struct{}, due time.Time) error {
	if err := ctx.Err(); err != nil {
		return provider.TransportError(ctx, p.name, err)
	}
	select {
	case <-done:
		return errClosed(p.name)
	default:
	}
	d := time.Until(due)
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
	return &provider.Error{Kind: provider.KindInternal, Provider: name, Message: "replay: stream closed"}
}

// Complete returns the recorded output after the simulated latency.
func (p *Provider) Complete(ctx context.Context, req provider.Request) (*provider.Response, error) {
	start := time.Now()
	f, err := p.lookup(req)
	if err != nil {
		return nil, err
	}
	if err := p.waitUntil(ctx, nil, start.Add(f.ttft+f.decode)); err != nil {
		return nil, err
	}
	return &provider.Response{Content: f.rec.text, FinishReason: finishReason, Usage: f.rec.usage()}, nil
}

// Stream waits the simulated time to first byte, then returns a stream of the
// recorded output in deltas of at most 16 bytes, split on UTF-8 boundaries and
// spread evenly over the decode time.
func (p *Provider) Stream(ctx context.Context, req provider.Request) (provider.Stream, error) {
	start := time.Now()
	f, err := p.lookup(req)
	if err != nil {
		return nil, err
	}
	first := start.Add(f.ttft)
	if err := p.waitUntil(ctx, nil, first); err != nil {
		return nil, err
	}
	// n sleeps of decode/n each: one before every delta after the first and
	// one before the final event, so the final event lands at ttft + decode.
	slots := max(countChunks(f.rec.text), 1)
	return &stream{
		p:     p,
		ctx:   ctx,
		rec:   f.rec,
		first: first,
		gap:   f.decode / time.Duration(slots),
		slots: slots,
		done:  make(chan struct{}),
	}, nil
}

// chunkEnd returns the end of the chunk that starts at pos: at most
// maxChunkBytes bytes, never splitting a UTF-8 sequence.
func chunkEnd(text string, pos int) int {
	end := pos + maxChunkBytes
	if end >= len(text) {
		return len(text)
	}
	for end > pos && !utf8.RuneStart(text[end]) {
		end--
	}
	if end == pos {
		// Only reachable for text that is not valid UTF-8, which JSON decoding
		// never produces. Advancing keeps a malformed record from looping.
		return pos + maxChunkBytes
	}
	return end
}

func countChunks(text string) int {
	n := 0
	for pos := 0; pos < len(text); pos = chunkEnd(text, pos) {
		n++
	}
	return n
}

// stream is one replayed streaming response. Recv must be called from a single
// goroutine; Close may be called concurrently with it.
type stream struct {
	p     *Provider
	ctx   context.Context
	rec   record
	first time.Time     // when the first delta is due
	gap   time.Duration // time between deltas
	slots int           // deltas, or 1 for an empty text

	pos   int // byte offset of the next delta
	index int // deltas sent
	err   error

	done chan struct{}
	once sync.Once
}

// Recv returns the next event: deltas, then one event carrying the finish
// reason and usage, then io.EOF.
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
	// Delta i is due i gaps after the first and the final event is due after
	// all of them. Due times come from one schedule, so timer overshoot on one
	// sleep does not delay the next.
	due := s.first.Add(time.Duration(s.index) * s.gap)
	if s.pos >= len(s.rec.text) {
		due = s.first.Add(time.Duration(s.slots) * s.gap)
	}
	if err := s.p.waitUntil(s.ctx, s.done, due); err != nil {
		return provider.StreamEvent{}, err
	}
	if s.pos < len(s.rec.text) {
		end := chunkEnd(s.rec.text, s.pos)
		delta := s.rec.text[s.pos:end]
		s.pos = end
		s.index++
		return provider.StreamEvent{Delta: delta}, nil
	}
	// The final event is followed by io.EOF through the sticky error.
	s.err = io.EOF
	return provider.StreamEvent{FinishReason: finishReason, Usage: s.rec.usage()}, nil
}

// Close releases the stream and wakes a blocked Recv. It is idempotent.
func (s *stream) Close() error {
	s.once.Do(func() { close(s.done) })
	return nil
}
