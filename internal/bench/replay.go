package bench

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// ParityThreshold is the agreement every deterministic strategy must reach for
// the gateway to pass the replay benchmark: the share of requests on which it
// chooses the model the trainer chose offline.
const ParityThreshold = 0.999

// ReplayLatencyNote is the text of ReplayResults.LatencyNote.
const ReplayLatencyNote = "The gateway's latency strategy adapts its estimates to observed attempts; " +
	"the offline latency router uses the SIMULATED priors only, so its agreement is reported but not part of the parity check."

// maxMismatches is how many disagreements a strategy's result lists.
const maxMismatches = 50

// replayGroup is one set of offline decisions: a strategy and, for quality,
// the profile.
type replayGroup struct{ strategy, profile string }

func (g replayGroup) label() string {
	if g.profile == "" {
		return g.strategy
	}
	return g.strategy + "/" + g.profile
}

// deterministic reports whether the group's decisions are a pure function of
// the request and the policy, which is what parity checks: the gateway must
// implement the offline decision function. Latency is the one that is not. Its
// gateway estimates move with every attempt the gateway observes,
// while the offline router uses the frozen simulated priors, so the two are
// different functions once the gateway has served traffic.
func (g replayGroup) deterministic() bool { return g.strategy != "latency" }

// replayGroups lists the groups in the order they run. Latency goes first, on a
// gateway no other strategy has touched: its choice follows the latency the
// gateway has observed, and only the latency pick is observed while it runs.
// The strategy names are the offline names, which are the gateway's.
var replayGroups = []replayGroup{
	{"latency", ""},
	{"strongest", ""},
	{"cheapest", ""},
	{"static", ""},
	{"quality", "strict"},
	{"quality", "balanced"},
	{"quality", "economy"},
}

// ReplayOptions configures RunReplay.
type ReplayOptions struct {
	Client *Client
	// Root is the repository root. The replay files are read from
	// trainer/artifacts/replay under it.
	Root string
	// Workers is how many requests are in flight at once.
	Workers int
}

// Mismatch is a request on which the gateway did not choose the offline model.
// Gateway is the model that served it, or the error code if it failed. It names
// the prompt by id and carries no text.
type Mismatch struct {
	PromptID string `json:"prompt_id"`
	Offline  string `json:"offline"`
	Gateway  string `json:"gateway"`
}

// StrategyResult is the outcome of one group of requests.
type StrategyResult struct {
	Strategy  string  `json:"strategy"`
	Profile   string  `json:"profile,omitempty"`
	N         int     `json:"n"`
	Agree     int     `json:"agree"`
	Agreement float64 `json:"agreement"` // Agree / N
	// Deterministic is whether the strategy is a pure function of the request
	// and the policy, so that its agreement is part of the parity check.
	Deterministic bool `json:"deterministic"`
	// Mismatches lists the first disagreements in prompt order, at most
	// maxMismatches of them.
	Mismatches []Mismatch     `json:"mismatches"`
	Errors     map[string]int `json:"errors"` // failed requests by error code

	// Accuracy, TotalCostUSD and CostPerSuccessUSD are what the gateway's
	// choices would have realized on the recorded outcomes of the test split.
	// A request that failed counts as a failure and costs nothing.
	Accuracy          float64  `json:"accuracy"`
	TotalCostUSD      float64  `json:"total_cost_usd"`
	CostPerSuccessUSD *float64 `json:"cost_per_success_usd"` // null when nothing succeeded
}

// ReplayResults is what RunReplay measured.
type ReplayResults struct {
	Prompts    int                       `json:"prompts"`
	Requests   int                       `json:"requests"`
	Strategies map[string]StrategyResult `json:"strategies"` // by "strategy" or "quality/<profile>"
	// DeterministicAgreement is the agreement pooled over the deterministic
	// groups and OverallAgreement the one over all groups.
	DeterministicAgreement float64 `json:"deterministic_agreement"`
	OverallAgreement       float64 `json:"overall_agreement"`
	// ParityPass is true when every deterministic group reached ParityThreshold.
	ParityPass bool `json:"parity_pass"`
	// LatencyNote says why the latency group is reported but not checked.
	LatencyNote string `json:"latency_note"`
}

type promptRecord struct {
	PromptID     string `json:"prompt_id"`
	Prompt       string `json:"prompt"`
	PromptSHA256 string `json:"prompt_sha256"`
}

type decisionRecord struct {
	PromptID string `json:"prompt_id"`
	Strategy string `json:"strategy"`
	Profile  string `json:"profile"` // null for the strategies without one
	Chosen   string `json:"chosen"`
}

type outcomeRecord struct {
	PromptSHA256 string  `json:"prompt_sha256"`
	Model        string  `json:"model"`
	Success      bool    `json:"success"`
	CostUSD      float64 `json:"cost_usd"`
}

type outcomeKey struct{ sha, model string }

type recorded struct {
	success bool
	costUSD float64
}

type offlineChoice struct{ promptID, chosen string }

type replayPrompt struct{ text, sha string }

// replayData is the content of the three replay files.
type replayData struct {
	prompts  map[string]replayPrompt
	groups   map[replayGroup][]offlineChoice // in file order, which is prompt order
	outcomes map[outcomeKey]recorded
}

// readJSONL calls each with every record of the JSON-lines file at path. An
// error names the file and the record, never its content.
func readJSONL[T any](path string, each func(row T) error) error {
	name := filepath.Base(path)
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", name, err)
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	for n := 1; ; n++ {
		var row T
		if err := dec.Decode(&row); errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return fmt.Errorf("read %s: record %d: %w", name, n, err)
		}
		if err := each(row); err != nil {
			return fmt.Errorf("read %s: record %d: %w", name, n, err)
		}
	}
}

// loadReplayData reads the replay files from dir. test.jsonl holds the model
// outputs of the test split; only whether each succeeded and what it cost is
// kept, so the output text is read past and dropped.
func loadReplayData(dir string) (*replayData, error) {
	data := &replayData{
		prompts:  map[string]replayPrompt{},
		groups:   map[replayGroup][]offlineChoice{},
		outcomes: map[outcomeKey]recorded{},
	}
	err := readJSONL(filepath.Join(dir, "prompts.jsonl"), func(r promptRecord) error {
		if r.PromptID == "" || r.PromptSHA256 == "" {
			return errors.New("prompt_id and prompt_sha256 are required")
		}
		if _, dup := data.prompts[r.PromptID]; dup {
			return fmt.Errorf("duplicate prompt_id %q", r.PromptID)
		}
		data.prompts[r.PromptID] = replayPrompt{text: r.Prompt, sha: r.PromptSHA256}
		return nil
	})
	if err != nil {
		return nil, err
	}

	known := map[replayGroup]bool{}
	for _, g := range replayGroups {
		known[g] = true
	}
	total := 0
	err = readJSONL(filepath.Join(dir, "decisions.jsonl"), func(r decisionRecord) error {
		g := replayGroup{r.Strategy, r.Profile}
		if !known[g] {
			return fmt.Errorf("unexpected offline decision: strategy %q with profile %q", r.Strategy, r.Profile)
		}
		if _, ok := data.prompts[r.PromptID]; !ok {
			return fmt.Errorf("decision for prompt_id %q, which prompts.jsonl does not have", r.PromptID)
		}
		if r.Chosen == "" {
			return fmt.Errorf("decision for prompt_id %q chooses no model", r.PromptID)
		}
		data.groups[g] = append(data.groups[g], offlineChoice{promptID: r.PromptID, chosen: r.Chosen})
		total++
		return nil
	})
	if err != nil {
		return nil, err
	}
	if total == 0 {
		return nil, errors.New("decisions.jsonl has no decisions")
	}

	err = readJSONL(filepath.Join(dir, "test.jsonl"), func(r outcomeRecord) error {
		k := outcomeKey{r.PromptSHA256, r.Model}
		if _, dup := data.outcomes[k]; !dup { // the replay provider keeps the first record too
			data.outcomes[k] = recorded{success: r.Success, costUSD: r.CostUSD}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return data, nil
}

// RunReplay checks the gateway's decisions against the trainer's. For every
// prompt of the test split and every offline decision about it, it sends one
// request that asks for that strategy (and profile) and compares the model
// that served it with the offline choice. The groups run one at a time, in the
// order of replayGroups, each with o.Workers requests in flight.
//
// Parity covers the deterministic groups (see replayGroup.deterministic); the
// latency group is run and reported but does not count towards ParityPass or
// DeterministicAgreement. The data must hold a decision of at least one
// deterministic group.
//
// A request that fails with an HTTP status counts as a disagreement and as an
// error of that code. A request that gets no response ends the run with an
// error: a gateway that cannot be reached says nothing about its decisions.
func RunReplay(ctx context.Context, o ReplayOptions) (ReplayResults, error) {
	if o.Client == nil {
		return ReplayResults{}, errors.New("replay options need a client")
	}
	if o.Workers < 1 {
		return ReplayResults{}, fmt.Errorf("workers must be at least 1, got %d", o.Workers)
	}
	defer o.Client.Close() // the gateway is stopped next, and waits for idle connections to go
	data, err := loadReplayData(filepath.Join(o.Root, "trainer", "artifacts", "replay"))
	if err != nil {
		return ReplayResults{}, err
	}
	hasDeterministic := false
	for g, rows := range data.groups {
		hasDeterministic = hasDeterministic || (g.deterministic() && len(rows) > 0)
	}
	if !hasDeterministic {
		return ReplayResults{}, errors.New("decisions.jsonl has no decision of a deterministic strategy, so there is no parity to check")
	}

	res := ReplayResults{Prompts: len(data.prompts), Strategies: map[string]StrategyResult{}, LatencyNote: ReplayLatencyNote}
	var agree, detRequests, detAgree int
	pass := true
	for _, g := range replayGroups {
		rows := data.groups[g]
		if len(rows) == 0 {
			continue
		}
		sr, err := runGroup(ctx, o, data, g, rows)
		if err != nil {
			return ReplayResults{}, err
		}
		res.Strategies[g.label()] = sr
		res.Requests += sr.N
		agree += sr.Agree
		if sr.Deterministic {
			detRequests += sr.N
			detAgree += sr.Agree
			pass = pass && sr.Agreement >= ParityThreshold
		}
	}
	res.OverallAgreement = float64(agree) / float64(res.Requests)
	res.DeterministicAgreement = float64(detAgree) / float64(detRequests)
	res.ParityPass = pass
	return res, nil
}

// runGroup sends the requests of one group and scores the answers.
func runGroup(ctx context.Context, o ReplayOptions, data *replayData, g replayGroup, rows []offlineChoice) (StrategyResult, error) {
	headers := map[string]string{"X-RouteForge-Strategy": g.strategy}
	if g.profile != "" {
		headers["X-RouteForge-Profile"] = g.profile
	}
	answers := make([]ChatOutcome, len(rows))
	err := forEach(ctx, len(rows), o.Workers, func(ctx context.Context, i int) error {
		out, err := o.Client.Chat(ctx, data.prompts[rows[i].promptID].text, headers)
		if err != nil {
			return fmt.Errorf("%s request for prompt %s: %w", g.label(), rows[i].promptID, err)
		}
		answers[i] = out
		return nil
	})
	if err != nil {
		return StrategyResult{}, err
	}

	r := StrategyResult{
		Strategy:      g.strategy,
		Profile:       g.profile,
		N:             len(rows),
		Deterministic: g.deterministic(),
		Mismatches:    []Mismatch{},
		Errors:        map[string]int{},
	}
	successes := 0
	for i, row := range rows {
		a := answers[i]
		served := a.Model
		if a.Status != http.StatusOK {
			served = a.ErrorCode
			r.Errors[a.ErrorCode]++
		} else {
			oc, ok := data.outcomes[outcomeKey{data.prompts[row.promptID].sha, a.Model}]
			if !ok {
				return StrategyResult{}, fmt.Errorf("test.jsonl has no outcome of model %q for prompt %s (%s)", a.Model, row.promptID, g.label())
			}
			if oc.success {
				successes++
			}
			r.TotalCostUSD += oc.costUSD
		}
		switch {
		case a.Status == http.StatusOK && a.Model == row.chosen:
			r.Agree++
		case len(r.Mismatches) < maxMismatches:
			r.Mismatches = append(r.Mismatches, Mismatch{PromptID: row.promptID, Offline: row.chosen, Gateway: served})
		}
	}
	r.Agreement = float64(r.Agree) / float64(r.N)
	r.Accuracy = float64(successes) / float64(r.N)
	if successes > 0 {
		cps := r.TotalCostUSD / float64(successes)
		r.CostPerSuccessUSD = &cps
	}
	return r, nil
}

// forEach calls fn for every index in [0, n) with at most workers calls at
// once, handing out the indexes in increasing order. It stops handing out work
// at the first error, which it returns once the calls in flight have finished;
// fn's ctx ends then. It returns the error of ctx if that ended the run
// early, and nil when every call succeeded.
func forEach(ctx context.Context, n, workers int, fn func(ctx context.Context, i int) error) error {
	if n <= 0 {
		return nil
	}
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		next, done atomic.Int64
		once       sync.Once
		first      error
		wg         sync.WaitGroup
	)
	for range min(max(workers, 1), n) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for runCtx.Err() == nil {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				if err := fn(runCtx, i); err != nil {
					once.Do(func() { first = err; cancel() })
					return
				}
				done.Add(1)
			}
		}()
	}
	wg.Wait()
	if first != nil {
		return first
	}
	if int(done.Load()) < n {
		return ctx.Err()
	}
	return nil
}
