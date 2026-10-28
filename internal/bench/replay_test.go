package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// sha returns a 64 character digest-shaped key for prompt i.
func sha(i int) string { return fmt.Sprintf("%064x", i) }

func promptID(i int) string { return fmt.Sprintf("p%d", i) }

// promptRow, decisionRow and outcomeRow build the rows of the three replay
// files, with the keys the trainer writes.
func promptRow(i int, text string) map[string]any {
	return map[string]any{"prompt_id": promptID(i), "prompt": text, "prompt_sha256": sha(i), "dataset": "d", "family": "f"}
}

func decisionRow(i int, strategy, profile, chosen string) map[string]any {
	var p any // null for the strategies without a profile, as the trainer writes it
	if profile != "" {
		p = profile
	}
	return map[string]any{"prompt_id": promptID(i), "prompt_sha256": sha(i), "strategy": strategy, "profile": p, "chosen": chosen, "reason": nil}
}

func outcomeRow(i int, model string, success bool, cost float64) map[string]any {
	return map[string]any{
		"prompt_id": promptID(i), "prompt_sha256": sha(i), "model": model, "success": success, "cost_usd": cost,
		"dataset": "d", "prompt_tokens": 10, "completion_tokens": 20, "score": 1.0, "text": "RECORDED-COMPLETION-TEXT",
	}
}

func writeJSONL(t *testing.T, path string, rows []map[string]any) {
	t.Helper()
	var sb strings.Builder
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		sb.Write(b)
		sb.WriteByte('\n')
	}
	writeFile(t, path, sb.String())
}

// writeReplayFiles writes prompts.jsonl, decisions.jsonl and test.jsonl under
// root, where RunReplay reads them.
func writeReplayFiles(t *testing.T, root string, prompts, decisions, outcomes []map[string]any) {
	t.Helper()
	dir := filepath.Join(root, "trainer", "artifacts", "replay")
	writeJSONL(t, filepath.Join(dir, "prompts.jsonl"), prompts)
	writeJSONL(t, filepath.Join(dir, "decisions.jsonl"), decisions)
	writeJSONL(t, filepath.Join(dir, "test.jsonl"), outcomes)
}

// stubAnswer is what the stub gateway replies to one request.
type stubAnswer struct {
	status int    // 0 means 200
	model  string // served model of a 200
	code   string // error code of any other status
}

// stubGateway answers chat requests from a function of the routing group
// ("strategy" or "strategy/profile") and the prompt text, and records what it
// was sent.
type stubGateway struct {
	*httptest.Server
	mu          sync.Mutex
	order       []string      // the group of every request, in arrival order
	headers     []http.Header // the headers of every request, in arrival order
	inflight    int
	maxInflight int
}

func newStubGateway(t *testing.T, delay time.Duration, answer func(group, prompt string) stubAnswer) *stubGateway {
	t.Helper()
	g := &stubGateway{}
	g.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &req); err != nil || req.Model != "routeforge-auto" || req.Stream ||
			len(req.Messages) != 1 || req.Messages[0].Role != "user" {
			t.Errorf("unexpected request body %q", body)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		group := r.Header.Get("X-RouteForge-Strategy")
		if p := r.Header.Get("X-RouteForge-Profile"); p != "" {
			group += "/" + p
		}

		g.mu.Lock()
		g.order = append(g.order, group)
		g.headers = append(g.headers, r.Header.Clone())
		g.inflight++
		g.maxInflight = max(g.maxInflight, g.inflight)
		g.mu.Unlock()
		defer func() {
			g.mu.Lock()
			g.inflight--
			g.mu.Unlock()
		}()
		time.Sleep(delay)

		a := answer(group, req.Messages[0].Content)
		w.Header().Set("Content-Type", "application/json")
		if a.status != 0 && a.status != http.StatusOK {
			w.WriteHeader(a.status)
			_, _ = fmt.Fprintf(w, `{"error":{"message":"m","type":"t","code":%q}}`, a.code)
			return
		}
		w.Header().Set("X-RouteForge-Model", a.model)
		w.Header().Set("X-RouteForge-Reason", "stub")
		w.Header().Set("X-RouteForge-Decision-Id", "dec_stub")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"GATEWAY-COMPLETION-TEXT"}}],"routeforge":{"decision_id":"dec_stub","cost_usd":0.5}}`)
	}))
	t.Cleanup(g.Close)
	return g
}

func replayOptions(g *stubGateway, root string, workers int) ReplayOptions {
	return ReplayOptions{Client: NewClient(g.URL, testKey), Root: root, Workers: workers}
}

func requireClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("%s = %v, want %v", name, got, want)
	}
}

// threePrompts is the data of the agreement tests: three prompts, two
// strategies, and a gateway that disagrees with the offline choice of
// "cheapest" on the third prompt.
func threePrompts(t *testing.T) (root string, answer func(group, prompt string) stubAnswer) {
	t.Helper()
	root = t.TempDir()
	texts := map[int]string{1: "PROMPT-TEXT-ONE", 2: "PROMPT-TEXT-TWO", 3: "PROMPT-TEXT-THREE"}
	var prompts, decisions, outcomes []map[string]any
	for i := 1; i <= 3; i++ {
		prompts = append(prompts, promptRow(i, texts[i]))
		decisions = append(decisions,
			decisionRow(i, "strongest", "", "m-strong"),
			decisionRow(i, "cheapest", "", "m-cheap"))
		outcomes = append(outcomes, outcomeRow(i, "m-strong", true, 0.10), outcomeRow(i, "m-cheap", true, 0.01))
	}
	writeReplayFiles(t, root, prompts, decisions, outcomes)
	return root, func(group, prompt string) stubAnswer {
		if group == "cheapest" && prompt != "PROMPT-TEXT-THREE" {
			return stubAnswer{model: "m-cheap"}
		}
		return stubAnswer{model: "m-strong"}
	}
}

func TestReplayAgreement(t *testing.T) {
	root, answer := threePrompts(t)
	g := newStubGateway(t, 0, answer)

	res, err := RunReplay(context.Background(), replayOptions(g, root, 2))
	if err != nil {
		t.Fatal(err)
	}

	if res.Requests != 6 || res.Prompts != 3 {
		t.Errorf("requests = %d, prompts = %d, want 6 and 3", res.Requests, res.Prompts)
	}
	strongest := res.Strategies["strongest"]
	if strongest.N != 3 || strongest.Agree != 3 || strongest.Agreement != 1 || len(strongest.Mismatches) != 0 {
		t.Errorf("strongest = %+v, want 3 of 3 agreeing", strongest)
	}
	cheapest := res.Strategies["cheapest"]
	if cheapest.N != 3 || cheapest.Agree != 2 {
		t.Fatalf("cheapest n = %d, agree = %d, want 3 and 2", cheapest.N, cheapest.Agree)
	}
	requireClose(t, "cheapest agreement", cheapest.Agreement, 2.0/3)
	want := Mismatch{PromptID: "p3", Offline: "m-cheap", Gateway: "m-strong"}
	if len(cheapest.Mismatches) != 1 || cheapest.Mismatches[0] != want {
		t.Errorf("mismatches = %+v, want [%+v]", cheapest.Mismatches, want)
	}
	requireClose(t, "overall agreement", res.OverallAgreement, 5.0/6)
	if res.ParityPass {
		t.Error("parity_pass is true with an agreement of 5/6, want false")
	}

	// The result names prompts by id only and never carries their text or the
	// recorded or generated output.
	data, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	for _, leaked := range []string{"PROMPT-TEXT", "RECORDED-COMPLETION-TEXT", "GATEWAY-COMPLETION-TEXT"} {
		if strings.Contains(string(data), leaked) {
			t.Errorf("result contains %q: %s", leaked, data)
		}
	}
}

func TestReplayParityThreshold(t *testing.T) {
	// 1000 prompts: one disagreement is an agreement of 0.999, which passes;
	// two make 0.998, which does not.
	for _, c := range []struct {
		disagree int
		pass     bool
	}{{0, true}, {1, true}, {2, false}} {
		t.Run(fmt.Sprintf("%d disagreements", c.disagree), func(t *testing.T) {
			root := t.TempDir()
			var prompts, decisions, outcomes []map[string]any
			for i := 0; i < 1000; i++ {
				prompts = append(prompts, promptRow(i, fmt.Sprintf("text %d", i)))
				decisions = append(decisions, decisionRow(i, "strongest", "", "m-a"))
				outcomes = append(outcomes, outcomeRow(i, "m-a", true, 0.01), outcomeRow(i, "m-b", true, 0.01))
			}
			writeReplayFiles(t, root, prompts, decisions, outcomes)
			g := newStubGateway(t, 0, func(_, prompt string) stubAnswer {
				for i := 0; i < c.disagree; i++ {
					if prompt == fmt.Sprintf("text %d", i) {
						return stubAnswer{model: "m-b"}
					}
				}
				return stubAnswer{model: "m-a"}
			})

			res, err := RunReplay(context.Background(), replayOptions(g, root, 16))
			if err != nil {
				t.Fatal(err)
			}
			if res.ParityPass != c.pass {
				t.Errorf("parity_pass = %v at agreement %v, want %v", res.ParityPass, res.Strategies["strongest"].Agreement, c.pass)
			}
		})
	}
}

func TestReplayRealizedCost(t *testing.T) {
	root := t.TempDir()
	prompts := []map[string]any{promptRow(1, "one"), promptRow(2, "two")}
	decisions := []map[string]any{
		decisionRow(1, "cheapest", "", "m-cheap"), decisionRow(2, "cheapest", "", "m-cheap"),
		decisionRow(1, "strongest", "", "m-strong"), decisionRow(2, "strongest", "", "m-strong"),
	}
	outcomes := []map[string]any{
		outcomeRow(1, "m-cheap", true, 0.01), outcomeRow(2, "m-cheap", false, 0.02),
		outcomeRow(1, "m-strong", true, 0.10), outcomeRow(2, "m-strong", true, 0.20),
	}
	writeReplayFiles(t, root, prompts, decisions, outcomes)
	g := newStubGateway(t, 0, func(group, _ string) stubAnswer {
		if group == "cheapest" {
			return stubAnswer{model: "m-cheap"}
		}
		return stubAnswer{model: "m-strong"}
	})

	res, err := RunReplay(context.Background(), replayOptions(g, root, 1))
	if err != nil {
		t.Fatal(err)
	}

	cheap := res.Strategies["cheapest"]
	requireClose(t, "cheapest accuracy", cheap.Accuracy, 0.5)
	requireClose(t, "cheapest total cost", cheap.TotalCostUSD, 0.03)
	if cheap.CostPerSuccessUSD == nil {
		t.Fatal("cheapest cost per success is nil, want 0.03")
	}
	requireClose(t, "cheapest cost per success", *cheap.CostPerSuccessUSD, 0.03)

	strong := res.Strategies["strongest"]
	requireClose(t, "strongest accuracy", strong.Accuracy, 1)
	requireClose(t, "strongest total cost", strong.TotalCostUSD, 0.30)
	if strong.CostPerSuccessUSD == nil {
		t.Fatal("strongest cost per success is nil, want 0.15")
	}
	requireClose(t, "strongest cost per success", *strong.CostPerSuccessUSD, 0.15)
}

func TestReplayNoSuccessHasNoCostPerSuccess(t *testing.T) {
	root := t.TempDir()
	writeReplayFiles(t, root,
		[]map[string]any{promptRow(1, "one")},
		[]map[string]any{decisionRow(1, "cheapest", "", "m-cheap")},
		[]map[string]any{outcomeRow(1, "m-cheap", false, 0.02)})
	g := newStubGateway(t, 0, func(_, _ string) stubAnswer { return stubAnswer{model: "m-cheap"} })

	res, err := RunReplay(context.Background(), replayOptions(g, root, 1))
	if err != nil {
		t.Fatal(err)
	}
	s := res.Strategies["cheapest"]
	if s.CostPerSuccessUSD != nil {
		t.Errorf("cost per success = %v with no success, want nil (it is undefined)", *s.CostPerSuccessUSD)
	}
	if s.Accuracy != 0 {
		t.Errorf("accuracy = %v, want 0", s.Accuracy)
	}
	data, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"cost_per_success_usd":null`) {
		t.Errorf("result does not encode the undefined cost per success as null: %s", data)
	}
}

func TestReplayErrorsCountAsMismatches(t *testing.T) {
	root, _ := threePrompts(t)
	g := newStubGateway(t, 0, func(group, prompt string) stubAnswer {
		switch {
		case group == "cheapest" && prompt == "PROMPT-TEXT-TWO":
			return stubAnswer{status: http.StatusTooManyRequests, code: "rate_limited"}
		case group == "cheapest" && prompt == "PROMPT-TEXT-THREE":
			return stubAnswer{status: http.StatusServiceUnavailable, code: "no_eligible_model"}
		case group == "cheapest":
			return stubAnswer{model: "m-cheap"}
		}
		return stubAnswer{model: "m-strong"}
	})

	res, err := RunReplay(context.Background(), replayOptions(g, root, 2))
	if err != nil {
		t.Fatal(err)
	}
	s := res.Strategies["cheapest"]
	if s.Agree != 1 || s.N != 3 {
		t.Errorf("agree = %d of %d, want 1 of 3", s.Agree, s.N)
	}
	if want := map[string]int{"rate_limited": 1, "no_eligible_model": 1}; fmt.Sprint(s.Errors) != fmt.Sprint(want) {
		t.Errorf("errors = %v, want %v", s.Errors, want)
	}
	want := []Mismatch{
		{PromptID: "p2", Offline: "m-cheap", Gateway: "rate_limited"},
		{PromptID: "p3", Offline: "m-cheap", Gateway: "no_eligible_model"},
	}
	if !slices.Equal(s.Mismatches, want) {
		t.Errorf("mismatches = %+v, want %+v", s.Mismatches, want)
	}
	// A request that failed has no served model: it is not a success and it
	// costs nothing.
	requireClose(t, "accuracy", s.Accuracy, 1.0/3)
	requireClose(t, "total cost", s.TotalCostUSD, 0.01)

	if none := res.Strategies["strongest"].Errors; none == nil || len(none) != 0 {
		t.Errorf("strongest errors = %#v, want an empty, non-nil map", none)
	}
}

func TestReplayMismatchesAreCapped(t *testing.T) {
	root := t.TempDir()
	var prompts, decisions, outcomes []map[string]any
	for i := 0; i < 60; i++ {
		prompts = append(prompts, promptRow(i, fmt.Sprintf("text %d", i)))
		decisions = append(decisions, decisionRow(i, "cheapest", "", "m-a"))
		outcomes = append(outcomes, outcomeRow(i, "m-a", true, 0.01), outcomeRow(i, "m-b", true, 0.01))
	}
	writeReplayFiles(t, root, prompts, decisions, outcomes)
	g := newStubGateway(t, 0, func(_, _ string) stubAnswer { return stubAnswer{model: "m-b"} })

	res, err := RunReplay(context.Background(), replayOptions(g, root, 8))
	if err != nil {
		t.Fatal(err)
	}
	s := res.Strategies["cheapest"]
	if s.Agree != 0 || s.N != 60 {
		t.Errorf("agree = %d of %d, want 0 of 60", s.Agree, s.N)
	}
	if len(s.Mismatches) != 50 {
		t.Fatalf("listed %d mismatches, want 50", len(s.Mismatches))
	}
	// The listed ones are the first in prompt order, whatever order the
	// workers finished in.
	for i, m := range s.Mismatches {
		if m.PromptID != promptID(i) {
			t.Fatalf("mismatch %d is %s, want %s", i, m.PromptID, promptID(i))
		}
	}
}

// allGroups is a dataset with every strategy the trainer writes, for the
// tests of what is sent and in which order.
func allGroups(t *testing.T, nPrompts int) string {
	t.Helper()
	root := t.TempDir()
	var prompts, decisions, outcomes []map[string]any
	for i := 0; i < nPrompts; i++ {
		prompts = append(prompts, promptRow(i, fmt.Sprintf("text %d", i)))
		for _, profile := range []string{"strict", "balanced", "economy"} {
			decisions = append(decisions, decisionRow(i, "quality", profile, "m-a"))
		}
		for _, s := range []string{"strongest", "cheapest", "static", "latency"} {
			decisions = append(decisions, decisionRow(i, s, "", "m-a"))
		}
		outcomes = append(outcomes, outcomeRow(i, "m-a", true, 0.01))
	}
	writeReplayFiles(t, root, prompts, decisions, outcomes)
	return root
}

func TestReplayRunsOneStrategyAtATimeInOrder(t *testing.T) {
	const nPrompts = 12
	root := allGroups(t, nPrompts)
	g := newStubGateway(t, time.Millisecond, func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} })

	res, err := RunReplay(context.Background(), replayOptions(g, root, 4))
	if err != nil {
		t.Fatal(err)
	}
	if res.Requests != 7*nPrompts {
		t.Errorf("requests = %d, want %d", res.Requests, 7*nPrompts)
	}

	// Latency first, on a gateway no other strategy has touched, then the
	// rest; every group is one unbroken run of requests.
	want := []string{"latency", "strongest", "cheapest", "static", "quality/strict", "quality/balanced", "quality/economy"}
	var runs []string
	for _, group := range g.order {
		if len(runs) == 0 || runs[len(runs)-1] != group {
			runs = append(runs, group)
		}
	}
	if !slices.Equal(runs, want) {
		t.Errorf("groups ran as %v, want %v", runs, want)
	}
	for _, name := range want {
		if n := res.Strategies[name].N; n != nPrompts {
			t.Errorf("%s sent %d requests, want %d", name, n, nPrompts)
		}
	}
	requireClose(t, "overall agreement", res.OverallAgreement, 1)
	if !res.ParityPass {
		t.Error("parity_pass is false with full agreement")
	}
}

func TestReplaySendsRoutingHeaders(t *testing.T) {
	root := allGroups(t, 1)
	g := newStubGateway(t, 0, func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} })
	if _, err := RunReplay(context.Background(), replayOptions(g, root, 1)); err != nil {
		t.Fatal(err)
	}

	got := map[string][2]string{}
	for _, h := range g.headers {
		strategy, profile := h.Get("X-RouteForge-Strategy"), h.Get("X-RouteForge-Profile")
		got[strategy+"/"+profile] = [2]string{strategy, profile}
		if h.Get("Authorization") != "Bearer "+testKey {
			t.Errorf("Authorization = %q", h.Get("Authorization"))
		}
	}
	want := []string{"latency/", "strongest/", "cheapest/", "static/", "quality/strict", "quality/balanced", "quality/economy"}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Errorf("no request with strategy/profile %q; sent %v", k, got)
		}
	}
	if len(got) != len(want) {
		t.Errorf("sent %d distinct strategy/profile pairs, want %d: %v", len(got), len(want), got)
	}
}

func TestReplayHonorsWorkerLimit(t *testing.T) {
	root := t.TempDir()
	var prompts, decisions, outcomes []map[string]any
	for i := 0; i < 40; i++ {
		prompts = append(prompts, promptRow(i, fmt.Sprintf("text %d", i)))
		decisions = append(decisions, decisionRow(i, "cheapest", "", "m-a"))
		outcomes = append(outcomes, outcomeRow(i, "m-a", true, 0.01))
	}
	writeReplayFiles(t, root, prompts, decisions, outcomes)
	g := newStubGateway(t, 10*time.Millisecond, func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} })

	if _, err := RunReplay(context.Background(), replayOptions(g, root, 4)); err != nil {
		t.Fatal(err)
	}
	if g.maxInflight > 4 {
		t.Errorf("up to %d requests in flight with 4 workers", g.maxInflight)
	}
	if g.maxInflight < 2 {
		t.Errorf("at most %d request in flight with 4 workers, want concurrency", g.maxInflight)
	}
}

func TestReplayTransportErrorStopsTheRun(t *testing.T) {
	root, answer := threePrompts(t)
	g := newStubGateway(t, 0, answer)
	opts := replayOptions(g, root, 2)
	g.Close() // the gateway is gone

	start := time.Now()
	_, err := RunReplay(context.Background(), opts)
	if err == nil {
		t.Fatal("want an error when the gateway cannot be reached")
	}
	if time.Since(start) > 10*time.Second {
		t.Errorf("took %s to give up", time.Since(start))
	}
	if strings.Contains(err.Error(), "PROMPT-TEXT") {
		t.Errorf("error %q repeats a prompt", err)
	}
}

func TestReplayContextCanceled(t *testing.T) {
	root, answer := threePrompts(t)
	g := newStubGateway(t, 200*time.Millisecond, answer)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if _, err := RunReplay(ctx, replayOptions(g, root, 1)); err == nil {
		t.Fatal("want an error when the context ends mid-run")
	}
}

func TestReplayRejectsBadInput(t *testing.T) {
	good := func() (prompts, decisions, outcomes []map[string]any) {
		return []map[string]any{promptRow(1, "one")},
			[]map[string]any{decisionRow(1, "cheapest", "", "m-a")},
			[]map[string]any{outcomeRow(1, "m-a", true, 0.01)}
	}
	serveA := func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} }

	for _, c := range []struct {
		name    string
		workers int
		mutate  func(prompts, decisions, outcomes *[]map[string]any)
		answer  func(group, prompt string) stubAnswer
		want    string
	}{
		{"no workers", 0, func(_, _, _ *[]map[string]any) {}, serveA, "workers"},
		{"unknown strategy", 1, func(_, d, _ *[]map[string]any) {
			*d = []map[string]any{decisionRow(1, "random", "", "m-a")}
		}, serveA, "random"},
		{"quality without profile", 1, func(_, d, _ *[]map[string]any) {
			*d = []map[string]any{decisionRow(1, "quality", "", "m-a")}
		}, serveA, "quality"},
		{"profile on a strategy without one", 1, func(_, d, _ *[]map[string]any) {
			*d = []map[string]any{decisionRow(1, "cheapest", "strict", "m-a")}
		}, serveA, "cheapest"},
		{"decision for an unknown prompt", 1, func(_, d, _ *[]map[string]any) {
			*d = []map[string]any{decisionRow(7, "cheapest", "", "m-a")}
		}, serveA, "p7"},
		{"no decisions", 1, func(_, d, _ *[]map[string]any) { *d = nil }, serveA, "decisions"},
		{"duplicate prompt id", 1, func(p, _, _ *[]map[string]any) {
			*p = append(*p, promptRow(1, "again"))
		}, serveA, "p1"},
		{"served model has no recorded outcome", 1, func(_, _, _ *[]map[string]any) {}, func(_, _ string) stubAnswer {
			return stubAnswer{model: "m-unrecorded"}
		}, "m-unrecorded"},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := t.TempDir()
			p, d, o := good()
			c.mutate(&p, &d, &o)
			writeReplayFiles(t, root, p, d, o)
			g := newStubGateway(t, 0, c.answer)

			_, err := RunReplay(context.Background(), replayOptions(g, root, c.workers))
			if err == nil {
				t.Fatal("want an error")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
}

func TestReplayMissingFiles(t *testing.T) {
	g := newStubGateway(t, 0, func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} })
	for _, missing := range []string{"prompts.jsonl", "decisions.jsonl", "test.jsonl"} {
		t.Run(missing, func(t *testing.T) {
			root := t.TempDir()
			writeReplayFiles(t, root,
				[]map[string]any{promptRow(1, "one")},
				[]map[string]any{decisionRow(1, "cheapest", "", "m-a")},
				[]map[string]any{outcomeRow(1, "m-a", true, 0.01)})
			if err := os.Remove(filepath.Join(root, "trainer", "artifacts", "replay", missing)); err != nil {
				t.Fatal(err)
			}
			_, err := RunReplay(context.Background(), replayOptions(g, root, 1))
			if err == nil || !strings.Contains(err.Error(), missing) {
				t.Fatalf("err = %v, want one that names %s", err, missing)
			}
		})
	}
}

func TestReplayMalformedLine(t *testing.T) {
	root := t.TempDir()
	writeReplayFiles(t, root,
		[]map[string]any{promptRow(1, "one")},
		[]map[string]any{decisionRow(1, "cheapest", "", "m-a")},
		[]map[string]any{outcomeRow(1, "m-a", true, 0.01)})
	path := filepath.Join(root, "trainer", "artifacts", "replay", "decisions.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("{\"prompt_id\": SECRET-PROMPT-TEXT\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	g := newStubGateway(t, 0, func(_, _ string) stubAnswer { return stubAnswer{model: "m-a"} })

	_, err = RunReplay(context.Background(), replayOptions(g, root, 1))
	if err == nil {
		t.Fatal("want an error for a line that is not JSON")
	}
	if !strings.Contains(err.Error(), "decisions.jsonl") {
		t.Errorf("error %q does not name the file", err)
	}
}

func TestForEachRunsEveryIndexOnce(t *testing.T) {
	const n = 500
	var seen [n]atomic.Int32
	if err := forEach(context.Background(), n, 8, func(_ context.Context, i int) error {
		seen[i].Add(1)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for i := range seen {
		if seen[i].Load() != 1 {
			t.Fatalf("index %d ran %d times", i, seen[i].Load())
		}
	}
}

func TestForEachStopsAtTheFirstError(t *testing.T) {
	boom := errors.New("boom")
	var ran atomic.Int32
	err := forEach(context.Background(), 10000, 4, func(ctx context.Context, i int) error {
		ran.Add(1)
		if i == 3 {
			return boom
		}
		select {
		case <-ctx.Done():
		case <-time.After(time.Millisecond):
		}
		return nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the failure of index 3", err)
	}
	if ran.Load() > 1000 {
		t.Errorf("ran %d indexes after a failure, want it to stop early", ran.Load())
	}
}

func TestForEachEmpty(t *testing.T) {
	if err := forEach(context.Background(), 0, 4, func(context.Context, int) error {
		t.Error("fn called for no work")
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
