package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// emDash is built from its code point so that no file of the repository holds
// the character itself.
const emDash = "\u2014"

// The figures of the stub gateway are synthetic. None of them was measured.
type demoStubModel struct {
	name   string
	p, lcb float64
	n, k   int
	cost   float64
}

var demoStubModels = []demoStubModel{
	{"model-a", 0.40, 0.2907, 50, 20, 0.000012},
	{"model-b", 0.83, 0.7431, 100, 83, 0.00025},
	{"model-c", 0.95, 0.9119, 200, 190, 0.0015},
}

// demoStubChoice is the model the stub explains for a prompt under a profile.
var demoStubChoice = map[string]map[string]string{
	"prompt one": {"strict": "model-c", "balanced": "model-b", "economy": "model-a"},
	"prompt two": {"strict": "model-c", "balanced": "model-c", "economy": "model-b"},
}

var demoStubLambda = map[string]float64{"strict": 0.9, "balanced": 0.7, "economy": 0.5}

const demoStubPolicy = "pol-test-1" // the policy_version of testdata/report/evaluation.json

type demoCall struct{ path, prompt, profile, strategy, auth string }

// demoStub plays a gateway for the demo: it answers explain and chat requests
// and records them. The fields are read on every request, so a test sets them
// before it runs the demo.
type demoStub struct {
	mu    sync.Mutex
	calls []demoCall

	explainStatus int                // 0 answers 200
	chatStatus    int                // 0 answers 200
	dropBlockK    bool               // leave block_k out of the chosen candidate
	echoProfile   string             // answer with this profile instead of the requested one
	served        map[string]string  // prompt -> model that serves the chat request
	noCost        bool               // leave cost_usd out of the chat response
	versionOf     func(n int) string // policy_version of the n-th explain response (1-based)
}

// requests returns the requests received so far.
func (s *demoStub) requests() []demoCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]demoCall(nil), s.calls...)
}

func newDemoStub(t *testing.T) (*demoStub, *Client) {
	t.Helper()
	s := &demoStub{}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	return s, NewClient(srv.URL, testKey)
}

func (s *demoStub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.Messages) != 1 || body.Model != "routeforge-auto" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	call := demoCall{
		path:     r.URL.Path,
		prompt:   body.Messages[0].Content,
		profile:  r.Header.Get("X-RouteForge-Profile"),
		strategy: r.Header.Get("X-RouteForge-Strategy"),
		auth:     r.Header.Get("Authorization"),
	}
	s.mu.Lock()
	s.calls = append(s.calls, call)
	explains, chats := 0, 0
	for _, c := range s.calls {
		if c.path == "/v1/routing/explain" {
			explains++
		} else {
			chats++
		}
	}
	s.mu.Unlock()

	switch r.URL.Path {
	case "/v1/routing/explain":
		s.explain(w, call, explains)
	case "/v1/chat/completions":
		s.chat(w, call, chats)
	default:
		http.NotFound(w, r)
	}
}

func (s *demoStub) explain(w http.ResponseWriter, call demoCall, n int) {
	if s.explainStatus != 0 {
		w.WriteHeader(s.explainStatus)
		_, _ = fmt.Fprint(w, `{"error":{"code":"unauthorized","message":"no"}}`)
		return
	}
	chosen := demoStubChoice[call.prompt][call.profile]
	candidates := make([]map[string]any, 0, len(demoStubModels))
	for _, m := range demoStubModels {
		c := map[string]any{
			"model": m.name, "eligible": true, "p": m.p, "lcb": m.lcb,
			"block_n": m.n, "block_k": m.k, "expected_cost_usd": m.cost,
		}
		if s.dropBlockK && m.name == chosen {
			delete(c, "block_k")
		}
		candidates = append(candidates, c)
	}
	version := demoStubPolicy
	if s.versionOf != nil {
		version = s.versionOf(n)
	}
	profile := call.profile
	if s.echoProfile != "" {
		profile = s.echoProfile
	}
	writeStubJSON(w, map[string]any{
		"id": "dec_explain", "policy_version": version, "strategy": "quality", "profile": profile,
		"chosen": chosen, "reason": "cheapest_passing", "lambda": demoStubLambda[call.profile],
		"candidates": candidates,
	})
}

func (s *demoStub) chat(w http.ResponseWriter, call demoCall, n int) {
	if s.chatStatus != 0 {
		w.WriteHeader(s.chatStatus)
		_, _ = fmt.Fprint(w, `{"error":{"code":"rate_limited","message":"slow down"}}`)
		return
	}
	model := demoStubChoice[call.prompt][call.profile]
	if m, ok := s.served[call.prompt]; ok {
		model = m
	}
	id := fmt.Sprintf("dec_chat_%d", n)
	w.Header().Set("X-RouteForge-Model", model)
	w.Header().Set("X-RouteForge-Reason", "cheapest_passing")
	w.Header().Set("X-RouteForge-Decision-Id", id)
	ext := map[string]any{"decision_id": id}
	if !s.noCost {
		ext["cost_usd"] = 0.00031
	}
	writeStubJSON(w, map[string]any{"routeforge": ext})
}

func writeStubJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

var demoTestPrompts = []DemoPrompt{
	{Name: "alpha", Prompt: "prompt one"},
	{Name: "beta", Prompt: "prompt two"},
}

func runStubDemo(c *Client) (DemoResults, error) {
	return RunDemo(context.Background(), DemoOptions{Client: c, Prompts: demoTestPrompts})
}

func goldenEvaluation(t *testing.T) *Evaluation {
	t.Helper()
	ev, err := LoadDemoEvaluation(reportTestdata)
	if err != nil {
		t.Fatal(err)
	}
	return ev
}

func renderStubDemo(t *testing.T, c *Client) string {
	t.Helper()
	res, err := runStubDemo(c)
	if err != nil {
		t.Fatal(err)
	}
	out, err := RenderDemo(res, goldenEvaluation(t))
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestDemoPromptsFile(t *testing.T) {
	prompts, err := LoadDemoPrompts(filepath.Join(committedRoot, "examples", "prompts.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(prompts) != 8 {
		t.Fatalf("examples/prompts.json has %d prompts, want 8", len(prompts))
	}
	names := map[string]bool{}
	for _, p := range prompts {
		if p.Name == "" || p.Prompt == "" {
			t.Errorf("prompt %+v has an empty name or text", p.Name)
		}
		if names[p.Name] {
			t.Errorf("prompt name %q is not unique", p.Name)
		}
		names[p.Name] = true
	}
	if raw := readCommitted(t, "examples/prompts.json"); strings.Contains(string(raw), emDash) {
		t.Error("examples/prompts.json has an em dash")
	}
}

func TestLoadDemoPromptsRejects(t *testing.T) {
	for _, c := range []struct{ name, content, want string }{
		{"empty list", `[]`, "no prompts"},
		{"duplicate name", `[{"name":"a","prompt":"x"},{"name":"a","prompt":"y"}]`, "duplicate"},
		{"empty name", `[{"name":"","prompt":"x"}]`, "name"},
		{"empty prompt", `[{"name":"a","prompt":""}]`, "prompt"},
		{"unknown field", `[{"name":"a","prompt":"x","extra":1}]`, "unknown field"},
		{"not a list", `{"name":"a","prompt":"x"}`, "prompts.json"},
		{"trailing data", `[{"name":"a","prompt":"x"}] []`, "trailing"},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "prompts.json")
			if err := os.WriteFile(path, []byte(c.content), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := LoadDemoPrompts(path)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error = %v, want one mentioning %q", err, c.want)
			}
		})
	}
	if _, err := LoadDemoPrompts(filepath.Join(t.TempDir(), "absent.json")); err == nil {
		t.Error("a missing prompts file loaded without an error")
	}
}

func TestLoadDemoEvaluationMissing(t *testing.T) {
	_, err := LoadDemoEvaluation(t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "trainer/artifacts/reports/evaluation.json") {
		t.Fatalf("error = %v, want one naming trainer/artifacts/reports/evaluation.json", err)
	}
}

func TestRunDemoSendsEachPromptUnderEachProfile(t *testing.T) {
	s, c := newDemoStub(t)
	if _, err := runStubDemo(c); err != nil {
		t.Fatal(err)
	}

	type step struct{ path, prompt, profile string }
	const explain, chat = "/v1/routing/explain", "/v1/chat/completions"
	want := []step{
		{explain, "prompt one", "strict"}, {explain, "prompt one", "balanced"}, {explain, "prompt one", "economy"},
		{chat, "prompt one", "balanced"},
		{explain, "prompt two", "strict"}, {explain, "prompt two", "balanced"}, {explain, "prompt two", "economy"},
		{chat, "prompt two", "balanced"},
	}
	calls := s.requests()
	if len(calls) != len(want) {
		t.Fatalf("got %d requests, want %d: %+v", len(calls), len(want), calls)
	}
	for i, call := range calls {
		if got := (step{call.path, call.prompt, call.profile}); got != want[i] {
			t.Errorf("request %d = %+v, want %+v", i+1, got, want[i])
		}
		if call.strategy != "" {
			t.Errorf("request %d sets the strategy %q, want none", i+1, call.strategy)
		}
		if call.auth != "Bearer "+testKey {
			t.Errorf("request %d Authorization = %q", i+1, call.auth)
		}
	}
}

func TestRunDemoResults(t *testing.T) {
	_, c := newDemoStub(t)
	res, err := runStubDemo(c)
	if err != nil {
		t.Fatal(err)
	}
	if res.PolicyVersion != demoStubPolicy || len(res.Entries) != 2 {
		t.Fatalf("policy version %q with %d entries, want %q with 2", res.PolicyVersion, len(res.Entries), demoStubPolicy)
	}
	alpha := res.Entries[0]
	if alpha.Name != "alpha" || len(alpha.Decisions) != 3 {
		t.Fatalf("first entry = %q with %d decisions, want alpha with 3", alpha.Name, len(alpha.Decisions))
	}
	got := alpha.Decisions[1] // balanced
	want := DemoDecision{
		Profile: "balanced", Reason: "cheapest_passing", Lambda: 0.7,
		Chosen: DemoCandidate{Model: "model-b", P: 0.83, LCB: 0.7431, ExpectedCostUSD: 0.00025, BlockN: 100, BlockK: 83},
	}
	if got != want {
		t.Errorf("balanced decision = %+v, want %+v", got, want)
	}
	if alpha.Served.Model != "model-b" || alpha.Served.DecisionID != "dec_chat_1" || alpha.Served.Candidate != want.Chosen {
		t.Errorf("served = %+v", alpha.Served)
	}
	if alpha.Served.CostUSD == nil || *alpha.Served.CostUSD != 0.00031 {
		t.Errorf("served cost = %v, want 0.00031", alpha.Served.CostUSD)
	}
}

func TestRenderDemo(t *testing.T) {
	_, c := newDemoStub(t)
	out := renderStubDemo(t, c)

	if !strings.HasPrefix(out, "# Routing demo\n\n"+DemoNotice+"\n\n") {
		t.Errorf("the document does not open with the title and the notice:\n%.300s", out)
	}
	for _, want := range []string{
		"Policy version `pol-test-1`.",
		"| Prompt | Profile | Chosen | Reason | Lambda | p | LCB | Calibration (k of n) | Expected cost |",
		"| alpha | strict | model-c | cheapest_passing | 0.900 | 0.950 | 0.912 | 190 of 200 | $0.001500 |",
		"| alpha | balanced | model-b | cheapest_passing | 0.700 | 0.830 | 0.743 | 83 of 100 | $0.000250 |",
		"| alpha | economy | model-a | cheapest_passing | 0.500 | 0.400 | 0.291 | 20 of 50 | $0.000012 |",
		"| beta | strict | model-c | cheapest_passing | 0.900 | 0.950 | 0.912 | 190 of 200 | $0.001500 |",
		"| beta | balanced | model-c | cheapest_passing | 0.700 | 0.950 | 0.912 | 190 of 200 | $0.001500 |",
		"| beta | economy | model-b | cheapest_passing | 0.500 | 0.830 | 0.743 | 83 of 100 | $0.000250 |",
		"Across these 6 decisions the router chose 3 different models: model-c, model-b, model-a.",
		"| Prompt | Served by | Decision | Cost |",
		"| alpha | model-b | dec_chat_1 | $0.000310 |",
		"| beta | model-c | dec_chat_2 | $0.000310 |",
		"- `alpha`: on the calibration set, 83 of 100 requests scored like this succeeded on model-b; 95% lower bound 0.74\n",
		"- `beta`: on the calibration set, 190 of 200 requests scored like this succeeded on model-c; 95% lower bound 0.91\n",
		"| Strategy | Accuracy | Cost per success | Regret rate |",
		"| always_strongest | 0.800 | $0.1000 | 0.000 |",
		"| share_matched_random/balanced | 0.650 | $0.0900 | 0.150 |",
		"| static_oracle_family | 0.700 | $0.0750 | 0.100 |",
		"| quality/lr/balanced | 0.750 | $0.0800 | 0.040 |",
		"\n\nH1: falsified (static_oracle_family)\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the document does not contain %q", want)
		}
	}

	last := -1
	for _, heading := range []string{
		"\n## Routing decisions\n", "\n## Balanced requests through the gateway\n",
		"\n## Why each balanced choice\n", "\n## Offline baseline comparison\n",
	} {
		i := strings.Index(out, heading)
		if i < 0 {
			t.Errorf("the document has no section %q", strings.TrimSpace(heading))
		}
		if i < last {
			t.Errorf("section %q comes out of order", strings.TrimSpace(heading))
		}
		last = i
	}
	if !strings.HasSuffix(out, "\n") || strings.HasSuffix(out, "\n\n") || strings.Contains(out, "\r") {
		t.Errorf("want exactly one trailing newline and no carriage return, got tail %q", out[max(0, len(out)-20):])
	}
	if strings.Contains(out, emDash) {
		t.Error("the document has an em dash")
	}
	for _, p := range demoTestPrompts {
		if strings.Contains(out, p.Prompt) {
			t.Errorf("the document holds the text of prompt %q", p.Name)
		}
	}
}

func TestRenderDemoSupportedVerdict(t *testing.T) {
	_, c := newDemoStub(t)
	res, err := runStubDemo(c)
	if err != nil {
		t.Fatal(err)
	}
	ev := goldenEvaluation(t)
	ev.H1.Falsified, ev.H1.FalsifiedBy = false, nil
	out, err := RenderDemo(res, ev)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "\n\nH1: supported\n") {
		t.Errorf("the document has no H1 supported line:\n%s", out)
	}
}

// The sentence names the model that served the request, with the figures of
// that model's block, even when the router explained another model.
func TestRenderDemoNamesTheServedModel(t *testing.T) {
	s, c := newDemoStub(t)
	s.served = map[string]string{"prompt one": "model-c"}
	out := renderStubDemo(t, c)

	for _, want := range []string{
		"| alpha | balanced | model-b | ", // the explained choice stays in the decision table
		"| alpha | model-c | dec_chat_1 | $0.000310 |",
		"- `alpha`: on the calibration set, 190 of 200 requests scored like this succeeded on model-c; 95% lower bound 0.91\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the document does not contain %q", want)
		}
	}
}

func TestRenderDemoWithoutCost(t *testing.T) {
	s, c := newDemoStub(t)
	s.noCost = true
	out := renderStubDemo(t, c)
	if want := "| alpha | model-b | dec_chat_1 | n/a |"; !strings.Contains(out, want) {
		t.Errorf("the document does not contain %q", want)
	}
}

func TestRunDemoErrors(t *testing.T) {
	for _, c := range []struct {
		name  string
		setup func(s *demoStub)
		want  string
	}{
		{"explain refused", func(s *demoStub) { s.explainStatus = http.StatusUnauthorized }, "status 401"},
		{"chat refused", func(s *demoStub) { s.chatStatus = http.StatusTooManyRequests }, "status 429"},
		{"block_k missing", func(s *demoStub) { s.dropBlockK = true }, "block_k"},
		{"profile not echoed", func(s *demoStub) { s.echoProfile = "economy" }, "profile"},
		{"served model unknown", func(s *demoStub) { s.served = map[string]string{"prompt one": "model-z"} }, "model-z"},
		{"no policy", func(s *demoStub) { s.versionOf = func(int) string { return "" } }, "policy"},
		{"policy changes", func(s *demoStub) {
			s.versionOf = func(n int) string {
				if n == 4 {
					return "pol-other"
				}
				return demoStubPolicy
			}
		}, "pol-other"},
	} {
		t.Run(c.name, func(t *testing.T) {
			s, client := newDemoStub(t)
			c.setup(s)
			_, err := runStubDemo(client)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("error = %v, want one mentioning %q", err, c.want)
			}
			if strings.Contains(err.Error(), "prompt one") || strings.Contains(err.Error(), "prompt two") {
				t.Errorf("error %q holds prompt text", err)
			}
		})
	}
}

func TestRunDemoNeedsInputs(t *testing.T) {
	if _, err := RunDemo(context.Background(), DemoOptions{Prompts: demoTestPrompts}); err == nil {
		t.Error("RunDemo without a client returned no error")
	}
	_, c := newDemoStub(t)
	if _, err := RunDemo(context.Background(), DemoOptions{Client: c}); err == nil {
		t.Error("RunDemo without prompts returned no error")
	}
}

func TestRenderDemoErrors(t *testing.T) {
	_, c := newDemoStub(t)
	res, err := runStubDemo(c)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		res  DemoResults
		edit func(ev *Evaluation) *Evaluation
		want []string
	}{
		{"no evaluation", res, func(*Evaluation) *Evaluation { return nil }, []string{"evaluation.json"}},
		{"no entries", DemoResults{PolicyVersion: demoStubPolicy}, func(ev *Evaluation) *Evaluation { return ev }, []string{"no prompts"}},
		{"another policy", res, func(ev *Evaluation) *Evaluation { ev.PolicyVersion = "pol-other"; return ev },
			[]string{"pol-other", demoStubPolicy, "evaluation.json"}},
		{"strategy missing", res, func(ev *Evaluation) *Evaluation { delete(ev.Strategies, "static_oracle_family"); return ev },
			[]string{"static_oracle_family", "evaluation.json"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RenderDemo(tc.res, tc.edit(goldenEvaluation(t)))
			if err == nil {
				t.Fatal("RenderDemo returned no error")
			}
			for _, w := range tc.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
		})
	}
}

func TestCommittedDemoOutput(t *testing.T) {
	out := string(readCommitted(t, "examples/demo-output.md"))
	if !strings.HasPrefix(out, "# Routing demo\n\n"+DemoNotice+"\n") {
		t.Errorf("examples/demo-output.md does not open with the title and the notice")
	}
	if strings.Contains(out, emDash) {
		t.Error("examples/demo-output.md has an em dash")
	}
	if strings.Contains(out, "\r") {
		t.Error("examples/demo-output.md has a carriage return")
	}
}
