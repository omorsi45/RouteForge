package router_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/provider"
	"github.com/omorsi45/routeforge/internal/router"
)

// Paths of the trainer outputs the engine must agree with, relative to this
// package.
const (
	parityPolicyFile = "../../trainer/artifacts/policy.json"
	parityPriceFile  = "../../deploy/prices.yaml"
	parityRowsFile   = "../../testdata/parity/features.jsonl"
)

// parityRow is the part of a trainer parity row this package checks.
type parityRow struct {
	Name      string                    `json:"name"`
	Text      string                    `json:"text"`
	Scores    []parityScore             `json:"scores"`
	Decisions map[string]parityDecision `json:"decisions"`
}

type parityScore struct {
	Model string  `json:"model"`
	P     float64 `json:"p"`
	LCB   float64 `json:"lcb"`
}

type parityDecision struct {
	Chosen string `json:"chosen"`
	Reason string `json:"reason"`
}

func loadParityRows(t *testing.T) []parityRow {
	t.Helper()
	data, err := os.ReadFile(parityRowsFile)
	if err != nil {
		t.Fatalf("read parity fixture: %v", err)
	}
	var rows []parityRow
	for i, line := range bytes.Split(data, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var row parityRow
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatalf("parity fixture line %d: %v", i+1, err)
		}
		rows = append(rows, row)
	}
	if len(rows) != 20 {
		t.Fatalf("parity fixture has %d rows, want 20", len(rows))
	}
	return rows
}

// parityEngine builds an engine over the committed policy and price file with
// the six pool models configured under their pool names and no signals.
func parityEngine(t *testing.T) (*router.Engine, *policy.Policy) {
	t.Helper()
	pol, err := policy.LoadFile(parityPolicyFile)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	table, err := accounting.LoadPrices(parityPriceFile)
	if err != nil {
		t.Fatalf("LoadPrices: %v", err)
	}
	names := make([]string, len(pol.Models))
	models := make([]config.Model, len(pol.Models))
	prices := make(map[string]accounting.Price, len(pol.Models))
	for i, m := range pol.Models {
		names[i] = m.Name
		models[i] = config.Model{Name: m.Name, PolicyName: m.Name, Strongest: m.Name == pol.Strongest}
		price, ok := table.Price(m.Name)
		if !ok {
			t.Fatalf("no price for %s", m.Name)
		}
		prices[m.Name] = price
	}
	reg := policy.NewRegistry(names, nil)
	if err := reg.Add(pol); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := reg.Activate(pol.Version, "test"); err != nil {
		t.Fatalf("Activate: %v", err)
	}
	e := router.NewEngine(router.EngineOptions{
		Models:   models,
		Prices:   prices,
		Policies: reg,
		Now:      func() time.Time { return time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC) },
	})
	return e, pol
}

// TestParityDecisions replays the trainer's routing decision for every
// parity row and profile: the same model, the same reason, and the candidate
// p and lcb of the trainer's scores. The prompt reaches the engine both as
// the precomputed feature text and as a message.
func TestParityDecisions(t *testing.T) {
	e, pol := parityEngine(t)
	for _, row := range loadParityRows(t) {
		for _, profile := range []string{"strict", "balanced", "economy"} {
			want, ok := row.Decisions[profile]
			if !ok {
				t.Fatalf("%s: fixture has no %s decision", row.Name, profile)
			}
			inputs := map[string]*router.Request{
				"text": {Text: row.Text, PromptBytes: len(row.Text)},
				"messages": {Messages: []provider.Message{
					{Role: provider.RoleUser, Content: row.Text},
				}},
			}
			for kind, req := range inputs {
				t.Run(row.Name+"/"+profile+"/"+kind, func(t *testing.T) {
					req.Tenant = &config.Tenant{Name: "parity"}
					req.Model = config.AutoModel
					req.Strategy = router.StrategyQuality
					req.Profile = profile
					req.MaxTokens = 1024

					d, err := e.Decide(context.Background(), req)
					if err != nil {
						t.Fatalf("Decide() error: %v", err)
					}
					if d.Chosen != want.Chosen || d.Reason != want.Reason {
						t.Errorf("decision = %s (%s), trainer chose %s (%s)", d.Chosen, d.Reason, want.Chosen, want.Reason)
					}
					if d.PolicyVersion != pol.Version {
						t.Errorf("policy version = %q, want %q", d.PolicyVersion, pol.Version)
					}
					if lambda := pol.Profiles[profile].Lambda; d.Lambda == nil || *d.Lambda != lambda {
						t.Errorf("lambda = %v, want %v", d.Lambda, lambda)
					}
					if !d.GuaranteeApplies {
						t.Error("GuaranteeApplies = false with every pool model eligible")
					}
					if len(d.Candidates) != len(row.Scores) {
						t.Fatalf("got %d candidates, want %d", len(d.Candidates), len(row.Scores))
					}
					for i, s := range row.Scores {
						c := d.Candidates[i]
						if c.Model != s.Model {
							t.Fatalf("candidates[%d] = %s, want %s", i, c.Model, s.Model)
						}
						if c.P == nil || *c.P != s.P || c.LCB == nil || *c.LCB != s.LCB {
							t.Errorf("%s: p/lcb = %v/%v, want %v/%v", s.Model, c.P, c.LCB, s.P, s.LCB)
						}
					}
				})
			}
		}
	}
}

// TestParityExpectedCost pins expected_cost_usd to the trainer's, bit for bit.
// The want rows were printed by routeforge_train.routing.expected_cost_usd
// over trainer/artifacts/policy.json and deploy/prices.yaml, with float
// repr, for these prompt sizes in bytes.
func TestParityExpectedCost(t *testing.T) {
	e, pol := parityEngine(t)
	cases := []struct {
		bytes int
		want  [6]float64 // in pool order
	}{
		{0, [6]float64{0.01312, 0.00522, 0.0215, 0.001065, 0.000284, 0.00018779999999999998}},
		{5, [6]float64{0.0131225, 0.005226, 0.0215025, 0.0010655999999999999, 0.00028454, 0.00018798}},
		{32, [6]float64{0.01313, 0.005244, 0.02151, 0.0010674, 0.00028616000000000004, 0.00018852}},
		{39, [6]float64{0.0131325, 0.00525, 0.0215125, 0.001068, 0.0002867, 0.00018869999999999998}},
		{66, [6]float64{0.01314125, 0.005271, 0.02152125, 0.0010700999999999998, 0.00028858999999999997, 0.00018932999999999998}},
		{92, [6]float64{0.01314875, 0.005289, 0.02152875, 0.0010719000000000002, 0.00029021, 0.00018986999999999997}},
		{302, [6]float64{0.013215, 0.005448, 0.021595, 0.0010877999999999999, 0.00030451999999999996, 0.00019464}},
		{4097, [6]float64{0.01440125, 0.008295, 0.02278125, 0.0013725, 0.00056075, 0.00028004999999999995}},
	}
	for _, tc := range cases {
		req := &router.Request{
			Tenant:    &config.Tenant{Name: "parity"},
			Model:     config.AutoModel,
			Strategy:  router.StrategyQuality,
			Profile:   "balanced",
			Text:      strings.Repeat("a", tc.bytes),
			MaxTokens: 1,
		}
		d, err := e.Decide(context.Background(), req)
		if err != nil {
			t.Fatalf("%d bytes: Decide() error: %v", tc.bytes, err)
		}
		for i, c := range d.Candidates {
			if c.Model != pol.Models[i].Name {
				t.Fatalf("candidates[%d] = %s, want %s", i, c.Model, pol.Models[i].Name)
			}
			if c.ExpectedCostUSD == nil || *c.ExpectedCostUSD != tc.want[i] {
				t.Errorf("%d bytes, %s: expected cost = %v, trainer computed %v", tc.bytes, c.Model, c.ExpectedCostUSD, tc.want[i])
			}
		}
	}
}
