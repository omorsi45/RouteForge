package policy

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestCalibrationLookupBoundaries(t *testing.T) {
	c := Calibration{Blocks: []Block{
		{Lo: 0, Hi: 0.3},
		{Lo: 0.3, Hi: 0.7},
		{Lo: 0.7, Hi: 1},
	}}
	cases := []struct {
		raw  float64
		want int
	}{
		{0.3, 1},
		{0.2999, 0},
		{0.7, 2},
		{0.6999, 1},
		{1.0, 2},
		{0.0, 0},
		{-0.1, 0},
		{0.5, 1},
	}
	for _, tc := range cases {
		if got := c.Lookup(tc.raw); got != tc.want {
			t.Errorf("Lookup(%v) = %d, want %d", tc.raw, got, tc.want)
		}
	}
}

func TestCalibrationLookupSingleBlock(t *testing.T) {
	c := Calibration{Blocks: []Block{{Lo: 0, Hi: 1}}}
	for _, raw := range []float64{-1, 0, 0.5, 1, 2} {
		if got := c.Lookup(raw); got != 0 {
			t.Errorf("Lookup(%v) = %d, want 0", raw, got)
		}
	}
}

// TestCalibrationLookupMatchesLinearScan checks the binary search against the
// definition: the index of the last block with Lo <= raw, 0 below the first.
func TestCalibrationLookupMatchesLinearScan(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	los := []float64{0}
	for i := 1; i < 50; i++ {
		los = append(los, rng.Float64())
	}
	slices.Sort(los)
	los = slices.Compact(los)
	blocks := make([]Block, len(los))
	for i, lo := range los {
		blocks[i].Lo = lo
	}
	c := Calibration{Blocks: blocks}
	linear := func(raw float64) int {
		idx := 0
		for i, b := range blocks {
			if b.Lo <= raw {
				idx = i
			}
		}
		return idx
	}
	probes := []float64{-0.5, 0, 1}
	for _, lo := range los {
		probes = append(probes, lo, lo-1e-12, lo+1e-12)
	}
	for i := 0; i < 500; i++ {
		probes = append(probes, rng.Float64())
	}
	for _, raw := range probes {
		if got, want := c.Lookup(raw), linear(raw); got != want {
			t.Fatalf("Lookup(%v) = %d, linear scan gives %d", raw, got, want)
		}
	}
}

func TestStaticClassify(t *testing.T) {
	rules := StaticRules{
		FamilyKeywords: map[string][]string{
			"reasoning": {"prove"},
			"code":      {"python", "function"},
		},
		FamilyModel:   map[string]string{"reasoning": "a", "code": "b"},
		FamilyOrder:   []string{"reasoning", "code"},
		DefaultFamily: "reasoning",
	}
	cases := []struct {
		name string
		text string
		want string
	}{
		{"two code keywords beat one reasoning keyword", "prove python function", "code"},
		{"no keyword gives the default", "hello", "reasoning"},
		{"empty text gives the default", "", "reasoning"},
		{"one keyword", "write a python script", "code"},
		{"case is folded", "PROVE it", "reasoning"},
		{"repeats count once", "python python python prove", "reasoning"},
		{"a tie goes to the earlier family", "prove python", "reasoning"},
		{"punctuation separates tokens", "(python)/function!", "code"},
		{"a keyword inside a longer token does not match", "pythonic", "reasoning"},
		{"non-ASCII bytes separate tokens", "prove\u00e9python function", "code"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := rules.Classify(tc.text); got != tc.want {
				t.Errorf("Classify(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestStaticClassifyDefaultsWhenNoFamilyWins(t *testing.T) {
	rules := StaticRules{
		FamilyKeywords: map[string][]string{"a": {"x"}, "b": {"y"}},
		FamilyOrder:    []string{"a", "b"},
		DefaultFamily:  "b",
	}
	if got := rules.Classify("nothing here"); got != "b" {
		t.Errorf("Classify() = %q, want the default b", got)
	}
	// With the default listed last, a single match in an earlier family wins.
	if got := rules.Classify("x"); got != "a" {
		t.Errorf("Classify(x) = %q, want a", got)
	}
}

func TestLoadCommittedPolicy(t *testing.T) {
	data := readPolicy(t)
	doc, err := ParseDocument(data)
	if err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
	p, err := Load(data)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if p.Version != doc.Version || p.CreatedAt != doc.CreatedAt || p.PredictorType != PredictorGBM {
		t.Errorf("Version, CreatedAt, PredictorType = %q, %q, %q; want %q, %q, %q",
			p.Version, p.CreatedAt, p.PredictorType, doc.Version, doc.CreatedAt, PredictorGBM)
	}
	if p.Strongest != doc.StrongestModel || p.Cheapest != doc.CheapestModel {
		t.Errorf("Strongest, Cheapest = %q, %q; want %q, %q", p.Strongest, p.Cheapest, doc.StrongestModel, doc.CheapestModel)
	}
	if len(p.Models) != len(poolOrder) {
		t.Fatalf("got %d models, want %d", len(p.Models), len(poolOrder))
	}
	for i, m := range p.Models {
		if m.Name != poolOrder[i] {
			t.Errorf("Models[%d].Name = %q, want %q", i, m.Name, poolOrder[i])
		}
		if m.ExpectedOutputTokens != doc.Models[i].ExpectedOutputTokens {
			t.Errorf("Models[%d].ExpectedOutputTokens = %v, want %v", i, m.ExpectedOutputTokens, doc.Models[i].ExpectedOutputTokens)
		}
		if len(m.Calibration.Blocks) != len(doc.Models[i].Calibration.Blocks) {
			t.Errorf("Models[%d] has %d blocks, want %d", i, len(m.Calibration.Blocks), len(doc.Models[i].Calibration.Blocks))
		}
		gbm, ok := m.Predictor.(*GBM)
		if !ok {
			t.Fatalf("Models[%d].Predictor is %T, want *GBM", i, m.Predictor)
		}
		want, err := decodeGBM(doc.Models[i].Predictor, "predictor")
		if err != nil {
			t.Fatalf("decodeGBM() error: %v", err)
		}
		if len(gbm.Trees) != len(want.Trees) || gbm.Init != want.Init || gbm.LearningRate != want.LearningRate {
			t.Errorf("Models[%d] GBM = init %v rate %v %d trees; want %v %v %d",
				i, gbm.Init, gbm.LearningRate, len(gbm.Trees), want.Init, want.LearningRate, len(want.Trees))
		}
	}
	if len(p.Profiles) != len(ProfileAlphas) {
		t.Fatalf("got %d profiles, want %d", len(p.Profiles), len(ProfileAlphas))
	}
	for _, d := range doc.Profiles {
		want := Profile{Name: d.Name, Alpha: d.Alpha, Delta: d.Delta, Lambda: d.Lambda, CertN: d.CertN, CertRegret: d.CertRegret, RegretUCB: d.RegretUCB}
		if got := p.Profiles[d.Name]; got != want {
			t.Errorf("Profiles[%q] = %+v, want %+v", d.Name, got, want)
		}
	}
	if !slices.Equal(p.Rules.FamilyOrder, doc.StaticRules.FamilyOrder) || p.Rules.DefaultFamily != doc.StaticRules.DefaultFamily {
		t.Errorf("Rules order, default = %v, %q; want %v, %q",
			p.Rules.FamilyOrder, p.Rules.DefaultFamily, doc.StaticRules.FamilyOrder, doc.StaticRules.DefaultFamily)
	}
	for i := range DenseNames {
		if p.DenseMean[i] != doc.FeatureSpec.DenseMean[i] || p.DenseStd[i] != doc.FeatureSpec.DenseStd[i] {
			t.Errorf("dense scaler %d = %v, %v; want %v, %v", i, p.DenseMean[i], p.DenseStd[i], doc.FeatureSpec.DenseMean[i], doc.FeatureSpec.DenseStd[i])
		}
	}
}

func TestLoadKeepsExactBytes(t *testing.T) {
	data := readPolicy(t)
	input := bytes.Clone(data)
	p, err := Load(input)
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if !bytes.Equal(p.Raw, data) {
		t.Errorf("Raw differs from the artifact bytes (%d vs %d bytes)", len(p.Raw), len(data))
	}
	// Raw is a copy: changing the caller's buffer must not change the policy.
	input[0] = 'X'
	if !bytes.Equal(p.Raw, data) {
		t.Error("Raw aliases the input buffer")
	}
}

func TestLoadLRPolicy(t *testing.T) {
	m := smallPolicyMap(t)
	asLR(m)
	p, err := Load(render(t, m, false))
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if p.PredictorType != PredictorLR {
		t.Fatalf("PredictorType = %q, want %q", p.PredictorType, PredictorLR)
	}
	lr, ok := p.Models[0].Predictor.(*LR)
	if !ok {
		t.Fatalf("Predictor is %T, want *LR", p.Models[0].Predictor)
	}
	if lr.Intercept != 0.25 || !slices.Equal(lr.SparseIndices, []int32{3, 17, 262143}) ||
		!slices.Equal(lr.SparseWeights, []float64{0.5, -0.5, 1e-05}) || lr.DenseWeights != [8]float64{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125} {
		t.Errorf("compiled LR = %+v", lr)
	}
	// Hand check: dense 0.125*1 + intercept 0.25 = 0.375; sparse 3 and 17
	// match: 0.5*0.5 + 0.25*-0.5 = 0.125; z = 0.5.
	f := &Features{Indices: []int32{3, 17, 100}, Values: []float64{0.5, 0.25, 0.75}}
	denseZ := [8]float64{1}
	if got, want := lr.Raw(f, &denseZ), logistic1(0.5); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestLoadRejectsInvalidDocument(t *testing.T) {
	m := smallPolicyMap(t)
	setAt(t, m, "nobody", "strongest_model")
	_, err := Load(render(t, m, false))
	ve := requireValidationError(t, err)
	if ve.Path != "strongest_model" {
		t.Errorf("error path = %q, want strongest_model", ve.Path)
	}
	if _, err := Load([]byte("not json")); err == nil {
		t.Error("Load(not json) error = nil, want an error")
	}
}

func TestLoadFile(t *testing.T) {
	p, err := LoadFile(policyFixture)
	if err != nil {
		t.Fatalf("LoadFile() error: %v", err)
	}
	if want := loadParityRows(t)[0].PolicyVersion; p.Version != want {
		t.Errorf("Version = %q, want %q", p.Version, want)
	}
}

func TestLoadFileErrors(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.json")
	if _, err := LoadFile(missing); err == nil {
		t.Error("LoadFile(missing) error = nil, want an error")
	} else if !strings.Contains(err.Error(), missing) {
		t.Errorf("error %q does not name the file %q", err, missing)
	}

	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte("{}"), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	_, err := LoadFile(bad)
	if err == nil {
		t.Fatal("LoadFile(invalid) error = nil, want an error")
	}
	requireValidationError(t, err)
	if !strings.Contains(err.Error(), bad) {
		t.Errorf("error %q does not name the file %q", err, bad)
	}
}

// recordingPredictor returns a fixed raw score and remembers its inputs.
type recordingPredictor struct {
	raw    float64
	gotZ   [8]float64
	called int
}

func (r *recordingPredictor) Raw(_ *Features, denseZ *[8]float64) float64 {
	r.called++
	r.gotZ = *denseZ
	return r.raw
}

func TestScoreInPoolOrder(t *testing.T) {
	pa, pb := &recordingPredictor{raw: 0.8}, &recordingPredictor{raw: 0.1}
	p := &Policy{
		Models: []Model{
			{Name: "b-model", Predictor: pa, Calibration: Calibration{Blocks: []Block{
				{Lo: 0, Hi: 0.5, N: 10, K: 4, P: 0.4, LCB: 0.2},
				{Lo: 0.5, Hi: 1, N: 20, K: 15, P: 0.75, LCB: 0.6},
			}}},
			{Name: "a-model", Predictor: pb, Calibration: Calibration{Blocks: []Block{
				{Lo: 0, Hi: 1, N: 8, K: 2, P: 0.25, LCB: 0.1},
			}}},
		},
		DenseMean: [8]float64{1, 2, 3, 4, 5, 6, 7, 8},
		DenseStd:  [8]float64{2, 2, 2, 2, 4, 4, 4, 4},
	}
	f := Featurize("hello world")
	f.DenseRaw = [8]float64{3, 4, 5, 6, 9, 10, 11, 12}

	got := p.Score(&f)
	want := []Score{
		{Model: "b-model", Raw: 0.8, P: 0.75, LCB: 0.6, Block: 1, BlockN: 20, BlockK: 15},
		{Model: "a-model", Raw: 0.1, P: 0.25, LCB: 0.1, Block: 0, BlockN: 8, BlockK: 2},
	}
	if !slices.Equal(got, want) {
		t.Errorf("Score() = %+v, want %+v", got, want)
	}
	// denseZ = (raw - mean) / std: 1 1 1 1 1 1 1 1.
	if pa.gotZ != [8]float64{1, 1, 1, 1, 1, 1, 1, 1} || pb.gotZ != pa.gotZ {
		t.Errorf("predictors saw denseZ %v and %v, want all ones", pa.gotZ, pb.gotZ)
	}
	if pa.called != 1 || pb.called != 1 {
		t.Errorf("predictors called %d and %d times, want once each", pa.called, pb.called)
	}
}

func TestModelIndex(t *testing.T) {
	p := &Policy{Models: []Model{{Name: "m0"}, {Name: "m1"}, {Name: "m2"}}}
	for i, name := range []string{"m0", "m1", "m2"} {
		if got, ok := p.ModelIndex(name); !ok || got != i {
			t.Errorf("ModelIndex(%q) = %d, %v; want %d, true", name, got, ok, i)
		}
	}
	if got, ok := p.ModelIndex("nope"); ok || got != -1 {
		t.Errorf("ModelIndex(nope) = %d, %v; want -1, false", got, ok)
	}
}
