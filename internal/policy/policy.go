package policy

import (
	"bytes"
	"fmt"
	"os"
)

// Model is one pool model of a policy.
type Model struct {
	Name                 string
	Predictor            Predictor
	Calibration          Calibration
	ExpectedOutputTokens float64
}

// Profile is one routing profile with its certified threshold: the target
// regret rate Alpha, the confidence Delta, the Lambda that the trainer
// certified, and the certification evidence behind it.
type Profile struct {
	Name       string
	Alpha      float64
	Delta      float64
	Lambda     float64
	CertN      int
	CertRegret int
	RegretUCB  float64
}

// StaticRules is the trainer's keyword baseline, reproduced for comparison
// and for the static strategy.
type StaticRules struct {
	FamilyKeywords map[string][]string
	FamilyModel    map[string]string
	FamilyOrder    []string
	DefaultFamily  string
}

// Classify mirrors the trainer's KeywordRules.classify: it counts, per family
// in FamilyOrder, the keywords that are among the distinct tokens of text.
// The first family with a strictly greater count wins; with no keyword match
// the default family is returned.
func (r *StaticRules) Classify(text string) string {
	tokens := make(map[string]struct{})
	for _, tok := range Tokenize([]byte(text)) {
		tokens[string(tok)] = struct{}{}
	}
	best, bestCount := r.DefaultFamily, 0
	for _, family := range r.FamilyOrder {
		count := 0
		for _, keyword := range r.FamilyKeywords[family] {
			if _, ok := tokens[keyword]; ok {
				count++
			}
		}
		if count > bestCount {
			best, bestCount = family, count
		}
	}
	return best
}

// Policy is a compiled policy artifact. Models are in pool order, which is
// the tie-break order everywhere. A Policy is immutable after Load.
type Policy struct {
	Version       string
	CreatedAt     string
	PredictorType string
	Strongest     string
	Cheapest      string
	Models        []Model
	Profiles      map[string]Profile // by name: strict, balanced, economy
	Rules         StaticRules
	DenseMean     [8]float64
	DenseStd      [8]float64
	Raw           []byte // the exact artifact bytes, for persistence
}

// Load parses, validates and compiles a policy artifact. Any problem is a
// *ValidationError naming the JSON path. The policy keeps its own copy of
// data in Raw.
func Load(data []byte) (*Policy, error) {
	doc, err := ParseDocument(data)
	if err != nil {
		return nil, err
	}
	p := &Policy{
		Version:       doc.Version,
		CreatedAt:     doc.CreatedAt,
		PredictorType: doc.PredictorType,
		Strongest:     doc.StrongestModel,
		Cheapest:      doc.CheapestModel,
		Models:        make([]Model, len(doc.Models)),
		Profiles:      make(map[string]Profile, len(doc.Profiles)),
		Rules: StaticRules{
			FamilyKeywords: doc.StaticRules.FamilyKeywords,
			FamilyModel:    doc.StaticRules.FamilyModel,
			FamilyOrder:    doc.StaticRules.FamilyOrder,
			DefaultFamily:  doc.StaticRules.DefaultFamily,
		},
		Raw: bytes.Clone(data),
	}
	copy(p.DenseMean[:], doc.FeatureSpec.DenseMean)
	copy(p.DenseStd[:], doc.FeatureSpec.DenseStd)
	for i := range doc.Models {
		m := &doc.Models[i]
		predictor, err := compilePredictor(doc, m, fmt.Sprintf("models[%d].predictor", i))
		if err != nil {
			return nil, err
		}
		p.Models[i] = Model{
			Name:                 m.Name,
			Predictor:            predictor,
			Calibration:          newCalibration(&m.Calibration),
			ExpectedOutputTokens: m.ExpectedOutputTokens,
		}
	}
	for _, d := range doc.Profiles {
		p.Profiles[d.Name] = Profile{
			Name:       d.Name,
			Alpha:      d.Alpha,
			Delta:      d.Delta,
			Lambda:     d.Lambda,
			CertN:      d.CertN,
			CertRegret: d.CertRegret,
			RegretUCB:  d.RegretUCB,
		}
	}
	return p, nil
}

// LoadFile reads and loads the policy artifact at path. Errors name the file.
func LoadFile(path string) (*Policy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read policy: %w", err)
	}
	p, err := Load(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return p, nil
}

// Score is one model's score for a prompt.
type Score struct {
	Model  string
	Raw    float64 // predictor output
	P      float64 // calibrated success rate of the block
	LCB    float64 // Wilson lower bound of the block
	Block  int     // calibration block index
	BlockN int     // trials in the block
	BlockK int     // successes in the block
}

// denseZ standardizes the dense features: (raw - mean) / std.
func (p *Policy) denseZ(f *Features) [8]float64 {
	var z [8]float64
	for i := range z {
		z[i] = (f.DenseRaw[i] - p.DenseMean[i]) / p.DenseStd[i]
	}
	return z
}

// Score returns the raw score and calibration block of every model for the
// prompt with features f, in pool order. It mirrors the trainer's
// score_policy.
func (p *Policy) Score(f *Features) []Score {
	z := p.denseZ(f)
	scores := make([]Score, len(p.Models))
	for i := range p.Models {
		m := &p.Models[i]
		raw := m.Predictor.Raw(f, &z)
		idx := m.Calibration.Lookup(raw)
		b := &m.Calibration.Blocks[idx]
		scores[i] = Score{
			Model:  m.Name,
			Raw:    raw,
			P:      b.P,
			LCB:    b.LCB,
			Block:  idx,
			BlockN: b.N,
			BlockK: b.K,
		}
	}
	return scores
}

// ModelIndex returns the position of the named model in pool order, or
// -1 and false when the policy has no such model.
func (p *Policy) ModelIndex(name string) (int, bool) {
	for i := range p.Models {
		if p.Models[i].Name == name {
			return i, true
		}
	}
	return -1, false
}
