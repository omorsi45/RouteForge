package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

// parityFixture is the trainer-written parity file, relative to this package.
const parityFixture = "../../testdata/parity/features.jsonl"

// parityRows is the number of rows the trainer writes.
const parityRows = 20

// parityRow is one line of the parity fixture. The fixture is decoded
// strictly, so every key the trainer writes is declared here.
type parityRow struct {
	Name          string                `json:"name"`
	Messages      json.RawMessage       `json:"messages"`
	Text          string                `json:"text"`
	PromptSHA256  string                `json:"prompt_sha256"`
	HashedIndices []int32               `json:"hashed_indices"`
	HashedValues  []float64             `json:"hashed_values"`
	DenseRaw      []float64             `json:"dense_raw"`
	Dense         []float64             `json:"dense"`
	StaticFamily  string                `json:"static_family"`
	PolicyVersion string                `json:"policy_version"`
	Scores        []parityScore         `json:"scores"`
	Decisions     map[string]parityPick `json:"decisions"`
}

// parityScore is one model's expected score for a row.
type parityScore struct {
	Model string  `json:"model"`
	Raw   float64 `json:"raw"`
	P     float64 `json:"p"`
	LCB   float64 `json:"lcb"`
	Block int     `json:"block"`
}

// parityPick is the expected routing decision of one profile for a row.
type parityPick struct {
	Chosen string `json:"chosen"`
	Reason string `json:"reason"`
}

// loadParityRows reads every row of the parity fixture.
func loadParityRows(t *testing.T) []parityRow {
	t.Helper()
	data, err := os.ReadFile(parityFixture)
	if err != nil {
		t.Fatalf("read parity fixture: %v", err)
	}
	var rows []parityRow
	for i, line := range bytes.Split(data, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		dec := json.NewDecoder(bytes.NewReader(line))
		dec.DisallowUnknownFields()
		var row parityRow
		if err := dec.Decode(&row); err != nil {
			t.Fatalf("parity fixture line %d: %v", i+1, err)
		}
		rows = append(rows, row)
	}
	if len(rows) != parityRows {
		t.Fatalf("parity fixture has %d rows, want %d", len(rows), parityRows)
	}
	return rows
}

func TestParityFeatures(t *testing.T) {
	const tol = 1e-12
	for _, row := range loadParityRows(t) {
		t.Run(row.Name, func(t *testing.T) {
			sum := sha256.Sum256([]byte(row.Text))
			if got := hex.EncodeToString(sum[:]); got != row.PromptSHA256 {
				t.Errorf("sha256(text) = %s, want %s", got, row.PromptSHA256)
			}
			f := Featurize(row.Text)
			if f.Bytes != len(row.Text) {
				t.Errorf("Bytes = %d, want %d", f.Bytes, len(row.Text))
			}
			if len(f.Indices) != len(row.HashedIndices) || (len(f.Indices) > 0 && !reflect.DeepEqual(f.Indices, row.HashedIndices)) {
				t.Fatalf("Indices = %v, want %v", f.Indices, row.HashedIndices)
			}
			if len(f.Values) != len(row.HashedValues) {
				t.Fatalf("got %d values, want %d", len(f.Values), len(row.HashedValues))
			}
			for i, want := range row.HashedValues {
				if d := math.Abs(f.Values[i] - want); d > tol {
					t.Errorf("Values[%d] (index %d) = %v, want %v (diff %g)", i, f.Indices[i], f.Values[i], want, d)
				}
			}
			if len(row.DenseRaw) != len(f.DenseRaw) {
				t.Fatalf("fixture has %d dense features, want %d", len(row.DenseRaw), len(f.DenseRaw))
			}
			for i, want := range row.DenseRaw {
				if d := math.Abs(f.DenseRaw[i] - want); d > tol {
					t.Errorf("DenseRaw[%d] (%s) = %v, want %v (diff %g)", i, DenseNames[i], f.DenseRaw[i], want, d)
				}
			}
		})
	}
}

// TestParityScores scores every fixture row with the committed policy and
// compares with the trainer: raw scores within 1e-9, standardized dense
// features within 1e-12, and block index, p, lcb and the static family
// exactly.
func TestParityScores(t *testing.T) {
	const (
		rawTol   = 1e-9
		denseTol = 1e-12
	)
	p, err := LoadFile(policyFixture)
	if err != nil {
		t.Fatalf("LoadFile(%s) error: %v", policyFixture, err)
	}
	for _, row := range loadParityRows(t) {
		t.Run(row.Name, func(t *testing.T) {
			if p.Version != row.PolicyVersion {
				t.Fatalf("Version = %q, fixture row was made with %q", p.Version, row.PolicyVersion)
			}
			f := Featurize(row.Text)

			z := p.denseZ(&f)
			if len(row.Dense) != len(z) {
				t.Fatalf("fixture has %d standardized dense features, want %d", len(row.Dense), len(z))
			}
			for i, want := range row.Dense {
				if d := math.Abs(z[i] - want); d > denseTol {
					t.Errorf("denseZ[%d] (%s) = %v, want %v (diff %g)", i, DenseNames[i], z[i], want, d)
				}
			}

			scores := p.Score(&f)
			if len(scores) != len(row.Scores) {
				t.Fatalf("got %d scores, want %d", len(scores), len(row.Scores))
			}
			for i, want := range row.Scores {
				got := scores[i]
				if got.Model != want.Model {
					t.Fatalf("scores[%d].Model = %q, want %q", i, got.Model, want.Model)
				}
				if d := math.Abs(got.Raw - want.Raw); d > rawTol {
					t.Errorf("%s: Raw = %v, want %v (diff %g)", got.Model, got.Raw, want.Raw, d)
				}
				if got.Block != want.Block {
					t.Errorf("%s: Block = %d, want %d (raw %v)", got.Model, got.Block, want.Block, got.Raw)
				}
				if got.P != want.P {
					t.Errorf("%s: P = %v, want %v", got.Model, got.P, want.P)
				}
				if got.LCB != want.LCB {
					t.Errorf("%s: LCB = %v, want %v", got.Model, got.LCB, want.LCB)
				}
			}

			if got := p.Rules.Classify(row.Text); got != row.StaticFamily {
				t.Errorf("Classify() = %q, want %q", got, row.StaticFamily)
			}
		})
	}
}
