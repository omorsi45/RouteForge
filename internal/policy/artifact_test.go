package policy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
)

// policyFixture is the trained policy the trainer commits.
const policyFixture = "../../trainer/artifacts/policy.json"

// poolOrder is the model order of the committed policy: pool order, which is
// the tie-break everywhere.
var poolOrder = []string{
	"gpt-5",
	"claude-sonnet-4",
	"gemini-2.5-pro",
	"gemini-2.5-flash",
	"deepseek-v3.1-terminus",
	"qwen3-235b-a22b-2507",
}

func readPolicy(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile(policyFixture)
	if err != nil {
		t.Fatalf("read committed policy: %v", err)
	}
	return data
}

// policyMap decodes the committed policy into a fresh generic tree whose
// numbers keep their text, so a re-marshalled copy only differs where a test
// changed it.
func policyMap(t *testing.T) map[string]any {
	t.Helper()
	tree, err := decodeTree(readPolicy(t))
	if err != nil {
		t.Fatalf("decode committed policy: %v", err)
	}
	m, ok := tree.(map[string]any)
	if !ok {
		t.Fatalf("committed policy is %T, want an object", tree)
	}
	return m
}

// smallTrees is how many trees per model the small policy keeps.
const smallTrees = 3

// smallPolicy is the committed policy with only the first trees of each
// model. It is still a valid policy once re-rendered (the tree count is not
// part of the schema) and parses in a few milliseconds, which keeps the many
// mutation cases fast under the race detector.
var smallPolicy = sync.OnceValues(func() ([]byte, error) {
	data, err := os.ReadFile(policyFixture)
	if err != nil {
		return nil, err
	}
	tree, err := decodeTree(data)
	if err != nil {
		return nil, err
	}
	doc, ok := tree.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("committed policy is %T, want an object", tree)
	}
	models, _ := doc["models"].([]any)
	for _, raw := range models {
		model, _ := raw.(map[string]any)
		predictor, _ := model["predictor"].(map[string]any)
		trees, _ := predictor["trees"].([]any)
		if len(trees) < smallTrees {
			return nil, fmt.Errorf("committed policy has a model with %d trees, want at least %d", len(trees), smallTrees)
		}
		predictor["trees"] = trees[:smallTrees]
	}
	return json.Marshal(doc)
})

// smallPolicyMap decodes a fresh copy of the small policy.
func smallPolicyMap(t *testing.T) map[string]any {
	t.Helper()
	data, err := smallPolicy()
	if err != nil {
		t.Fatalf("build small policy: %v", err)
	}
	tree, err := decodeTree(data)
	if err != nil {
		t.Fatalf("decode small policy: %v", err)
	}
	m, ok := tree.(map[string]any)
	if !ok {
		t.Fatalf("small policy is %T, want an object", tree)
	}
	return m
}

func marshalMap(t *testing.T, m map[string]any) []byte {
	t.Helper()
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal mutated policy: %v", err)
	}
	return data
}

// render marshals m. Unless keepVersion is set it first recomputes the
// version, so the only problem left in the document is the one a test planted.
func render(t *testing.T, m map[string]any, keepVersion bool) []byte {
	t.Helper()
	if !keepVersion {
		m["version"] = "000000000000"
		version, err := CanonicalVersion(marshalMap(t, m))
		if err != nil {
			t.Fatalf("CanonicalVersion() of mutated policy: %v", err)
		}
		m["version"] = version
	}
	return marshalMap(t, m)
}

// at walks path (string keys into objects, int indexes into arrays) from root.
func at(t *testing.T, root any, path ...any) any {
	t.Helper()
	cur := root
	for i, step := range path {
		switch s := step.(type) {
		case string:
			obj, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("path %v: step %d is %T, want an object", path, i, cur)
			}
			next, ok := obj[s]
			if !ok {
				t.Fatalf("path %v: key %q is missing", path, s)
			}
			cur = next
		case int:
			arr, ok := cur.([]any)
			if !ok {
				t.Fatalf("path %v: step %d is %T, want an array", path, i, cur)
			}
			if s < 0 || s >= len(arr) {
				t.Fatalf("path %v: index %d out of range %d", path, s, len(arr))
			}
			cur = arr[s]
		default:
			t.Fatalf("path %v: unsupported step %T", path, step)
		}
	}
	return cur
}

func setAt(t *testing.T, root any, value any, path ...any) {
	t.Helper()
	parent := at(t, root, path[:len(path)-1]...)
	switch last := path[len(path)-1].(type) {
	case string:
		obj, ok := parent.(map[string]any)
		if !ok {
			t.Fatalf("path %v: parent is %T, want an object", path, parent)
		}
		obj[last] = value
	case int:
		arr, ok := parent.([]any)
		if !ok || last < 0 || last >= len(arr) {
			t.Fatalf("path %v: parent is %T, want an array holding index %d", path, parent, last)
		}
		arr[last] = value
	default:
		t.Fatalf("path %v: unsupported last step %T", path, last)
	}
}

func delAt(t *testing.T, root any, path ...any) {
	t.Helper()
	parent := at(t, root, path[:len(path)-1]...)
	key, _ := path[len(path)-1].(string)
	obj, ok := parent.(map[string]any)
	if !ok || key == "" {
		t.Fatalf("path %v: want a key of an object", path)
	}
	if _, ok := obj[key]; !ok {
		t.Fatalf("path %v: key %q is missing", path, key)
	}
	delete(obj, key)
}

func jn(s string) json.Number { return json.Number(s) }

// numbers returns n copies of the number text v.
func numbers(n int, v string) []any {
	out := make([]any, n)
	for i := range out {
		out[i] = jn(v)
	}
	return out
}

// asLR rewrites the committed (gbm) tree into an lr policy so the lr rules
// can be tested. The result is a valid policy.
func asLR(m map[string]any) {
	m["predictor_type"] = "lr"
	models, _ := m["models"].([]any)
	for _, raw := range models {
		model, _ := raw.(map[string]any)
		model["predictor"] = map[string]any{
			"intercept":      jn("0.25"),
			"sparse_indices": []any{jn("3"), jn("17"), jn("262143")},
			"sparse_weights": []any{jn("0.5"), jn("-0.5"), jn("1e-05")},
			"dense_weights":  numbers(8, "0.125"),
		}
	}
}

func requireValidationError(t *testing.T, err error) *ValidationError {
	t.Helper()
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("error = %v (%T), want *ValidationError", err, err)
	}
	return ve
}

func TestValidationErrorFormat(t *testing.T) {
	err := &ValidationError{Path: "models[0].calibration.blocks[1].lo", Problem: "must be 0.0"}
	if got, want := err.Error(), "models[0].calibration.blocks[1].lo: must be 0.0"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestPolicyConstants(t *testing.T) {
	if SchemaVersion != 1 || SentinelLambda != 1.01 || Delta != 0.05 {
		t.Errorf("SchemaVersion, SentinelLambda, Delta = %v, %v, %v; want 1, 1.01, 0.05", SchemaVersion, SentinelLambda, Delta)
	}
	want := map[string]float64{"strict": 0.02, "balanced": 0.05, "economy": 0.10}
	if len(ProfileAlphas) != len(want) {
		t.Fatalf("ProfileAlphas has %d entries, want %d", len(ProfileAlphas), len(want))
	}
	for i, name := range []string{"strict", "balanced", "economy"} {
		if ProfileAlphas[i].Name != name || ProfileAlphas[i].Alpha != want[name] {
			t.Errorf("ProfileAlphas[%d] = %+v, want {%s %v}", i, ProfileAlphas[i], name, want[name])
		}
	}
}

func TestParseCommittedPolicy(t *testing.T) {
	doc, err := ParseDocument(readPolicy(t))
	if err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
	if doc.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %d, want %d", doc.SchemaVersion, SchemaVersion)
	}
	if doc.PredictorType != PredictorGBM {
		t.Errorf("PredictorType = %q, want %q", doc.PredictorType, PredictorGBM)
	}
	if len(doc.Models) != len(poolOrder) {
		t.Fatalf("got %d models, want %d", len(doc.Models), len(poolOrder))
	}
	for i, m := range doc.Models {
		if m.Name != poolOrder[i] {
			t.Errorf("Models[%d].Name = %q, want %q", i, m.Name, poolOrder[i])
		}
		if len(m.Calibration.Blocks) == 0 {
			t.Errorf("Models[%d] has no calibration blocks", i)
		}
		if len(m.Predictor) == 0 {
			t.Errorf("Models[%d] has no predictor", i)
		}
	}
	if len(doc.Profiles) != 3 {
		t.Fatalf("got %d profiles, want 3", len(doc.Profiles))
	}
	for i, name := range []string{"strict", "balanced", "economy"} {
		if doc.Profiles[i].Name != name {
			t.Errorf("Profiles[%d].Name = %q, want %q", i, doc.Profiles[i].Name, name)
		}
	}
	if rows := loadParityRows(t); doc.Version != rows[0].PolicyVersion {
		t.Errorf("Version = %q, parity fixtures were made with %q", doc.Version, rows[0].PolicyVersion)
	}
}

func TestParseAcceptsLRPolicy(t *testing.T) {
	m := smallPolicyMap(t)
	asLR(m)
	doc, err := ParseDocument(render(t, m, false))
	if err != nil {
		t.Fatalf("ParseDocument() of an lr policy error: %v", err)
	}
	if doc.PredictorType != PredictorLR {
		t.Errorf("PredictorType = %q, want %q", doc.PredictorType, PredictorLR)
	}
}

func TestParseAcceptsLRWithoutSparseWeights(t *testing.T) {
	m := smallPolicyMap(t)
	asLR(m)
	for i := range poolOrder {
		setAt(t, m, []any{}, "models", i, "predictor", "sparse_indices")
		setAt(t, m, []any{}, "models", i, "predictor", "sparse_weights")
	}
	if _, err := ParseDocument(render(t, m, false)); err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
}

func TestParseAcceptsLambdaGridAndSentinel(t *testing.T) {
	for _, lambda := range []string{"1.01", "1.0", "0.0", "0.81", "0.01"} {
		t.Run(lambda, func(t *testing.T) {
			m := smallPolicyMap(t)
			setAt(t, m, jn(lambda), "profiles", 0, "lambda")
			if _, err := ParseDocument(render(t, m, false)); err != nil {
				t.Fatalf("ParseDocument() with lambda %s error: %v", lambda, err)
			}
		})
	}
}

func TestParseAcceptsTrailingWhitespace(t *testing.T) {
	data := append(readPolicy(t), " \n\t\n"...)
	if _, err := ParseDocument(data); err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
}

// rejectCase plants one problem in the committed policy and states where the
// validator must report it.
type rejectCase struct {
	name        string
	lr          bool                                 // rewrite to an lr policy before the change
	path        []any                                // location of the change
	value       any                                  // new value at path
	del         bool                                 // delete the key at path instead
	custom      func(t *testing.T, m map[string]any) // extra change, after path
	keepVersion bool                                 // leave the version as planted
	wantPath    string
	wantProblem string // substring of the problem, optional
}

func TestParseRejects(t *testing.T) {
	base := smallPolicyMap(t)
	blocks := at(t, base, "models", 0, "calibration", "blocks").([]any)
	lastBlock := len(blocks) - 1
	tree0 := at(t, base, "models", 0, "predictor", "trees", 0).(map[string]any)
	left := tree0["left"].([]any)
	if left[0] == jn("-1") {
		t.Fatalf("test needs tree 0 of model 0 to have a split at its root")
	}
	leaf := -1
	for i, v := range left {
		if v == jn("-1") {
			leaf = i
			break
		}
	}
	if leaf < 0 {
		t.Fatalf("test needs tree 0 of model 0 to have a leaf")
	}
	block1N, err := at(t, base, "models", 0, "calibration", "blocks", 1, "n").(json.Number).Int64()
	if err != nil {
		t.Fatalf("block 1 n: %v", err)
	}

	cal := func(i int, key string) []any { return []any{"models", 0, "calibration", "blocks", i, key} }
	tree := func(key string) []any { return []any{"models", 0, "predictor", "trees", 0, key} }
	pred := func(key string) []any { return []any{"models", 0, "predictor", key} }
	blockPath := func(i int, key string) string {
		return fmt.Sprintf("models[0].calibration.blocks[%d].%s", i, key)
	}
	treePath := "models[0].predictor.trees[0]"
	denseNames := []any{"log_tokens", "log_bytes", "digit_frac", "upper_frac", "code_fences", "lines", "question_marks", "max_token_len"}

	cases := []rejectCase{
		// Document level.
		{name: "schema version 2", path: []any{"schema_version"}, value: 2, wantPath: "schema_version"},
		{name: "schema version as float", path: []any{"schema_version"}, value: jn("1.0"), wantPath: "schema_version"},
		{name: "schema version missing", del: true, path: []any{"schema_version"}, wantPath: "schema_version"},
		{name: "unknown top-level key", path: []any{"surprise"}, value: 1, wantPath: "document", wantProblem: "unknown field"},
		{name: "tampered version", keepVersion: true, path: []any{"version"}, value: "000000000000", wantPath: "version", wantProblem: "does not match"},
		{name: "version not hex", keepVersion: true, path: []any{"version"}, value: "ZZZZZZZZZZZZ", wantPath: "version", wantProblem: "12 lowercase hex"},
		{name: "version uppercase", keepVersion: true, path: []any{"version"}, value: "7AE71D0E6DEB", wantPath: "version", wantProblem: "12 lowercase hex"},
		{name: "version too short", keepVersion: true, path: []any{"version"}, value: "7ae71d0e6de", wantPath: "version", wantProblem: "12 lowercase hex"},
		{name: "created_at empty", path: []any{"created_at"}, value: "", wantPath: "created_at"},
		{name: "source key missing", del: true, path: []any{"source", "git_sha"}, wantPath: "source.git_sha", wantProblem: "is missing"},
		{name: "source value empty", path: []any{"source", "dataset"}, value: "", wantPath: "source.dataset"},
		{name: "source value not a string", path: []any{"source", "revision"}, value: 7, wantPath: "source.revision"},
		{name: "predictor_type unknown", path: []any{"predictor_type"}, value: "xgb", wantPath: "predictor_type"},
		{name: "strongest not a model", path: []any{"strongest_model"}, value: "nope", wantPath: "strongest_model"},
		{name: "cheapest not a model", path: []any{"cheapest_model"}, value: "nope", wantPath: "cheapest_model"},
		{name: "wrong json type", path: []any{"models", 0, "name"}, value: 5, wantPath: "models[0].name", wantProblem: "wrong type"},

		// Feature spec.
		{name: "hash_dim", path: []any{"feature_spec", "hash_dim"}, value: 1024, wantPath: "feature_spec.hash_dim"},
		{name: "ngram", path: []any{"feature_spec", "ngram"}, value: []any{1, 3}, wantPath: "feature_spec.ngram"},
		{name: "dense names", path: []any{"feature_spec", "dense"}, value: denseNames, wantPath: "feature_spec.dense"},
		{name: "dense_mean short", path: []any{"feature_spec", "dense_mean"}, value: numbers(7, "0.5"), wantPath: "feature_spec.dense_mean", wantProblem: "must have 8 entries, has 7"},
		{name: "dense_std short", path: []any{"feature_spec", "dense_std"}, value: numbers(9, "0.5"), wantPath: "feature_spec.dense_std"},
		{name: "dense_std zero", path: []any{"feature_spec", "dense_std", 2}, value: 0, wantPath: "feature_spec.dense_std[2]", wantProblem: "must be positive"},
		{name: "dense_std negative", path: []any{"feature_spec", "dense_std", 5}, value: -1.5, wantPath: "feature_spec.dense_std[5]", wantProblem: "must be positive"},
		{name: "dense_std missing", del: true, path: []any{"feature_spec", "dense_std"}, wantPath: "feature_spec.dense_std", wantProblem: "is missing"},

		// Models.
		{name: "models empty", path: []any{"models"}, value: []any{}, wantPath: "models"},
		{name: "model name empty", path: []any{"models", 0, "name"}, value: "", wantPath: "models[0].name"},
		{name: "duplicate model name", path: []any{"models", 1, "name"}, value: "gpt-5", wantPath: "models[1].name", wantProblem: "duplicates"},
		{name: "expected tokens negative", path: []any{"models", 0, "expected_output_tokens"}, value: -1, wantPath: "models[0].expected_output_tokens", wantProblem: "must not be negative"},
		{name: "expected tokens null", path: []any{"models", 0, "expected_output_tokens"}, value: nil, wantPath: "models[0].expected_output_tokens", wantProblem: "must not be null"},
		{name: "expected tokens missing", del: true, path: []any{"models", 0, "expected_output_tokens"}, wantPath: "models[0].expected_output_tokens", wantProblem: "is missing"},
		{name: "predictor missing", del: true, path: []any{"models", 0, "predictor"}, wantPath: "models[0].predictor", wantProblem: "is missing"},
		{name: "predictor not an object", path: []any{"models", 0, "predictor"}, value: []any{}, wantPath: "models[0].predictor", wantProblem: "must be a JSON object"},

		// Calibration.
		{name: "blocks empty", path: []any{"models", 0, "calibration", "blocks"}, value: []any{}, wantPath: "models[0].calibration.blocks", wantProblem: "non-empty"},
		{name: "first block lo", path: cal(0, "lo"), value: 0.5, wantPath: blockPath(0, "lo")},
		{name: "block lo not increasing", path: cal(1, "lo"), value: 0.0, wantPath: blockPath(1, "lo"), wantProblem: "greater than the previous"},
		{name: "block gap", path: cal(0, "hi"), value: 0.1, wantPath: blockPath(0, "hi"), wantProblem: "must equal"},
		{name: "last block hi", path: cal(lastBlock, "hi"), value: 0.9, wantPath: blockPath(lastBlock, "hi"), wantProblem: "must be 1.0"},
		{name: "block n zero", path: cal(1, "n"), value: 0, wantPath: blockPath(1, "n"), wantProblem: "at least 1"},
		{name: "block k above n", path: cal(1, "k"), value: 100000, wantPath: blockPath(1, "k")},
		{name: "block k negative", path: cal(1, "k"), value: -1, wantPath: blockPath(1, "k")},
		{name: "p differs from k/n", path: cal(1, "p"), value: 0.25, wantPath: blockPath(1, "p"), wantProblem: "k / n"},
		{name: "lcb above p", path: cal(1, "lcb"), value: 0.9, wantPath: blockPath(1, "lcb")},
		{name: "lcb negative", path: cal(1, "lcb"), value: -0.1, wantPath: blockPath(1, "lcb")},
		{name: "lcb missing", del: true, path: cal(0, "lcb"), wantPath: blockPath(0, "lcb"), wantProblem: "is missing"},
		{name: "block key missing k", del: true, path: cal(2, "k"), wantPath: blockPath(2, "k"), wantProblem: "is missing"},
		{
			name: "p below previous block", wantPath: blockPath(1, "p"), wantProblem: "below the previous",
			custom: func(t *testing.T, m map[string]any) {
				setAt(t, m, 1, cal(1, "k")...)
				setAt(t, m, float64(1)/float64(block1N), cal(1, "p")...)
				setAt(t, m, 0, cal(1, "lcb")...)
			},
		},
		{
			name: "block unknown key", wantPath: "document", wantProblem: "unknown field",
			custom: func(t *testing.T, m map[string]any) { setAt(t, m, 1, cal(0, "extra")...) },
		},

		// Static rules.
		{name: "family_model value not a model", path: []any{"static_rules", "family_model", "code"}, value: "nope", wantPath: "static_rules.family_model.code", wantProblem: "not a model"},
		{name: "family_order empty", path: []any{"static_rules", "family_order"}, value: []any{}, wantPath: "static_rules.family_order"},
		{name: "family_order element empty", path: []any{"static_rules", "family_order", 0}, value: "", wantPath: "static_rules.family_order[0]"},
		{
			name: "family_keywords extra family", wantPath: "static_rules.family_keywords", wantProblem: "exactly",
			custom: func(t *testing.T, m map[string]any) {
				setAt(t, m, []any{"x"}, "static_rules", "family_keywords", "zzz")
			},
		},
		{name: "family_model missing family", del: true, path: []any{"static_rules", "family_model", "qa"}, wantPath: "static_rules.family_model", wantProblem: "exactly"},
		{name: "keyword empty", path: []any{"static_rules", "family_keywords", "qa", 0}, value: "", wantPath: "static_rules.family_keywords.qa[0]"},
		{name: "keyword list null", path: []any{"static_rules", "family_keywords", "qa"}, value: nil, wantPath: "static_rules.family_keywords.qa", wantProblem: "must be a list"},
		{name: "default_family not in order", path: []any{"static_rules", "default_family"}, value: "zzz", wantPath: "static_rules.default_family"},
		{name: "static_rules key missing", del: true, path: []any{"static_rules", "default_family"}, wantPath: "static_rules.default_family", wantProblem: "is missing"},

		// Profiles.
		{name: "profile lambda missing", del: true, path: []any{"profiles", 0, "lambda"}, wantPath: "profiles[0].lambda", wantProblem: "is missing"},
		{name: "profile lambda off grid", path: []any{"profiles", 1, "lambda"}, value: 0.955, wantPath: "profiles[1].lambda"},
		{name: "profile lambda above sentinel", path: []any{"profiles", 1, "lambda"}, value: 1.02, wantPath: "profiles[1].lambda"},
		{name: "profile lambda negative", path: []any{"profiles", 1, "lambda"}, value: -0.01, wantPath: "profiles[1].lambda"},
		{name: "profile count", path: []any{"profiles"}, value: []any{}, wantPath: "profiles", wantProblem: "strict, balanced, economy"},
		{name: "profile name", path: []any{"profiles", 0, "name"}, value: "strong", wantPath: "profiles[0].name"},
		{name: "profile alpha", path: []any{"profiles", 2, "alpha"}, value: 0.2, wantPath: "profiles[2].alpha"},
		{name: "profile alpha out of range", path: []any{"profiles", 0, "alpha"}, value: jn("1e999"), wantPath: "profiles[0].alpha", wantProblem: "wrong type"},
		{name: "profile delta", path: []any{"profiles", 0, "delta"}, value: 0.1, wantPath: "profiles[0].delta"},
		{name: "cert_n as float", path: []any{"profiles", 0, "cert_n"}, value: jn("1747.5"), wantPath: "profiles[0].cert_n", wantProblem: "wrong type"},
		{name: "cert_regret above cert_n", path: []any{"profiles", 0, "cert_regret"}, value: 100000, wantPath: "profiles[0].cert_regret"},
		{name: "cert_regret negative", path: []any{"profiles", 0, "cert_regret"}, value: -1, wantPath: "profiles[0].cert_regret"},
		{name: "regret_ucb above one", path: []any{"profiles", 0, "regret_ucb"}, value: 1.5, wantPath: "profiles[0].regret_ucb"},
		{name: "regret_ucb missing", del: true, path: []any{"profiles", 0, "regret_ucb"}, wantPath: "profiles[0].regret_ucb", wantProblem: "is missing"},

		// gbm predictor (the committed policy).
		{name: "gbm feature_dtype", path: pred("feature_dtype"), value: "float64", wantPath: "models[0].predictor.feature_dtype"},
		{name: "gbm trees missing", del: true, path: pred("trees"), wantPath: "models[0].predictor.trees", wantProblem: "is missing"},
		{name: "gbm learning_rate wrong type", path: pred("learning_rate"), value: "fast", wantPath: "models[0].predictor.learning_rate", wantProblem: "wrong type"},
		{name: "gbm unknown key", path: pred("extra"), value: 1, wantPath: "models[0].predictor", wantProblem: "unknown field"},
		{name: "gbm tree key missing", del: true, path: tree("value"), wantPath: treePath + ".value", wantProblem: "is missing"},
		{
			name: "gbm tree arrays differ", wantPath: treePath, wantProblem: "arrays differ in length: threshold",
			custom: func(t *testing.T, m map[string]any) {
				thr := at(t, m, tree("threshold")...).([]any)
				setAt(t, m, thr[:len(thr)-1], tree("threshold")...)
			},
		},
		{name: "gbm tree empty", path: tree("feature"), value: []any{}, wantPath: treePath + ".feature", wantProblem: "must not be empty"},
		{name: "gbm child not after parent", path: append(tree("left"), 0), value: 0, wantPath: treePath + ".left[0]", wantProblem: "must be -1 or in"},
		{name: "gbm leaf right child", path: append(tree("right"), leaf), value: 1, wantPath: fmt.Sprintf("%s.right[%d]", treePath, leaf), wantProblem: "-1 on a leaf"},
		{name: "gbm feature range", path: append(tree("feature"), 0), value: HashDim + len(DenseNames), wantPath: treePath + ".feature[0]"},
		{name: "gbm feature negative", path: append(tree("feature"), 0), value: -1, wantPath: treePath + ".feature[0]"},

		// lr predictor.
		{name: "lr indices not ascending", lr: true, path: pred("sparse_indices"), value: []any{17, 3, 262143}, wantPath: "models[0].predictor.sparse_indices[1]"},
		{name: "lr indices repeat", lr: true, path: pred("sparse_indices"), value: []any{3, 3, 262143}, wantPath: "models[0].predictor.sparse_indices[1]"},
		{name: "lr index out of range", lr: true, path: pred("sparse_indices"), value: []any{3, 17, 262144}, wantPath: "models[0].predictor.sparse_indices[2]"},
		{name: "lr index negative", lr: true, path: pred("sparse_indices"), value: []any{-1, 17, 262143}, wantPath: "models[0].predictor.sparse_indices[0]"},
		{name: "lr weights length", lr: true, path: pred("sparse_weights"), value: []any{}, wantPath: "models[0].predictor.sparse_weights", wantProblem: "has 0 entries, sparse_indices has 3"},
		{name: "lr dense_weights short", lr: true, path: pred("dense_weights"), value: numbers(7, "0.5"), wantPath: "models[0].predictor.dense_weights", wantProblem: "must have 8 entries, has 7"},
		{name: "lr indices missing", lr: true, del: true, path: pred("sparse_indices"), wantPath: "models[0].predictor.sparse_indices", wantProblem: "is missing"},
		{name: "lr weights null", lr: true, path: pred("sparse_weights"), value: nil, wantPath: "models[0].predictor.sparse_weights", wantProblem: "must not be null"},
		{name: "lr intercept wrong type", lr: true, path: pred("intercept"), value: "x", wantPath: "models[0].predictor.intercept", wantProblem: "wrong type"},
		{name: "lr unknown key", lr: true, path: pred("extra"), value: 1, wantPath: "models[0].predictor", wantProblem: "unknown field"},
		{name: "lr rule applies to later models", lr: true, path: []any{"models", 3, "predictor", "dense_weights"}, value: numbers(9, "0.5"), wantPath: "models[3].predictor.dense_weights"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := smallPolicyMap(t)
			if tc.lr {
				asLR(m)
			}
			switch {
			case tc.del:
				delAt(t, m, tc.path...)
			case tc.path != nil:
				setAt(t, m, tc.value, tc.path...)
			}
			if tc.custom != nil {
				tc.custom(t, m)
			}
			doc, err := ParseDocument(render(t, m, tc.keepVersion))
			if err == nil {
				t.Fatalf("ParseDocument() = %+v, want error at %s", doc, tc.wantPath)
			}
			ve := requireValidationError(t, err)
			if ve.Path != tc.wantPath {
				t.Errorf("error path = %q (%s), want %q", ve.Path, ve.Problem, tc.wantPath)
			}
			if !strings.Contains(ve.Problem, tc.wantProblem) {
				t.Errorf("error problem = %q, want it to contain %q", ve.Problem, tc.wantProblem)
			}
		})
	}
}

func TestParseRejectsMalformedInput(t *testing.T) {
	data := readPolicy(t)
	replace := func(old, repl string) func([]byte) []byte {
		return func(b []byte) []byte {
			if !bytes.Contains(b, []byte(old)) {
				t.Fatalf("committed policy does not contain %q", old)
			}
			return bytes.Replace(b, []byte(old), []byte(repl), 1)
		}
	}
	cases := []struct {
		name      string
		transform func([]byte) []byte
	}{
		{"NaN in a number", replace(`"hash_dim":262144`, `"hash_dim":NaN`)},
		{"Infinity in a number", replace(`"hash_dim":262144`, `"hash_dim":Infinity`)},
		{"negative Infinity in a number", replace(`"hash_dim":262144`, `"hash_dim":-Infinity`)},
		{"trailing object", func(b []byte) []byte { return append(bytes.Clone(b), "{}"...) }},
		{"trailing garbage", func(b []byte) []byte { return append(bytes.Clone(b), "x"...) }},
		{"two documents", func(b []byte) []byte { return append(bytes.Clone(b), b...) }},
		{"truncated", func(b []byte) []byte { return b[:len(b)/2] }},
		{"empty", func([]byte) []byte { return nil }},
		{"whitespace only", func([]byte) []byte { return []byte(" \n") }},
		{"array document", func([]byte) []byte { return []byte(`[]`) }},
		{"null document", func([]byte) []byte { return []byte(`null`) }},
		{"string document", func([]byte) []byte { return []byte(`"policy"`) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			doc, err := ParseDocument(tc.transform(data))
			if err == nil {
				t.Fatalf("ParseDocument() = %+v, want error", doc)
			}
			requireValidationError(t, err)
		})
	}
}

func TestParseReportsContentBeforeVersionMismatch(t *testing.T) {
	// Content problems come first, as in the trainer, so a stale version hash
	// never hides the useful problem.
	m := smallPolicyMap(t)
	setAt(t, m, "nope", "strongest_model")
	m["version"] = "000000000000"
	_, err := ParseDocument(marshalMap(t, m))
	if ve := requireValidationError(t, err); ve.Path != "strongest_model" {
		t.Errorf("error path = %q, want strongest_model", ve.Path)
	}
}

func TestValidateTypedDocument(t *testing.T) {
	doc, err := ParseDocument(readPolicy(t))
	if err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
	if err := Validate(doc); err != nil {
		t.Fatalf("Validate() of a parsed document error: %v", err)
	}

	doc.Profiles[1].Lambda = 0.955
	ve := requireValidationError(t, Validate(doc))
	if ve.Path != "profiles[1].lambda" {
		t.Errorf("error path = %q, want profiles[1].lambda", ve.Path)
	}

	doc, err = ParseDocument(readPolicy(t))
	if err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
	doc.Models = nil
	if ve := requireValidationError(t, Validate(doc)); ve.Path != "models" {
		t.Errorf("error path = %q, want models", ve.Path)
	}

	doc, err = ParseDocument(readPolicy(t))
	if err != nil {
		t.Fatalf("ParseDocument() error: %v", err)
	}
	doc.Source = nil
	if ve := requireValidationError(t, Validate(doc)); ve.Path != "source" {
		t.Errorf("error path = %q, want source", ve.Path)
	}
}
