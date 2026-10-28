package policy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Policy document constants shared with the trainer (certify.py, policy.py).
const (
	// SchemaVersion is the only artifact schema this gateway reads.
	SchemaVersion = 1
	// SentinelLambda marks a profile whose certification failed at every
	// threshold: no model can pass, so every request escalates.
	SentinelLambda = 1.01
	// Delta is the confidence parameter every profile is certified at.
	Delta = 0.05
)

// Predictor types an artifact may carry.
const (
	PredictorLR  = "lr"
	PredictorGBM = "gbm"
)

// ProfileAlphas lists the certified profiles in artifact order with their
// regret rates.
var ProfileAlphas = [3]struct {
	Name  string
	Alpha float64
}{{"strict", 0.02}, {"balanced", 0.05}, {"economy", 0.10}}

// Document is the policy artifact exactly as the trainer writes it. A parsed
// Document has passed Validate.
type Document struct {
	SchemaVersion  int            `json:"schema_version"`
	Version        string         `json:"version"`
	CreatedAt      string         `json:"created_at"`
	Source         map[string]any `json:"source"`
	FeatureSpec    FeatureSpecDoc `json:"feature_spec"`
	PredictorType  string         `json:"predictor_type"`
	Models         []ModelDoc     `json:"models"`
	StrongestModel string         `json:"strongest_model"`
	CheapestModel  string         `json:"cheapest_model"`
	StaticRules    StaticRulesDoc `json:"static_rules"`
	Profiles       []ProfileDoc   `json:"profiles"`
}

// FeatureSpecDoc describes the feature extraction the models were trained on.
type FeatureSpecDoc struct {
	HashDim   int       `json:"hash_dim"`
	NGram     []int     `json:"ngram"`
	Dense     []string  `json:"dense"`
	DenseMean []float64 `json:"dense_mean"`
	DenseStd  []float64 `json:"dense_std"`
}

// ModelDoc is one pool model. Predictor stays raw because its shape depends
// on the document's predictor type; see LRDoc and GBMDoc.
type ModelDoc struct {
	Name                 string          `json:"name"`
	Predictor            json.RawMessage `json:"predictor"`
	Calibration          CalibrationDoc  `json:"calibration"`
	ExpectedOutputTokens float64         `json:"expected_output_tokens"`
}

// CalibrationDoc maps a raw score to an empirical success rate by block.
type CalibrationDoc struct {
	Blocks []BlockDoc `json:"blocks"`
}

// BlockDoc is one calibration block: scores in [Lo, Hi) saw K successes in N
// trials, P = K/N, and LCB is the Wilson lower bound.
type BlockDoc struct {
	Lo  float64 `json:"lo"`
	Hi  float64 `json:"hi"`
	N   int     `json:"n"`
	K   int     `json:"k"`
	P   float64 `json:"p"`
	LCB float64 `json:"lcb"`
}

// LRDoc is a logistic regression predictor over the hashed and dense features.
type LRDoc struct {
	Intercept     float64   `json:"intercept"`
	SparseIndices []int     `json:"sparse_indices"`
	SparseWeights []float64 `json:"sparse_weights"`
	DenseWeights  []float64 `json:"dense_weights"`
}

// GBMDoc is a gradient boosting predictor.
type GBMDoc struct {
	Init         float64   `json:"init"`
	LearningRate float64   `json:"learning_rate"`
	FeatureDType string    `json:"feature_dtype"`
	Trees        []TreeDoc `json:"trees"`
}

// TreeDoc is one regression tree as parallel node arrays. A leaf has
// left == right == -1.
type TreeDoc struct {
	Feature   []int     `json:"feature"`
	Threshold []float64 `json:"threshold"`
	Left      []int     `json:"left"`
	Right     []int     `json:"right"`
	Value     []float64 `json:"value"`
}

// StaticRulesDoc is the keyword baseline the trainer evaluates offline.
type StaticRulesDoc struct {
	FamilyKeywords map[string][]string `json:"family_keywords"`
	FamilyModel    map[string]string   `json:"family_model"`
	FamilyOrder    []string            `json:"family_order"`
	DefaultFamily  string              `json:"default_family"`
}

// ProfileDoc is one certified routing profile (spec 8.5).
type ProfileDoc struct {
	Name       string  `json:"name"`
	Alpha      float64 `json:"alpha"`
	Delta      float64 `json:"delta"`
	Lambda     float64 `json:"lambda"`
	CertN      int     `json:"cert_n"`
	CertRegret int     `json:"cert_regret"`
	RegretUCB  float64 `json:"regret_ucb"`
}

// ValidationError names the JSON path of the first problem found in a policy
// document, in the trainer's path style: models[0].calibration.blocks[1].lo.
type ValidationError struct {
	Path    string
	Problem string
}

func (e *ValidationError) Error() string { return e.Path + ": " + e.Problem }

// fail builds a ValidationError.
func fail(path, format string, args ...any) *ValidationError {
	return &ValidationError{Path: path, Problem: fmt.Sprintf(format, args...)}
}

// joinPath appends a key to a JSON path.
func joinPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

// ParseDocument reads and validates a policy artifact. It rejects unknown
// keys, trailing data, missing or null required keys, every rule of the
// trainer's validate_policy_doc, and a version that is not the canonical
// hash of the content. Every error is a *ValidationError.
func ParseDocument(data []byte) (*Document, error) {
	tree, err := decodeTree(data)
	if err != nil {
		return nil, fail("document", "invalid JSON: %s", err)
	}
	if err := checkSchemaVersion(tree); err != nil {
		return nil, err
	}
	if err := checkRequired(tree); err != nil {
		return nil, err
	}
	var doc Document
	if err := decodeStrict(data, "", &doc); err != nil {
		return nil, err
	}
	if err := Validate(&doc); err != nil {
		return nil, err
	}
	want, err := versionOf(tree)
	if err != nil {
		return nil, fail("document", "cannot be serialized as canonical JSON: %s", err)
	}
	if doc.Version != want {
		return nil, fail("version", "does not match the document content (%s)", want)
	}
	return &doc, nil
}

// checkSchemaVersion runs first so a future schema is named as such instead
// of failing on its unfamiliar keys.
func checkSchemaVersion(root any) error {
	top, ok := root.(map[string]any)
	if !ok {
		return fail("document", "must be a JSON object")
	}
	number, ok := top["schema_version"].(json.Number)
	if v, err := number.Int64(); !ok || err != nil || v != SchemaVersion {
		return fail("schema_version", "must be %d", SchemaVersion)
	}
	return nil
}

// Keys every object of a policy document must carry. A typed decode cannot
// tell a missing key from a zero value, and several zero values (a lambda of
// 0.0, a lower bound of 0) are valid, so presence is checked on the generic
// tree.
var (
	topLevelKeys    = []string{"schema_version", "version", "created_at", "source", "feature_spec", "predictor_type", "models", "strongest_model", "cheapest_model", "static_rules", "profiles"}
	sourceKeys      = []string{"dataset", "revision", "git_sha"}
	featureSpecKeys = []string{"hash_dim", "ngram", "dense", "dense_mean", "dense_std"}
	modelKeys       = []string{"name", "predictor", "calibration", "expected_output_tokens"}
	blockKeys       = []string{"lo", "hi", "n", "k", "p", "lcb"}
	treeKeys        = []string{"feature", "threshold", "left", "right", "value"}
	staticRulesKeys = []string{"family_keywords", "family_model", "family_order", "default_family"}
	profileKeys     = []string{"name", "alpha", "delta", "lambda", "cert_n", "cert_regret", "regret_ucb"}
	predictorKeys   = map[string][]string{
		PredictorLR:  {"intercept", "sparse_indices", "sparse_weights", "dense_weights"},
		PredictorGBM: {"init", "learning_rate", "feature_dtype", "trees"},
	}
)

// requireKeys fails on the first of keys that obj lacks or holds as null.
func requireKeys(obj map[string]any, path string, keys []string) error {
	for _, key := range keys {
		v, ok := obj[key]
		switch {
		case !ok:
			return fail(joinPath(path, key), "is missing")
		case v == nil:
			return fail(joinPath(path, key), "must not be null")
		}
	}
	return nil
}

// requireIn checks the keys of parent[key] when it is an object. A value of
// another type is left to the typed decode, which names the wrong type.
func requireIn(parent map[string]any, key, path string, keys []string) error {
	obj, ok := parent[key].(map[string]any)
	if !ok {
		return nil
	}
	return requireKeys(obj, path, keys)
}

// requireEach checks the keys of every object in the list parent[key].
func requireEach(parent map[string]any, key, path string, keys []string) error {
	list, _ := parent[key].([]any)
	for i, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		if err := requireKeys(obj, fmt.Sprintf("%s[%d]", path, i), keys); err != nil {
			return err
		}
	}
	return nil
}

// checkRequired verifies that every required key of the document is present.
func checkRequired(root any) error {
	top, ok := root.(map[string]any)
	if !ok {
		return fail("document", "must be a JSON object")
	}
	if err := requireKeys(top, "", topLevelKeys); err != nil {
		return err
	}
	if err := requireIn(top, "source", "source", sourceKeys); err != nil {
		return err
	}
	if err := requireIn(top, "feature_spec", "feature_spec", featureSpecKeys); err != nil {
		return err
	}
	predictorType, _ := top["predictor_type"].(string)
	models, _ := top["models"].([]any)
	for i, item := range models {
		model, ok := item.(map[string]any)
		if !ok {
			continue
		}
		path := fmt.Sprintf("models[%d]", i)
		if err := requireKeys(model, path, modelKeys); err != nil {
			return err
		}
		if err := checkPredictorKeys(model, predictorType, path+".predictor"); err != nil {
			return err
		}
		if err := requireIn(model, "calibration", path+".calibration", []string{"blocks"}); err != nil {
			return err
		}
		if cal, ok := model["calibration"].(map[string]any); ok {
			if err := requireEach(cal, "blocks", path+".calibration.blocks", blockKeys); err != nil {
				return err
			}
		}
	}
	if err := requireIn(top, "static_rules", "static_rules", staticRulesKeys); err != nil {
		return err
	}
	return requireEach(top, "profiles", "profiles", profileKeys)
}

// checkPredictorKeys checks the keys of model["predictor"] for the document's
// predictor type. An unknown type is left to Validate.
func checkPredictorKeys(model map[string]any, predictorType, path string) error {
	keys, known := predictorKeys[predictorType]
	if !known {
		return nil
	}
	predictor, ok := model["predictor"].(map[string]any)
	if !ok {
		return nil
	}
	if err := requireKeys(predictor, path, keys); err != nil {
		return err
	}
	if predictorType == PredictorGBM {
		return requireEach(predictor, "trees", path+".trees", treeKeys)
	}
	return nil
}

// decodeStrict decodes one JSON value into v, rejecting unknown keys. Errors
// are reported at base, the JSON path of the value being decoded.
func decodeStrict(data []byte, base string, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return decodeFailure(err, base)
	}
	return nil
}

// decodeFailure turns a decode error into a ValidationError. A type mismatch
// is named by its key path.
func decodeFailure(err error, base string) error {
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		path := base
		if typeErr.Field != "" {
			path = joinPath(base, indexedPath(typeErr.Field))
		}
		if path == "" {
			path = "document"
		}
		return fail(path, "has the wrong type: expected %s, got JSON %s", typeErr.Type, typeErr.Value)
	}
	if base == "" {
		base = "document"
	}
	return fail(base, "%s", err)
}

// indexedPath rewrites encoding/json's field path "models.0.name" in the
// trainer's style, "models[0].name". An all-digit segment is taken to be an
// array index.
func indexedPath(field string) string {
	var out strings.Builder
	for i, segment := range strings.Split(field, ".") {
		if segment != "" && strings.Trim(segment, "0123456789") == "" {
			out.WriteString("[" + segment + "]")
			continue
		}
		if i > 0 {
			out.WriteByte('.')
		}
		out.WriteString(segment)
	}
	return out.String()
}

// decodePredictor decodes the raw predictor object of the model at path.
func decodePredictor(raw json.RawMessage, path string, v any) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return fail(path, "is missing")
	}
	if trimmed[0] != '{' {
		return fail(path, "must be a JSON object")
	}
	return decodeStrict(trimmed, path, v)
}

// decodeLR decodes an lr predictor.
func decodeLR(raw json.RawMessage, path string) (*LRDoc, error) {
	var p LRDoc
	if err := decodePredictor(raw, path, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// decodeGBM decodes a gbm predictor.
func decodeGBM(raw json.RawMessage, path string) (*GBMDoc, error) {
	var p GBMDoc
	if err := decodePredictor(raw, path, &p); err != nil {
		return nil, err
	}
	return &p, nil
}
