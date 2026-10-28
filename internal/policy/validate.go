package policy

import (
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
)

// featureIDLimit bounds a tree split feature id: hashed ids first, then the
// eight dense features.
const featureIDLimit = HashDim + len(DenseNames)

// Validate checks a decoded document against every rule of the trainer's
// validate_policy_doc, in the same order, and returns a *ValidationError for
// the first violation. The version hash is not checked here because it
// depends on the source bytes; ParseDocument checks it last.
func Validate(doc *Document) error {
	if doc.SchemaVersion != SchemaVersion {
		return fail("schema_version", "must be %d", SchemaVersion)
	}
	if !validVersion(doc.Version) {
		return fail("version", "must be 12 lowercase hex characters")
	}
	if doc.CreatedAt == "" {
		return fail("created_at", "must be a non-empty string")
	}
	if err := validateSource(doc.Source); err != nil {
		return err
	}
	if doc.PredictorType != PredictorLR && doc.PredictorType != PredictorGBM {
		return fail("predictor_type", "must be one of %s, %s", PredictorLR, PredictorGBM)
	}
	if err := validateFeatureSpec(&doc.FeatureSpec); err != nil {
		return err
	}
	names, err := validateModels(doc.Models, doc.PredictorType)
	if err != nil {
		return err
	}
	for _, f := range []struct{ key, name string }{
		{"strongest_model", doc.StrongestModel},
		{"cheapest_model", doc.CheapestModel},
	} {
		if !names[f.name] {
			return fail(f.key, "is not a model of the policy")
		}
	}
	if err := validateStaticRules(&doc.StaticRules, names); err != nil {
		return err
	}
	return validateProfiles(doc.Profiles)
}

// validVersion reports whether v is 12 lowercase hex characters.
func validVersion(v string) bool {
	if len(v) != 12 {
		return false
	}
	for i := 0; i < len(v); i++ {
		if c := v[i]; !('0' <= c && c <= '9') && !('a' <= c && c <= 'f') {
			return false
		}
	}
	return true
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// requireFinite fails unless v is a finite number.
func requireFinite(path string, v float64) error {
	if !finite(v) {
		return fail(path, "must be a finite number")
	}
	return nil
}

// checkNumbers validates a list of finite numbers. A negative count accepts
// any length.
func checkNumbers(path string, values []float64, count int) error {
	if values == nil {
		return fail(path, "must be a list")
	}
	if count >= 0 && len(values) != count {
		return fail(path, "must have %d entries, has %d", count, len(values))
	}
	for i, v := range values {
		if err := requireFinite(fmt.Sprintf("%s[%d]", path, i), v); err != nil {
			return err
		}
	}
	return nil
}

func validateSource(source map[string]any) error {
	if source == nil {
		return fail("source", "must be a JSON object")
	}
	for _, key := range sourceKeys {
		v, ok := source[key]
		if !ok {
			return fail("source."+key, "is missing")
		}
		if s, isString := v.(string); !isString || s == "" {
			return fail("source."+key, "must be a non-empty string")
		}
	}
	return nil
}

func validateFeatureSpec(fs *FeatureSpecDoc) error {
	if fs.HashDim != HashDim {
		return fail("feature_spec.hash_dim", "must be %d", HashDim)
	}
	if !slices.Equal(fs.NGram, []int{1, 2}) {
		return fail("feature_spec.ngram", "must be [1, 2]")
	}
	if !slices.Equal(fs.Dense, DenseNames[:]) {
		return fail("feature_spec.dense", "must be %v", DenseNames)
	}
	if err := checkNumbers("feature_spec.dense_mean", fs.DenseMean, len(DenseNames)); err != nil {
		return err
	}
	if err := checkNumbers("feature_spec.dense_std", fs.DenseStd, len(DenseNames)); err != nil {
		return err
	}
	for i, std := range fs.DenseStd {
		if std <= 0 {
			return fail(fmt.Sprintf("feature_spec.dense_std[%d]", i), "must be positive")
		}
	}
	return nil
}

// validateModels checks every model and returns the set of model names.
func validateModels(models []ModelDoc, predictorType string) (map[string]bool, error) {
	if len(models) == 0 {
		return nil, fail("models", "must be a non-empty list")
	}
	names := make(map[string]bool, len(models))
	for i := range models {
		m := &models[i]
		path := fmt.Sprintf("models[%d]", i)
		if m.Name == "" {
			return nil, fail(path+".name", "must be a non-empty string")
		}
		if names[m.Name] {
			return nil, fail(path+".name", "duplicates an earlier model %q", m.Name)
		}
		names[m.Name] = true
		if err := requireFinite(path+".expected_output_tokens", m.ExpectedOutputTokens); err != nil {
			return nil, err
		}
		if m.ExpectedOutputTokens < 0 {
			return nil, fail(path+".expected_output_tokens", "must not be negative")
		}
		var err error
		if predictorType == PredictorLR {
			err = validateLR(m, path+".predictor")
		} else {
			err = validateGBM(m, path+".predictor")
		}
		if err != nil {
			return nil, err
		}
		if err := validateCalibration(&m.Calibration, path+".calibration"); err != nil {
			return nil, err
		}
	}
	return names, nil
}

func validateLR(m *ModelDoc, path string) error {
	p, err := decodeLR(m.Predictor, path)
	if err != nil {
		return err
	}
	if err := requireFinite(path+".intercept", p.Intercept); err != nil {
		return err
	}
	if p.SparseIndices == nil {
		return fail(path+".sparse_indices", "must be a list")
	}
	previous := -1
	for j, index := range p.SparseIndices {
		if index <= previous || index >= HashDim {
			return fail(fmt.Sprintf("%s.sparse_indices[%d]", path, j),
				"must be an integer in [0, %d) and greater than the one before it", HashDim)
		}
		previous = index
	}
	if err := checkNumbers(path+".sparse_weights", p.SparseWeights, -1); err != nil {
		return err
	}
	if len(p.SparseWeights) != len(p.SparseIndices) {
		return fail(path+".sparse_weights", "has %d entries, sparse_indices has %d",
			len(p.SparseWeights), len(p.SparseIndices))
	}
	return checkNumbers(path+".dense_weights", p.DenseWeights, len(DenseNames))
}

func validateGBM(m *ModelDoc, path string) error {
	p, err := decodeGBM(m.Predictor, path)
	if err != nil {
		return err
	}
	if err := requireFinite(path+".init", p.Init); err != nil {
		return err
	}
	if err := requireFinite(path+".learning_rate", p.LearningRate); err != nil {
		return err
	}
	if p.FeatureDType != "float32" {
		return fail(path+".feature_dtype", `must be "float32"`)
	}
	if p.Trees == nil {
		return fail(path+".trees", "must be a list")
	}
	for t := range p.Trees {
		if err := validateTree(&p.Trees[t], fmt.Sprintf("%s.trees[%d]", path, t)); err != nil {
			return err
		}
	}
	return nil
}

func validateTree(tree *TreeDoc, path string) error {
	lengths := [...]struct {
		key    string
		isNil  bool
		length int
	}{
		{"feature", tree.Feature == nil, len(tree.Feature)},
		{"threshold", tree.Threshold == nil, len(tree.Threshold)},
		{"left", tree.Left == nil, len(tree.Left)},
		{"right", tree.Right == nil, len(tree.Right)},
		{"value", tree.Value == nil, len(tree.Value)},
	}
	for _, a := range lengths {
		if a.isNil {
			return fail(joinPath(path, a.key), "must be a list")
		}
	}
	size := len(tree.Feature)
	if size == 0 {
		return fail(joinPath(path, "feature"), "must not be empty")
	}
	for _, a := range lengths {
		if a.length != size {
			return fail(path, "arrays differ in length: %s has %d, feature %d", a.key, a.length, size)
		}
	}
	for node := 0; node < size; node++ {
		if err := requireFinite(fmt.Sprintf("%s.threshold[%d]", path, node), tree.Threshold[node]); err != nil {
			return err
		}
		if err := requireFinite(fmt.Sprintf("%s.value[%d]", path, node), tree.Value[node]); err != nil {
			return err
		}
		left, right := tree.Left[node], tree.Right[node]
		if left == -1 {
			if right != -1 {
				return fail(fmt.Sprintf("%s.right[%d]", path, node), "must be -1 on a leaf")
			}
			continue
		}
		if feature := tree.Feature[node]; feature < 0 || feature >= featureIDLimit {
			return fail(fmt.Sprintf("%s.feature[%d]", path, node), "must be an integer in [0, %d)", featureIDLimit)
		}
		for _, side := range []struct {
			name  string
			child int
		}{{"left", left}, {"right", right}} {
			if side.child <= node || side.child >= size {
				return fail(fmt.Sprintf("%s.%s[%d]", path, side.name, node), "must be -1 or in (%d, %d)", node, size)
			}
		}
	}
	return nil
}

func validateCalibration(c *CalibrationDoc, path string) error {
	if len(c.Blocks) == 0 {
		return fail(path+".blocks", "must be a non-empty list")
	}
	var previous *BlockDoc
	for i := range c.Blocks {
		b := &c.Blocks[i]
		bp := fmt.Sprintf("%s.blocks[%d]", path, i)
		for _, f := range []struct {
			key string
			v   float64
		}{{"lo", b.Lo}, {"hi", b.Hi}, {"p", b.P}, {"lcb", b.LCB}} {
			if err := requireFinite(bp+"."+f.key, f.v); err != nil {
				return err
			}
		}
		if previous == nil {
			if b.Lo != 0 {
				return fail(bp+".lo", "must be 0.0 for the first block")
			}
		} else {
			if b.Lo <= previous.Lo {
				return fail(bp+".lo", "must be greater than the previous block's lo")
			}
			if previous.Hi != b.Lo {
				return fail(fmt.Sprintf("%s.blocks[%d].hi", path, i-1), "must equal blocks[%d].lo", i)
			}
		}
		if b.Hi < b.Lo {
			return fail(bp+".hi", "must not be below lo")
		}
		if b.N < 1 {
			return fail(bp+".n", "must be at least 1")
		}
		if b.K < 0 || b.K > b.N {
			return fail(bp+".k", "must be in [0, n] with n = %d", b.N)
		}
		if want := float64(b.K) / float64(b.N); b.P != want {
			return fail(bp+".p", "must equal k / n (%v)", want)
		}
		if b.LCB < 0 || b.LCB > b.P {
			return fail(bp+".lcb", "must be in [0, p] with p = %v", b.P)
		}
		if previous != nil && b.P < previous.P {
			return fail(bp+".p", "must not be below the previous block's p")
		}
		previous = b
	}
	if previous.Hi != 1 {
		return fail(fmt.Sprintf("%s.blocks[%d].hi", path, len(c.Blocks)-1), "must be 1.0 for the last block")
	}
	return nil
}

// sameFamilies reports whether table has exactly the families in order.
func sameFamilies[V any](table map[string]V, order map[string]bool) bool {
	if len(table) != len(order) {
		return false
	}
	for family := range table {
		if !order[family] {
			return false
		}
	}
	return true
}

func validateStaticRules(r *StaticRulesDoc, models map[string]bool) error {
	if len(r.FamilyOrder) == 0 {
		return fail("static_rules.family_order", "must be a non-empty list")
	}
	order := make(map[string]bool, len(r.FamilyOrder))
	for i, family := range r.FamilyOrder {
		if family == "" {
			return fail(fmt.Sprintf("static_rules.family_order[%d]", i), "must be a non-empty string")
		}
		order[family] = true
	}
	if r.FamilyKeywords == nil {
		return fail("static_rules.family_keywords", "must be a JSON object")
	}
	if r.FamilyModel == nil {
		return fail("static_rules.family_model", "must be a JSON object")
	}
	if !sameFamilies(r.FamilyKeywords, order) {
		return fail("static_rules.family_keywords", "must have exactly the families of family_order")
	}
	if !sameFamilies(r.FamilyModel, order) {
		return fail("static_rules.family_model", "must have exactly the families of family_order")
	}
	// Sorted so the first reported problem does not depend on map order.
	for _, family := range slices.Sorted(maps.Keys(r.FamilyKeywords)) {
		path := "static_rules.family_keywords." + family
		words := r.FamilyKeywords[family]
		if words == nil {
			return fail(path, "must be a list")
		}
		for j, word := range words {
			if word == "" {
				return fail(fmt.Sprintf("%s[%d]", path, j), "must be a non-empty string")
			}
		}
	}
	for _, family := range slices.Sorted(maps.Keys(r.FamilyModel)) {
		if !models[r.FamilyModel[family]] {
			return fail("static_rules.family_model."+family, "is not a model of the policy")
		}
	}
	if !order[r.DefaultFamily] {
		return fail("static_rules.default_family", "is not in family_order")
	}
	return nil
}

// onLambdaGrid reports whether lambda is one of 1.00, 0.99, ..., 0.00,
// computed as the trainer computes the grid.
func onLambdaGrid(lambda float64) bool {
	for i := 0; i <= 100; i++ {
		if lambda == float64(100-i)/100 {
			return true
		}
	}
	return false
}

func validateProfiles(profiles []ProfileDoc) error {
	if profiles == nil {
		return fail("profiles", "must be a list")
	}
	if len(profiles) != len(ProfileAlphas) {
		names := make([]string, len(ProfileAlphas))
		for i, p := range ProfileAlphas {
			names[i] = p.Name
		}
		return fail("profiles", "must list exactly %s", strings.Join(names, ", "))
	}
	for i := range profiles {
		p, want := &profiles[i], ProfileAlphas[i]
		path := fmt.Sprintf("profiles[%d]", i)
		if p.Name != want.Name {
			return fail(path+".name", "must be %q", want.Name)
		}
		if err := requireFinite(path+".alpha", p.Alpha); err != nil {
			return err
		}
		if p.Alpha != want.Alpha {
			return fail(path+".alpha", "must be %v", want.Alpha)
		}
		if err := requireFinite(path+".delta", p.Delta); err != nil {
			return err
		}
		if p.Delta != Delta {
			return fail(path+".delta", "must be %v", Delta)
		}
		if err := requireFinite(path+".lambda", p.Lambda); err != nil {
			return err
		}
		if p.Lambda != SentinelLambda && !onLambdaGrid(p.Lambda) {
			return fail(path+".lambda", "must be on the 0.00 to 1.00 grid or %v", SentinelLambda)
		}
		if p.CertRegret < 0 || p.CertRegret > p.CertN {
			return fail(path+".cert_regret", "must be in [0, cert_n] with cert_n = %d", p.CertN)
		}
		if err := requireFinite(path+".regret_ucb", p.RegretUCB); err != nil {
			return err
		}
		if p.RegretUCB < 0 || p.RegretUCB > 1 {
			return fail(path+".regret_ucb", "must be in [0, 1]")
		}
	}
	return nil
}
