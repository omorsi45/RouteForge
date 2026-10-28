package policy

import (
	"math"
	"testing"
)

// logistic1 is the reference logistic the predictors must produce.
func logistic1(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

func TestLRRawSmallModel(t *testing.T) {
	lr := &LR{
		Intercept:     0.5,
		DenseWeights:  [8]float64{1},
		SparseIndices: []int32{7},
		SparseWeights: []float64{2.0},
	}
	// Indices 3 and 9 have no sparse weight and must be ignored.
	f := &Features{Indices: []int32{3, 7, 9}, Values: []float64{0.1, 0.5, 0.2}}
	denseZ := [8]float64{1}
	// z = 1*1 + 0.5 (intercept) + 0.5*2.0 = 2.5, all exact in binary.
	if got, want := lr.Raw(f, &denseZ), logistic1(0.5+1+1.0); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestLRRawDenseWeightsSumInIndexOrder(t *testing.T) {
	lr := &LR{
		Intercept:    -1,
		DenseWeights: [8]float64{0.5, -1, 2, 0, 0, 0, 0, 0.25},
	}
	denseZ := [8]float64{1, 2, 3, 4, 5, 6, 7, 8}
	// 0.5 - 2 + 6 + 2 = 6.5, minus 1.
	if got, want := lr.Raw(&Features{}, &denseZ), logistic1(5.5); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestLRRawSparseMergeWalk(t *testing.T) {
	lr := &LR{
		SparseIndices: []int32{2, 5, 8, 100, 262143},
		SparseWeights: []float64{1, 2, 4, 8, 16},
	}
	// Matches: 2, 5, 100 and 262143. Feature ids 1, 6, 101 are absent from
	// the weights, and weight 8 is absent from the features.
	f := &Features{
		Indices: []int32{1, 2, 5, 6, 100, 101, 262143},
		Values:  []float64{9, 0.5, 0.25, 9, 0.125, 9, 0.0625},
	}
	var denseZ [8]float64
	// 0.5*1 + 0.25*2 + 0.125*8 + 0.0625*16 = 0.5 + 0.5 + 1 + 1 = 3.
	if got, want := lr.Raw(f, &denseZ), logistic1(3); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestLRRawWithoutSparseWeights(t *testing.T) {
	lr := &LR{Intercept: 0.25, DenseWeights: [8]float64{2}}
	f := &Features{Indices: []int32{1, 2}, Values: []float64{0.5, 0.5}}
	denseZ := [8]float64{0.5}
	if got, want := lr.Raw(f, &denseZ), logistic1(1.25); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestLRRawSaturates(t *testing.T) {
	lr := &LR{Intercept: -1e6}
	var denseZ [8]float64
	if got := lr.Raw(&Features{}, &denseZ); got != 0 {
		t.Errorf("Raw() with z = -1e6 = %v, want 0", got)
	}
	lr.Intercept = 1e6
	if got := lr.Raw(&Features{}, &denseZ); got != 1 {
		t.Errorf("Raw() with z = 1e6 = %v, want 1", got)
	}
}

// stump is a one-split tree: left leaf lv, right leaf rv, split on feature
// id at threshold.
func stump(feature int32, threshold, lv, rv float64) Tree {
	return Tree{
		Feature:   []int32{feature, -1, -1},
		Threshold: []float64{threshold, 0, 0},
		Left:      []int32{1, -1, -1},
		Right:     []int32{2, -1, -1},
		Value:     []float64{0, lv, rv},
	}
}

func TestGBMFloat32Comparison(t *testing.T) {
	const threshold = 0.1 // exactly as a float64
	g := &GBM{LearningRate: 1, Trees: []Tree{stump(5, threshold, -1, 1)}}
	var denseZ [8]float64

	// float32(0.1) is 0.10000000149..., above the float64 threshold 0.1, so
	// a value equal to the threshold goes right.
	x := 0.1
	if float64(float32(x)) <= threshold {
		t.Fatalf("precondition: float64(float32(%v)) = %v should exceed %v", x, float64(float32(x)), threshold)
	}
	// 0.09999999999 is below the threshold in float64 but rounds to the same
	// float32 as 0.1, so it also goes right: this is the case a plain float64
	// comparison would get wrong.
	below := 0.09999999999
	if !(below < threshold) || float64(float32(below)) <= threshold {
		t.Fatalf("precondition: %v must be below %v in float64 and above it after float32 rounding", below, threshold)
	}
	// A value whose float32 equals the threshold when widened goes left.
	exact := float64(float32(0.1))
	gExact := &GBM{LearningRate: 1, Trees: []Tree{stump(5, exact, -1, 1)}}

	cases := []struct {
		name string
		g    *GBM
		x    float64
		want float64
	}{
		{"threshold value goes right", g, x, logistic1(1)},
		{"float64 below, float32 above goes right", g, below, logistic1(1)},
		{"well below goes left", g, 0.05, logistic1(-1)},
		{"float32 equal goes left", gExact, exact, logistic1(-1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := &Features{Indices: []int32{5}, Values: []float64{tc.x}}
			if got := tc.g.Raw(f, &denseZ); got != tc.want {
				t.Errorf("Raw() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGBMAbsentHashedFeatureReadsZero(t *testing.T) {
	g := &GBM{LearningRate: 1, Trees: []Tree{stump(5, 0, -1, 1)}}
	var denseZ [8]float64
	// Feature 5 is not in the vector, so x = 0 and 0 <= 0 goes left.
	f := &Features{Indices: []int32{4, 6}, Values: []float64{1, 1}}
	if got, want := g.Raw(f, &denseZ), logistic1(-1); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
	// An empty feature vector behaves the same way.
	if got, want := g.Raw(&Features{}, &denseZ), logistic1(-1); got != want {
		t.Errorf("Raw() on empty features = %v, want %v", got, want)
	}
}

func TestGBMDenseFeatureIDs(t *testing.T) {
	// Ids at or above HashDim read the standardized dense features.
	g := &GBM{LearningRate: 1, Trees: []Tree{stump(HashDim+2, 0.5, -1, 1)}}
	f := &Features{Indices: []int32{2}, Values: []float64{99}} // hashed id 2 must not be read
	left := [8]float64{0, 0, 0.5}
	right := [8]float64{0, 0, 0.75}
	if got, want := g.Raw(f, &left), logistic1(-1); got != want {
		t.Errorf("Raw() with dense 0.5 = %v, want %v", got, want)
	}
	if got, want := g.Raw(f, &right), logistic1(1); got != want {
		t.Errorf("Raw() with dense 0.75 = %v, want %v", got, want)
	}
}

func TestGBMSumsTreesInOrder(t *testing.T) {
	// A depth-2 tree: node 0 splits on id 1, node 1 splits on id 2.
	deep := Tree{
		Feature:   []int32{1, 2, -1, -1, -1},
		Threshold: []float64{0.5, 0.5, 0, 0, 0},
		Left:      []int32{1, 3, -1, -1, -1},
		Right:     []int32{2, 4, -1, -1, -1},
		Value:     []float64{0, 0, 0.3, 0.7, -0.2},
	}
	g := &GBM{
		Init:         0.25,
		LearningRate: 0.1,
		Trees:        []Tree{stump(9, 0.5, 1.5, -1.5), deep},
	}
	var denseZ [8]float64
	// Feature 9 absent: tree 1 goes left to 1.5. Feature 1 is 0.25 (left),
	// feature 2 is 0.75 (right): tree 2 reaches node 4 = -0.2.
	f := &Features{Indices: []int32{1, 2}, Values: []float64{0.25, 0.75}}
	// Variables, not constants: constant arithmetic is exact and would not
	// round like the float64 products the predictor computes.
	rate, first, second := 0.1, 1.5, -0.2
	sum := 0.25
	sum += float64(rate * first)
	sum += float64(rate * second)
	if got, want := g.Raw(f, &denseZ), logistic1(sum); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestGBMWithoutTreesIsInit(t *testing.T) {
	g := &GBM{Init: 0.75, LearningRate: 0.1}
	var denseZ [8]float64
	if got, want := g.Raw(&Features{}, &denseZ), logistic1(0.75); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}

func TestGBMLeafOnlyTree(t *testing.T) {
	leaf := Tree{
		Feature:   []int32{-1},
		Threshold: []float64{0},
		Left:      []int32{-1},
		Right:     []int32{-1},
		Value:     []float64{2},
	}
	g := &GBM{Init: 0, LearningRate: 0.5, Trees: []Tree{leaf}}
	var denseZ [8]float64
	if got, want := g.Raw(&Features{}, &denseZ), logistic1(1); got != want {
		t.Errorf("Raw() = %v, want %v", got, want)
	}
}
