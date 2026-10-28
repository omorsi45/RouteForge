package policy

import (
	"math"
	"slices"
)

// Predictor maps one prompt's features to a raw success score in [0, 1].
// denseZ holds the standardized dense features. Implementations must not
// retain f or denseZ.
type Predictor interface {
	Raw(f *Features, denseZ *[8]float64) float64
}

// logistic is 1 / (1 + exp(-z)), the trainer's expit.
func logistic(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// LR is a logistic regression over the hashed and dense features. Only the
// hashed features with a retained weight are listed.
type LR struct {
	Intercept     float64
	SparseIndices []int32 // ascending hashed feature ids
	SparseWeights []float64
	DenseWeights  [8]float64
}

// newLR compiles a validated LR document.
func newLR(d *LRDoc) *LR {
	lr := &LR{
		Intercept:     d.Intercept,
		SparseIndices: make([]int32, len(d.SparseIndices)),
		SparseWeights: append([]float64(nil), d.SparseWeights...),
	}
	for i, idx := range d.SparseIndices {
		lr.SparseIndices[i] = int32(idx)
	}
	copy(lr.DenseWeights[:], d.DenseWeights)
	return lr
}

// Raw mirrors the trainer's LRModel.raw: the dense dot product plus the
// intercept, then the sparse dot product added to it. Sums run in ascending
// index order and every product is an explicit float64 conversion so no
// fused multiply-add can change a rounding.
func (m *LR) Raw(f *Features, denseZ *[8]float64) float64 {
	var dense float64
	for j := range m.DenseWeights {
		dense += float64(m.DenseWeights[j] * denseZ[j])
	}
	z := dense + m.Intercept

	// Merge walk over two ascending index lists.
	var sparse float64
	i, j := 0, 0
	for i < len(f.Indices) && j < len(m.SparseIndices) {
		switch fi, wi := f.Indices[i], m.SparseIndices[j]; {
		case fi < wi:
			i++
		case fi > wi:
			j++
		default:
			sparse += float64(f.Values[i] * m.SparseWeights[j])
			i++
			j++
		}
	}
	return logistic(sparse + z)
}

// Tree is one regression tree as parallel node arrays, root at node 0. A
// leaf has Left == Right == -1. Feature ids below HashDim are hashed
// features; HashDim+k is standardized dense feature k.
type Tree struct {
	Feature   []int32
	Threshold []float64
	Left      []int32
	Right     []int32
	Value     []float64
}

// GBM is a gradient boosting predictor.
type GBM struct {
	Init         float64
	LearningRate float64
	Trees        []Tree
}

// newGBM compiles a validated GBM document.
func newGBM(d *GBMDoc) *GBM {
	g := &GBM{Init: d.Init, LearningRate: d.LearningRate, Trees: make([]Tree, len(d.Trees))}
	for i := range d.Trees {
		t := &d.Trees[i]
		g.Trees[i] = Tree{
			Feature:   toInt32(t.Feature),
			Threshold: append([]float64(nil), t.Threshold...),
			Left:      toInt32(t.Left),
			Right:     toInt32(t.Right),
			Value:     append([]float64(nil), t.Value...),
		}
	}
	return g
}

func toInt32(values []int) []int32 {
	out := make([]int32, len(values))
	for i, v := range values {
		out[i] = int32(v)
	}
	return out
}

// featureValue reads feature id: a hashed value (0 when absent) below
// HashDim, a standardized dense feature from HashDim on.
func featureValue(f *Features, denseZ *[8]float64, id int32) float64 {
	if id >= HashDim {
		return denseZ[id-HashDim]
	}
	if i, found := slices.BinarySearch(f.Indices, id); found {
		return f.Values[i]
	}
	return 0
}

// Raw mirrors the trainer's GBMModel.raw. The trainer's trees were fit on
// float32 inputs, so every comparison is float64(float32(x)) <= threshold.
// Leaf contributions are summed in tree order, each product explicitly
// converted to float64.
func (m *GBM) Raw(f *Features, denseZ *[8]float64) float64 {
	sum := m.Init
	for i := range m.Trees {
		t := &m.Trees[i]
		node := 0
		for t.Left[node] != -1 {
			x := featureValue(f, denseZ, t.Feature[node])
			if float64(float32(x)) <= t.Threshold[node] {
				node = int(t.Left[node])
			} else {
				node = int(t.Right[node])
			}
		}
		sum += float64(m.LearningRate * t.Value[node])
	}
	return logistic(sum)
}

// compilePredictor builds the predictor of the model at path from its raw
// document. The document has passed Validate.
func compilePredictor(doc *Document, m *ModelDoc, path string) (Predictor, error) {
	switch doc.PredictorType {
	case PredictorLR:
		d, err := decodeLR(m.Predictor, path)
		if err != nil {
			return nil, err
		}
		return newLR(d), nil
	case PredictorGBM:
		d, err := decodeGBM(m.Predictor, path)
		if err != nil {
			return nil, err
		}
		return newGBM(d), nil
	}
	return nil, fail(path, "unsupported predictor type %q", doc.PredictorType)
}
