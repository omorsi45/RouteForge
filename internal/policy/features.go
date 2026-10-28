// Package policy mirrors the offline trainer: feature extraction, the served
// policy artifact, scoring and calibration. Every definition here is a byte
// level contract with trainer/routeforge_train, pinned by
// testdata/parity/features.jsonl. Do not change one side without the other.
package policy

import (
	"math"
	"slices"
	"strings"

	"github.com/omorsi45/routeforge/internal/provider"
)

// HashDim is the size of the hashed feature space.
const HashDim = 262144

// DenseNames lists the dense features in the order of Features.DenseRaw.
var DenseNames = [8]string{
	"log_bytes",
	"log_tokens",
	"digit_frac",
	"upper_frac",
	"code_fences",
	"lines",
	"question_marks",
	"max_token_len",
}

// FNV-1a 32 bit parameters.
const (
	fnvOffset32 = 2166136261
	fnvPrime32  = 16777619
)

// Features is the model input derived from one prompt.
type Features struct {
	Indices  []int32   // hashed feature ids, ascending
	Values   []float64 // L2-normalized 1 + ln(count), parallel to Indices
	DenseRaw [8]float64
	Bytes    int // length of the feature text in bytes
}

// FeatureText joins message contents with "\n" in order. Roles are ignored.
// Array text parts were already joined by the request decoder.
func FeatureText(msgs []provider.Message) string {
	parts := make([]string, len(msgs))
	for i, m := range msgs {
		parts[i] = m.Content
	}
	return strings.Join(parts, "\n")
}

// FNV1a32 is the 32 bit FNV-1a hash of b.
func FNV1a32(b []byte) uint32 {
	h := uint32(fnvOffset32)
	for _, c := range b {
		h = (h ^ uint32(c)) * fnvPrime32
	}
	return h
}

// isTokenByte reports whether c belongs to a token: [a-z0-9].
func isTokenByte(c byte) bool {
	return ('a' <= c && c <= 'z') || ('0' <= c && c <= '9')
}

// Tokenize lowers ASCII A-Z and returns the maximal runs of [a-z0-9] bytes.
// Every other byte, including all of UTF-8 above 0x7f, separates tokens. The
// tokens alias one lowered copy of b.
func Tokenize(b []byte) [][]byte {
	lowered := make([]byte, len(b))
	for i, c := range b {
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		lowered[i] = c
	}
	var tokens [][]byte
	start := -1
	for i, c := range lowered {
		switch {
		case isTokenByte(c):
			if start < 0 {
				start = i
			}
		case start >= 0:
			tokens = append(tokens, lowered[start:i:i])
			start = -1
		}
	}
	if start >= 0 {
		tokens = append(tokens, lowered[start:len(lowered):len(lowered)])
	}
	return tokens
}

// hashIndex maps a unigram or bigram to its hashed feature id.
func hashIndex(gram []byte) int32 {
	return int32(FNV1a32(gram) & (HashDim - 1))
}

// Featurize computes the hashed and dense features of text. It is the
// counterpart of the trainer's hashed_features and dense_raw.
//
// Parity notes: the L2 norm is summed sequentially in ascending index order
// and each value is divided by the norm (never multiplied by its
// reciprocal). Every product that feeds a sum is an explicit float64
// conversion so no compiler can fuse it into a multiply-add.
func Featurize(text string) Features {
	b := []byte(text)
	n := len(b)
	tokens := Tokenize(b)

	counts := make(map[int32]int, 2*len(tokens))
	for _, tok := range tokens {
		counts[hashIndex(tok)]++
	}
	var gram []byte
	for i := 0; i+1 < len(tokens); i++ {
		gram = append(gram[:0], tokens[i]...)
		gram = append(gram, ' ')
		gram = append(gram, tokens[i+1]...)
		counts[hashIndex(gram)]++
	}

	indices := make([]int32, 0, len(counts))
	for idx := range counts {
		indices = append(indices, idx)
	}
	slices.Sort(indices)

	values := make([]float64, len(indices))
	var squares float64
	for i, idx := range indices {
		v := 1 + math.Log(float64(counts[idx]))
		values[i] = v
		squares += float64(v * v)
	}
	norm := math.Sqrt(squares)
	for i := range values {
		values[i] = values[i] / norm
	}

	return Features{Indices: indices, Values: values, DenseRaw: denseRaw(text, tokens), Bytes: n}
}

// denseRaw computes the eight raw dense features, in DenseNames order. All
// counts are over the bytes of text, like the trainer's.
func denseRaw(text string, tokens [][]byte) [8]float64 {
	n := len(text)
	var digits, upper, newlines, questions int
	for i := 0; i < n; i++ {
		switch c := text[i]; {
		case '0' <= c && c <= '9':
			digits++
		case 'A' <= c && c <= 'Z':
			upper++
		case c == '\n':
			newlines++
		case c == '?':
			questions++
		}
	}
	maxToken := 0
	for _, tok := range tokens {
		maxToken = max(maxToken, len(tok))
	}

	var d [8]float64
	d[0] = math.Log(1 + float64(n))
	d[1] = math.Log(1 + float64(len(tokens)))
	if n > 0 {
		d[2] = float64(digits) / float64(n)
		d[3] = float64(upper) / float64(n)
		d[5] = float64(newlines + 1)
	}
	d[4] = float64(strings.Count(text, "```"))
	d[6] = float64(questions)
	d[7] = float64(maxToken)
	return d
}
