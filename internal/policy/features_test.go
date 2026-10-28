package policy

import (
	"math"
	"reflect"
	"testing"

	"github.com/omorsi45/routeforge/internal/provider"
)

func TestFNV1a32KnownVectors(t *testing.T) {
	cases := []struct {
		in   string
		want uint32
	}{
		{"", 2166136261},
		{"a", 0xe40c292c},
		{"foobar", 0xbf9cf968},
	}
	for _, tc := range cases {
		if got := FNV1a32([]byte(tc.in)); got != tc.want {
			t.Errorf("FNV1a32(%q) = %#x, want %#x", tc.in, got, tc.want)
		}
	}
}

func TestTokenizeLowersASCIIOnly(t *testing.T) {
	got := Tokenize([]byte("Café NAÏVE x9"))
	var words []string
	for _, tok := range got {
		words = append(words, string(tok))
	}
	want := []string{"caf", "na", "ve", "x9"}
	if !reflect.DeepEqual(words, want) {
		t.Errorf("Tokenize() = %q, want %q", words, want)
	}
}

func TestTokenizeNoTokens(t *testing.T) {
	for _, in := range []string{"", "?!?... ---", "\u6771\u4eac"} {
		if got := Tokenize([]byte(in)); len(got) != 0 {
			t.Errorf("Tokenize(%q) = %q, want no tokens", in, got)
		}
	}
}

func TestFeatureTextJoinsContentsIgnoringRoles(t *testing.T) {
	msgs := []provider.Message{
		{Role: provider.RoleSystem, Content: "a"},
		{Role: provider.RoleUser, Content: ""},
		{Role: provider.RoleAssistant, Content: "c\r\nd"},
	}
	if got, want := FeatureText(msgs), "a\n\nc\r\nd"; got != want {
		t.Errorf("FeatureText() = %q, want %q", got, want)
	}
	if got := FeatureText(nil); got != "" {
		t.Errorf("FeatureText(nil) = %q, want empty", got)
	}
}

func TestFeaturizeEmpty(t *testing.T) {
	f := Featurize("")
	if len(f.Indices) != 0 || len(f.Values) != 0 {
		t.Errorf("Indices = %v, Values = %v, want both empty", f.Indices, f.Values)
	}
	if f.DenseRaw != [8]float64{} {
		t.Errorf("DenseRaw = %v, want all zeros", f.DenseRaw)
	}
	if f.Bytes != 0 {
		t.Errorf("Bytes = %d, want 0", f.Bytes)
	}
	for i, v := range f.DenseRaw {
		if math.IsNaN(v) {
			t.Errorf("DenseRaw[%d] is NaN", i)
		}
	}
}

func TestFeaturizeRepeatedTokens(t *testing.T) {
	f := Featurize("the the")
	if len(f.Indices) != 2 || len(f.Values) != 2 {
		t.Fatalf("got %d indices and %d values, want 2 each", len(f.Indices), len(f.Values))
	}
	const mask = HashDim - 1
	unigram := int32(FNV1a32([]byte("the")) & mask)
	bigram := int32(FNV1a32([]byte("the the")) & mask)
	if unigram == bigram {
		t.Fatalf("test needs distinct indices, both are %d", unigram)
	}
	rawUnigram := 1 + math.Log(2)
	rawBigram := 1.0
	norm := math.Sqrt(rawUnigram*rawUnigram + rawBigram*rawBigram)
	want := map[int32]float64{unigram: rawUnigram / norm, bigram: rawBigram / norm}
	for i, idx := range f.Indices {
		w, ok := want[idx]
		if !ok {
			t.Fatalf("unexpected index %d", idx)
		}
		if math.Abs(f.Values[i]-w) > 1e-12 {
			t.Errorf("value at index %d = %v, want %v", idx, f.Values[i], w)
		}
	}
	if f.Indices[0] >= f.Indices[1] {
		t.Errorf("indices not ascending: %v", f.Indices)
	}
	var squares float64
	for _, v := range f.Values {
		squares += v * v
	}
	if math.Abs(squares-1) > 1e-12 {
		t.Errorf("sum of squares = %v, want 1", squares)
	}
}

func TestFeaturizeDenseCounts(t *testing.T) {
	// 8 bytes: A b 1 ? \n ` ` `. One token "ab1".
	f := Featurize("Ab1?\n```")
	want := [8]float64{
		math.Log(9),
		math.Log(2),
		1.0 / 8.0,
		1.0 / 8.0,
		1,
		2,
		1,
		3,
	}
	for i := range want {
		if math.Abs(f.DenseRaw[i]-want[i]) > 1e-12 {
			t.Errorf("DenseRaw[%d] (%s) = %v, want %v", i, DenseNames[i], f.DenseRaw[i], want[i])
		}
	}
	if f.Bytes != 8 {
		t.Errorf("Bytes = %d, want 8", f.Bytes)
	}
}

func TestFeaturizeCountsBytesNotRunes(t *testing.T) {
	f := Featurize("\u6771\u4eac")
	if f.Bytes != 6 {
		t.Errorf("Bytes = %d, want 6", f.Bytes)
	}
	if want := math.Log(7); math.Abs(f.DenseRaw[0]-want) > 1e-12 {
		t.Errorf("log_bytes = %v, want %v", f.DenseRaw[0], want)
	}
	if len(f.Indices) != 0 {
		t.Errorf("Indices = %v, want none for non-ASCII only text", f.Indices)
	}
}

func TestFeaturizeCodeFencesDoNotOverlap(t *testing.T) {
	// Python bytes.count counts non-overlapping matches: 4 backticks hold one fence.
	if got := Featurize("````").DenseRaw[4]; got != 1 {
		t.Errorf("code_fences for 4 backticks = %v, want 1", got)
	}
	if got := Featurize("``````").DenseRaw[4]; got != 2 {
		t.Errorf("code_fences for 6 backticks = %v, want 2", got)
	}
}
