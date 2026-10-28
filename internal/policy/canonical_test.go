package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"testing"
)

func TestCanonicalVersionMatchesCommittedPolicy(t *testing.T) {
	data := readPolicy(t)
	var doc struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("decode committed policy: %v", err)
	}
	got, err := CanonicalVersion(data)
	if err != nil {
		t.Fatalf("CanonicalVersion() error: %v", err)
	}
	if got != doc.Version {
		t.Errorf("CanonicalVersion() = %q, want the trainer's %q", got, doc.Version)
	}
}

// canonicalOf decodes data like CanonicalVersion does and returns the
// canonical bytes of the whole document.
func canonicalOf(t *testing.T, data string) string {
	t.Helper()
	tree, err := decodeTree([]byte(data))
	if err != nil {
		t.Fatalf("decodeTree(%s) error: %v", data, err)
	}
	out, err := canonicalBytes(tree)
	if err != nil {
		t.Fatalf("canonicalBytes(%s) error: %v", data, err)
	}
	return string(out)
}

func TestCanonicalEscapesLikePython(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			// Number text is copied verbatim: 2.50 stays 2.50 and 1e-05 stays 1e-05.
			"quote backslash newline control and non-ASCII",
			`{"b":"a\"\\\n` + `\u0001é","a":[1,2.50,1e-05]}`,
			`{"a":[1,2.50,1e-05],"b":"a\"\\\n` + `\u0001é"}`,
		},
		{
			"two-character escapes, lowercase hex, raw DEL, raw U+2028 and slash",
			`{"s":"\b\f\t\r\u001f\u007f\u2028/"}`,
			"{\"s\":\"\\b\\f\\t\\r\\u001f\x7f\u2028/\"}",
		},
		{
			"uppercase hex escape in the input becomes the character itself",
			`{"s":"\u00E9\u4E1C"}`,
			"{\"s\":\"\u00e9\u4e1c\"}",
		},
		{
			"surrogate pair escape becomes the astral character",
			`{"s":"\ud83d\ude80"}`,
			"{\"s\":\"\U0001f680\"}",
		},
		{
			"empty containers, booleans and null",
			`{"b":[],"a":{},"c":[true,false,null]}`,
			`{"a":{},"b":[],"c":[true,false,null]}`,
		},
		{
			"whitespace is dropped",
			" {\n \"k\" : [ 1 , { \"z\" : 0 , \"y\" : -0.0 } ] \n} ",
			`{"k":[1,{"y":-0.0,"z":0}]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := canonicalOf(t, tc.in); got != tc.want {
				t.Errorf("canonical bytes =\n%q\nwant\n%q", got, tc.want)
			}
		})
	}
}

func TestCanonicalSortsKeysByCodePoint(t *testing.T) {
	// Python sorts keys by code point. UTF-8 byte order is the same order;
	// UTF-16 order would put the astral key before U+FFFF.
	in := "{\"\U0001f680\":1,\"\uffff\":2,\"\u00e9\":3,\"z\":4,\"Z\":5,\"\":6}"
	want := "{\"\":6,\"Z\":5,\"z\":4,\"\u00e9\":3,\"\uffff\":2,\"\U0001f680\":1}"
	if got := canonicalOf(t, in); got != want {
		t.Errorf("canonical bytes =\n%q\nwant\n%q", got, want)
	}
}

func TestCanonicalVersionHashesBodyWithoutVersionAndCreatedAt(t *testing.T) {
	got, err := CanonicalVersion([]byte(`{"version":"x","created_at":"y","a":1}`))
	if err != nil {
		t.Fatalf("CanonicalVersion() error: %v", err)
	}
	sum := sha256.Sum256([]byte(`{"a":1}`))
	if want := hex.EncodeToString(sum[:])[:12]; got != want {
		t.Errorf("CanonicalVersion() = %q, want %q", got, want)
	}
}

func TestCanonicalIgnoresVersionAndCreatedAt(t *testing.T) {
	m := policyMap(t)
	want, err := CanonicalVersion(marshalMap(t, m))
	if err != nil {
		t.Fatalf("CanonicalVersion() error: %v", err)
	}

	m["created_at"] = "2030-01-01T00:00:00Z"
	m["version"] = "ffffffffffff"
	got, err := CanonicalVersion(marshalMap(t, m))
	if err != nil {
		t.Fatalf("CanonicalVersion() error: %v", err)
	}
	if got != want {
		t.Errorf("version changed with created_at and version: %q, want %q", got, want)
	}

	// Nested keys with the same names are content, not metadata.
	setAt(t, m, "nested", "source", "created_at")
	if got, err := CanonicalVersion(marshalMap(t, m)); err != nil || got == want {
		t.Errorf("a nested created_at must change the version: got %q (error %v), unchanged %q", got, err, want)
	}
}

func TestCanonicalVersionChangesWithContent(t *testing.T) {
	m := policyMap(t)
	want, err := CanonicalVersion(marshalMap(t, m))
	if err != nil {
		t.Fatalf("CanonicalVersion() error: %v", err)
	}
	cases := []struct {
		name string
		do   func(t *testing.T, m map[string]any)
	}{
		{"weight value", func(t *testing.T, m map[string]any) {
			setAt(t, m, jn("0.5"), "models", 0, "predictor", "init")
		}},
		{"number text", func(t *testing.T, m map[string]any) {
			// Same value, different text: 1312.0 -> 1312.00.
			setAt(t, m, jn("1312.00"), "models", 0, "expected_output_tokens")
		}},
		{"string", func(t *testing.T, m map[string]any) {
			setAt(t, m, "other", "strongest_model")
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := policyMap(t)
			tc.do(t, m)
			got, err := CanonicalVersion(marshalMap(t, m))
			if err != nil {
				t.Fatalf("CanonicalVersion() error: %v", err)
			}
			if got == want {
				t.Errorf("version did not change: %q", got)
			}
		})
	}
}

func TestCanonicalVersionRejects(t *testing.T) {
	for name, in := range map[string]string{
		"not json":      `{"a":`,
		"not an object": `[1]`,
		"null":          `null`,
		"trailing data": `{"a":1} {}`,
		"empty":         ``,
		"NaN":           `{"a":NaN}`,
	} {
		t.Run(name, func(t *testing.T) {
			if got, err := CanonicalVersion([]byte(in)); err == nil {
				t.Errorf("CanonicalVersion(%q) = %q, want error", in, got)
			}
		})
	}
}
