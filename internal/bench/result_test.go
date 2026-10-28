package bench

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// objectKeys returns the keys of every JSON object in data, in document order,
// one slice per object.
func objectKeys(t *testing.T, data []byte) [][]string {
	t.Helper()
	type frame struct {
		object    int  // index in out of the object, or -1 for an array
		expectKey bool // the next token is a key of the object
	}
	var (
		out   [][]string
		stack []*frame
	)
	// valueDone records that a value ended inside the enclosing container.
	valueDone := func() {
		if n := len(stack); n > 0 && stack[n-1].object >= 0 {
			stack[n-1].expectKey = true
		}
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	for {
		tok, err := dec.Token()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("decode result: %v", err)
		}
		if d, ok := tok.(json.Delim); ok {
			switch d {
			case '{':
				out = append(out, nil)
				stack = append(stack, &frame{object: len(out) - 1, expectKey: true})
			case '[':
				stack = append(stack, &frame{object: -1})
			default: // } or ]
				stack = stack[:len(stack)-1]
				valueDone()
			}
			continue
		}
		if n := len(stack); n > 0 && stack[n-1].object >= 0 && stack[n-1].expectKey {
			f := stack[n-1]
			out[f.object] = append(out[f.object], tok.(string))
			f.expectKey = false
			continue
		}
		valueDone()
	}
}

func TestWriteResultSortedKeys(t *testing.T) {
	type nested struct {
		Zeta  int `json:"zeta"`
		Alpha int `json:"alpha"`
		Mid   struct {
			Two int `json:"two"`
			One int `json:"one"`
		} `json:"mid"`
	}
	n := nested{Zeta: 1, Alpha: 2}
	n.Mid.Two, n.Mid.One = 4, 3
	r := Result{
		SchemaVersion: 1,
		Kind:          "replay",
		RunID:         "run-1",
		CreatedAt:     "2026-10-28T00:00:00Z",
		Env:           Env{OS: "linux", CPUs: 4, GoVersion: "go1.27.3"},
		ConfigPath:    "bench/configs/replay.yaml",
		Params:        map[string]any{"workers": 16, "base_url": "http://127.0.0.1:18080?a=1&b=<2>"},
		Results:       n,
	}

	dir := filepath.Join(t.TempDir(), "run-1") // does not exist yet
	path, err := WriteResult(dir, r)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(dir, "replay.json"); path != want {
		t.Fatalf("path = %q, want %q", path, want)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasSuffix(data, []byte("}\n")) || bytes.HasSuffix(data, []byte("\n\n")) {
		t.Errorf("result must end with exactly one newline, ends with %q", data[max(0, len(data)-4):])
	}
	if strings.Contains(string(data), `\u003c`) {
		t.Error("result escapes < and >, want them verbatim")
	}

	objects := objectKeys(t, data)
	for _, keys := range objects {
		if !slices.IsSorted(keys) {
			t.Errorf("keys %v are not sorted", keys)
		}
	}
	want := []string{"config_path", "created_at", "env", "kind", "params", "results", "run_id", "schema_version"}
	if !slices.Equal(objects[0], want) {
		t.Errorf("top-level keys = %v, want %v", objects[0], want)
	}
	if len(objects) < 5 {
		t.Errorf("found %d objects, want the nested ones too", len(objects))
	}

	var back struct {
		SchemaVersion int `json:"schema_version"`
		Results       struct {
			Zeta int `json:"zeta"`
			Mid  struct {
				One int `json:"one"`
			} `json:"mid"`
		} `json:"results"`
	}
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.SchemaVersion != 1 || back.Results.Zeta != 1 || back.Results.Mid.One != 3 {
		t.Errorf("values changed on the way out: %+v", back)
	}
}

func TestWriteResultKeepsNumberText(t *testing.T) {
	r := Result{SchemaVersion: 1, Kind: "load-zero", RunID: "r", Results: map[string]any{
		"tiny": 1.5e-9, "big": int64(1) << 60, "round": 0.1,
	}}
	path, err := WriteResult(t.TempDir(), r)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"big": 1152921504606846976`, `"round": 0.1`, `"tiny": 1.5e-9`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("result lacks %s:\n%s", want, data)
		}
	}
}

func TestWriteResultRejectsPromptKey(t *testing.T) {
	for _, c := range []struct {
		name    string
		results any
		reject  bool
	}{
		{"top level", map[string]any{"prompt": "x"}, true},
		{"nested in an object", map[string]any{"a": map[string]any{"prompt": "x"}}, true},
		{"nested in an array", map[string]any{"a": []any{map[string]any{"prompt": "x"}}}, true},
		{"prompt_id is an identifier", map[string]any{"prompt_id": "aime:1"}, false},
		{"a value may say prompt", map[string]any{"note": "prompt"}, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			_, err := WriteResult(dir, Result{SchemaVersion: 1, Kind: "replay", RunID: "r", Results: c.results})
			if c.reject != (err != nil) {
				t.Fatalf("err = %v, want rejection = %v", err, c.reject)
			}
			if c.reject {
				if !strings.Contains(err.Error(), "prompt") {
					t.Errorf("error %q does not name the key", err)
				}
				if _, statErr := os.Stat(filepath.Join(dir, "replay.json")); !errors.Is(statErr, os.ErrNotExist) {
					t.Errorf("a rejected result must not be written (stat error %v)", statErr)
				}
			}
		})
	}
}

func TestWriteResultRejectsBadKind(t *testing.T) {
	for _, kind := range []string{"", "../replay", "a/b", "Replay", "re play"} {
		if _, err := WriteResult(t.TempDir(), Result{Kind: kind}); err == nil {
			t.Errorf("kind %q accepted, want an error", kind)
		}
	}
}

func TestWriteResultRejectsUnencodableValue(t *testing.T) {
	_, err := WriteResult(t.TempDir(), Result{Kind: "replay", Results: map[string]any{"x": func() {}}})
	if err == nil {
		t.Fatal("want an error for a value JSON cannot encode")
	}
}
