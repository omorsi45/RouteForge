// Package bench drives the RouteForge benchmarks: it sends requests to a
// running gateway, measures what comes back and writes result files.
//
// A result file is a Result: the same envelope for every kind of run, so that
// the report renderer reads them all alike. The files hold identifiers,
// counts and timings. They never hold prompt or completion text.
package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
)

// SchemaVersion is the version of the Result envelope.
const SchemaVersion = 1

// promptKey is the one object key a result may never carry, at any depth: a
// field of that name would hold prompt text. Identifiers such as prompt_id are
// different keys.
const promptKey = "prompt"

var kindRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

// Result is the content of one result file.
type Result struct {
	SchemaVersion int    `json:"schema_version"`
	Kind          string `json:"kind"` // replay, load-zero, load-profiled or faults
	RunID         string `json:"run_id"`
	CreatedAt     string `json:"created_at"` // RFC 3339, UTC
	Env           Env    `json:"env"`
	ConfigPath    string `json:"config_path"`
	Params        any    `json:"params"`  // what the run was asked to do
	Results       any    `json:"results"` // what it measured
}

// WriteResult writes r to dir/<kind>.json, creating dir, and returns the path.
// Object keys are sorted at every depth and the file ends with a newline, so
// equal results are equal bytes. A result with a "prompt" key anywhere is
// refused and nothing is written.
func WriteResult(dir string, r Result) (path string, err error) {
	if !kindRE.MatchString(r.Kind) {
		return "", fmt.Errorf("write result: invalid kind %q", r.Kind)
	}
	data, err := canonicalJSON(r)
	if err != nil {
		return "", fmt.Errorf("write %s result: %w", r.Kind, err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("write %s result: %w", r.Kind, err)
	}
	path = filepath.Join(dir, r.Kind+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("write %s result: %w", r.Kind, err)
	}
	return path, nil
}

// canonicalJSON renders v as indented JSON with sorted keys and a trailing
// newline. It goes through a generic value, whose maps the encoder sorts, and
// keeps the text of every number as the first encoding wrote it.
func canonicalJSON(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var generic any
	if err := dec.Decode(&generic); err != nil {
		return nil, err
	}
	if findKey(generic, promptKey) {
		return nil, fmt.Errorf("the result has a %q key, which would hold prompt text", promptKey)
	}
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(generic); err != nil { // Encode ends with a newline
		return nil, err
	}
	return out.Bytes(), nil
}

// findKey reports whether any object in v, at any depth, has the given key.
func findKey(v any, key string) bool {
	switch v := v.(type) {
	case map[string]any:
		for k, child := range v {
			if k == key || findKey(child, key) {
				return true
			}
		}
	case []any:
		for _, child := range v {
			if findKey(child, key) {
				return true
			}
		}
	}
	return false
}
