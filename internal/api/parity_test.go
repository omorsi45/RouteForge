package api

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/policy"
)

// parityFixture is the trainer-written parity file, relative to this package.
const parityFixture = "../../testdata/parity/features.jsonl"

// parityCase is the part of a parity row this package checks.
type parityCase struct {
	Name     string          `json:"name"`
	Messages json.RawMessage `json:"messages"`
	Text     string          `json:"text"`
}

func loadParityCases(t *testing.T) []parityCase {
	t.Helper()
	data, err := os.ReadFile(parityFixture)
	if err != nil {
		t.Fatalf("read parity fixture: %v", err)
	}
	var cases []parityCase
	for i, line := range bytes.Split(data, []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var c parityCase
		if err := json.Unmarshal(line, &c); err != nil {
			t.Fatalf("parity fixture line %d: %v", i+1, err)
		}
		cases = append(cases, c)
	}
	if len(cases) != 20 {
		t.Fatalf("parity fixture has %d rows, want 20", len(cases))
	}
	return cases
}

// TestParityFeatureTextThroughAPI feeds each row's messages through the real
// request decoder and requires the feature text the trainer computed. It
// covers a lone surrogate (U+FFFD), CRLF, array content, an empty content
// list and several messages.
func TestParityFeatureTextThroughAPI(t *testing.T) {
	for _, c := range loadParityCases(t) {
		t.Run(c.Name, func(t *testing.T) {
			body := `{"model":"m","messages":` + string(c.Messages) + `}`
			req, derr := decodeChatRequest(strings.NewReader(body))
			if derr != nil {
				t.Fatalf("decodeChatRequest() error: %+v", derr)
			}
			if got := policy.FeatureText(req.Messages); got != c.Text {
				t.Errorf("FeatureText() = %q, want %q", got, c.Text)
			}
		})
	}
}
