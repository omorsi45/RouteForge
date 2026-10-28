package provider_test

import (
	"testing"

	"github.com/omorsi45/routeforge/internal/provider"
)

func TestDecodeJSON(t *testing.T) {
	type doc struct {
		A int `json:"a"`
	}
	cases := []struct {
		name    string
		in      string
		wantErr bool
	}{
		{"valid object", `{"a":1}`, false},
		{"trailing newline", "{\"a\":1}\n", false},
		{"trailing whitespace", "{\"a\":1} \t\r\n", false},
		{"trailing second object", `{"a":1}{"a":2}`, true},
		{"trailing garbage", `{"a":1} garbage`, true},
		{"trailing comma", `{"a":1},`, true},
		{"empty input", ``, true},
		{"malformed", `{"a":`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got doc
			err := provider.DecodeJSON([]byte(tc.in), &got)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("DecodeJSON(%q) = nil error, want error", tc.in)
				}
				return
			}
			if err != nil {
				t.Fatalf("DecodeJSON(%q) error: %v", tc.in, err)
			}
			if got.A != 1 {
				t.Fatalf("A = %d, want 1", got.A)
			}
		})
	}
}

func TestMaxErrorBodyBytes(t *testing.T) {
	if provider.MaxErrorBodyBytes != 64<<10 {
		t.Fatalf("MaxErrorBodyBytes = %d, want %d", provider.MaxErrorBodyBytes, 64<<10)
	}
}
