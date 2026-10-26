package auth_test

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
)

func TestHashKeyVector(t *testing.T) {
	want := "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" // FIPS 180-2 SHA-256("abc")
	if got := auth.HashKey("abc"); got != want {
		t.Fatal(got)
	}
}

func TestGenerateKeyFormat(t *testing.T) {
	k, err := auth.GenerateKey(bytes.NewReader(make([]byte, 32)))
	if err != nil {
		t.Fatal(err)
	}
	if want := "rf_" + strings.Repeat("A", 43); k != want {
		t.Fatalf("key = %q, want %q", k, want)
	}
	if len(k) != 46 {
		t.Fatalf("len = %d, want 46", len(k))
	}
	if _, err := auth.GenerateKey(bytes.NewReader(make([]byte, 5))); err == nil {
		t.Fatal("short reader: want error")
	}
}

func TestStoreLookup(t *testing.T) {
	s := auth.NewStore([]config.Tenant{
		{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey("test-key-tenant-a")}},
	})
	got, ok := s.Lookup("test-key-tenant-a")
	if !ok || got.Name != "tenant-a" {
		t.Fatalf("Lookup = %v, %v", got, ok)
	}
	if _, ok := s.Lookup("test-key-tenant-b"); ok {
		t.Fatal("unknown key authenticated")
	}
	if _, ok := s.Lookup(""); ok {
		t.Fatal("empty key authenticated")
	}
}

func TestStoreLookupDoesNotRequireKeyPrefix(t *testing.T) {
	s := auth.NewStore([]config.Tenant{
		{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey("plain-key-no-prefix")}},
	})
	got, ok := s.Lookup("plain-key-no-prefix")
	if !ok || got.Name != "tenant-a" {
		t.Fatalf("Lookup = %v, %v", got, ok)
	}
}

func TestStoreMultipleDigestsPerTenant(t *testing.T) {
	s := auth.NewStore([]config.Tenant{
		{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey("test-key-aaaa"), auth.HashKey("test-key-aaab")}},
		{Name: "tenant-b", APIKeySHA256: []string{auth.HashKey("test-key-bbbb")}},
	})
	for key, want := range map[string]string{"test-key-aaaa": "tenant-a", "test-key-aaab": "tenant-a", "test-key-bbbb": "tenant-b"} {
		got, ok := s.Lookup(key)
		if !ok || got.Name != want {
			t.Fatalf("Lookup(%q) = %v, %v; want %s", key, got, ok, want)
		}
	}
}

func TestStoreHoldsCopies(t *testing.T) {
	tenants := []config.Tenant{{Name: "tenant-a", APIKeySHA256: []string{auth.HashKey("test-key-tenant-a")}, AllowedModels: []string{"m1"}}}
	s := auth.NewStore(tenants)
	tenants[0].Name = "mutated"
	tenants[0].AllowedModels[0] = "mutated"
	got, ok := s.Lookup("test-key-tenant-a")
	if !ok || got.Name != "tenant-a" || got.AllowedModels[0] != "m1" {
		t.Fatalf("store aliased caller data: %+v", got)
	}
}

func TestBearerToken(t *testing.T) {
	tests := []struct {
		name   string
		header string
		set    bool
		want   string
		ok     bool
	}{
		{"canonical", "Bearer x", true, "x", true},
		{"lowercase scheme", "bearer x", true, "x", true},
		{"extra spaces", "Bearer  x ", true, "x", true},
		{"wrong scheme", "Basic x", true, "", false},
		{"no token", "Bearer", true, "", false},
		{"blank token", "Bearer ", true, "", false},
		{"missing header", "", false, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := http.Header{}
			if tt.set {
				h.Set("Authorization", tt.header)
			}
			got, ok := auth.BearerToken(h)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("BearerToken = %q, %v; want %q, %v", got, ok, tt.want, tt.ok)
			}
		})
	}
}
