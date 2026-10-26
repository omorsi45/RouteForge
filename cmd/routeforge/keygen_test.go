package main

import (
	"bytes"
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/auth"
)

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errors.New("write failed") }

func TestKeygenOutput(t *testing.T) {
	var out bytes.Buffer
	if code := cmdKeygen(&out, bytes.NewReader(make([]byte, 32))); code != 0 {
		t.Fatal(code)
	}
	key := "rf_" + strings.Repeat("A", 43)
	if out.String() != "key: "+key+"\nsha256: "+auth.HashKey(key)+"\n" {
		t.Fatalf("%q", out.String())
	}
}

func TestKeygenViaRun(t *testing.T) {
	var out, errb bytes.Buffer
	code := run(context.Background(), []string{"keygen"}, &out, &errb, func(string) string { return "" })
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errb.String())
	}
	first, _, _ := strings.Cut(out.String(), "\n")
	if !regexp.MustCompile(`^key: rf_[A-Za-z0-9_-]{43}$`).MatchString(first) {
		t.Fatalf("first line = %q", first)
	}
}

func TestKeygenRandomFailure(t *testing.T) {
	var out bytes.Buffer
	if code := cmdKeygen(&out, bytes.NewReader(make([]byte, 5))); code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if out.Len() != 0 {
		t.Fatalf("printed output on failure: %q", out.String())
	}
}

func TestKeygenWriteFailure(t *testing.T) {
	if code := cmdKeygen(failWriter{}, bytes.NewReader(make([]byte, 32))); code == 0 {
		t.Fatal("exit 0 despite stdout write failure")
	}
}
