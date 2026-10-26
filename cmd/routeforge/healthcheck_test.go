package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHealthcheckOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	if code := cmdHealthcheck(context.Background(), []string{"--url", srv.URL}, &out, &errb); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errb.String())
	}
	if out.String() != "ok\n" {
		t.Fatalf("stdout %q, want %q", out.String(), "ok\n")
	}
}

func TestHealthcheckUnhealthy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	if code := cmdHealthcheck(context.Background(), []string{"--url", srv.URL}, &out, &errb); code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if !strings.Contains(errb.String(), "503") {
		t.Fatalf("stderr %q does not mention 503", errb.String())
	}
	if out.Len() != 0 {
		t.Fatalf("stdout %q, want empty", out.String())
	}
}

func TestHealthcheckUnreachable(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close()

	var out, errb bytes.Buffer
	start := time.Now()
	code := cmdHealthcheck(context.Background(), []string{"--url", url, "--timeout", "200ms"}, &out, &errb)
	if code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("took %s, want under 1s", elapsed)
	}
	if errb.Len() == 0 {
		t.Fatal("stderr is empty, want the reason")
	}
}
