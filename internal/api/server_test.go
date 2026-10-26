package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestModelsList(t *testing.T) {
	srv := newServer(t, testConfig(), nil)
	h := srv.Handler()

	ids := func(key string) []string {
		t.Helper()
		rec := do(t, h, http.MethodGet, "/v1/models", key, "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
		var got struct {
			Object string `json:"object"`
			Data   []struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				OwnedBy string `json:"owned_by"`
			} `json:"data"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("body is not JSON: %v", err)
		}
		if got.Object != "list" {
			t.Errorf("object = %q, want list", got.Object)
		}
		out := make([]string, len(got.Data))
		for i, m := range got.Data {
			out[i] = m.ID
			if m.Object != "model" || m.Created != 0 || m.OwnedBy != "routeforge" {
				t.Errorf("entry %d = %+v, want object model, created 0, owned_by routeforge", i, m)
			}
		}
		return out
	}

	if got, want := ids(keyA), []string{"routeforge-auto", "m-fast", "m-strong"}; !reflect.DeepEqual(got, want) {
		t.Errorf("tenant-a models = %v, want %v", got, want)
	}
	if got, want := ids(keyB), []string{"routeforge-auto", "m-fast"}; !reflect.DeepEqual(got, want) {
		t.Errorf("tenant-b models = %v, want %v", got, want)
	}

	rec := do(t, h, http.MethodGet, "/v1/models", "", "", nil)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("no key: status = %d, want 401", rec.Code)
	}
	if code, _ := errorFields(t, rec); code != CodeInvalidAPIKey {
		t.Errorf("no key: code = %q, want %q", code, CodeInvalidAPIKey)
	}
}

func TestHealthz(t *testing.T) {
	rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/healthz", "", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got := strings.TrimSpace(rec.Body.String()); got != `{"status":"ok"}` {
		t.Errorf("body = %q, want {\"status\":\"ok\"}", got)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestReadyz(t *testing.T) {
	assertJSON := func(t *testing.T, rec *httptest.ResponseRecorder, wantCode int, want string) {
		t.Helper()
		body := rec.Body.Bytes()
		if rec.Code != wantCode {
			t.Fatalf("status = %d, want %d; body %s", rec.Code, wantCode, body)
		}
		var got, exp any
		if err := json.Unmarshal(body, &got); err != nil {
			t.Fatalf("body is not JSON: %v", err)
		}
		if err := json.Unmarshal([]byte(want), &exp); err != nil {
			t.Fatalf("bad expectation: %v", err)
		}
		if !reflect.DeepEqual(got, exp) {
			t.Errorf("body = %s, want %s", body, want)
		}
	}

	t.Run("no checks", func(t *testing.T) {
		rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusOK, `{"status":"ready"}`)
	})

	t.Run("failing check", func(t *testing.T) {
		srv := newServer(t, testConfig(), nil, func(o *Options) {
			o.ReadyChecks = []ReadyCheck{{
				Name:  "policy",
				Check: func(context.Context) error { return errors.New("not loaded") },
			}}
		})
		rec := do(t, srv.Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusServiceUnavailable, `{"status":"not_ready","failed":{"policy":"not loaded"}}`)
	})

	t.Run("draining", func(t *testing.T) {
		srv := newServer(t, testConfig(), nil)
		srv.SetDraining()
		rec := do(t, srv.Handler(), http.MethodGet, "/readyz", "", "", nil)
		assertJSON(t, rec, http.StatusServiceUnavailable, `{"status":"draining"}`)
	})
}

func TestNewHTTPServerLimits(t *testing.T) {
	c := testConfig().Server
	hs := NewHTTPServer(c, http.NewServeMux())
	if hs.Addr != ":8080" {
		t.Errorf("Addr = %q, want :8080", hs.Addr)
	}
	if hs.ReadHeaderTimeout != 5*time.Second {
		t.Errorf("ReadHeaderTimeout = %v, want 5s", hs.ReadHeaderTimeout)
	}
	if hs.IdleTimeout != 120*time.Second {
		t.Errorf("IdleTimeout = %v, want 120s", hs.IdleTimeout)
	}
	if hs.MaxHeaderBytes != 1048576 {
		t.Errorf("MaxHeaderBytes = %d, want 1048576", hs.MaxHeaderBytes)
	}
	if hs.ReadTimeout != 0 || hs.WriteTimeout != 0 {
		t.Errorf("ReadTimeout = %v, WriteTimeout = %v, want both 0", hs.ReadTimeout, hs.WriteTimeout)
	}
}

func TestMethodNotAllowed(t *testing.T) {
	rec := do(t, newServer(t, testConfig(), nil).Handler(), http.MethodGet, "/v1/chat/completions", keyA, "", nil)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}
