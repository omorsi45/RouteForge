package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/policy"
)

const adminKey = "test-admin-key"

// committedPolicyFile is the trained policy the trainer commits.
const committedPolicyFile = "../../trainer/artifacts/policy.json"

// smallTrees is how many trees per model the test policies keep. The policy
// is still valid and loads in a few milliseconds.
const smallTrees = 3

// smallPolicyJSON is the committed policy cut down to its first trees.
var smallPolicyJSON = sync.OnceValues(func() ([]byte, error) {
	data, err := os.ReadFile(committedPolicyFile)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber() // keep number text, which the version hash covers
	var tree map[string]any
	if err := dec.Decode(&tree); err != nil {
		return nil, err
	}
	models, _ := tree["models"].([]any)
	for i, raw := range models {
		model, _ := raw.(map[string]any)
		predictor, _ := model["predictor"].(map[string]any)
		trees, _ := predictor["trees"].([]any)
		if len(trees) < smallTrees {
			return nil, fmt.Errorf("committed policy: models[%d] has %d trees, want at least %d", i, len(trees), smallTrees)
		}
		predictor["trees"] = trees[:smallTrees]
	}
	return json.Marshal(tree)
})

// policyTree decodes a fresh copy of the small policy for editing.
func policyTree(t *testing.T) map[string]any {
	t.Helper()
	data, err := smallPolicyJSON()
	if err != nil {
		t.Fatalf("build small policy: %v", err)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var tree map[string]any
	if err := dec.Decode(&tree); err != nil {
		t.Fatalf("decode small policy: %v", err)
	}
	return tree
}

// renderPolicy marshals tree with a recomputed version, so the only problem
// left in the document is one a test planted.
func renderPolicy(t *testing.T, tree map[string]any) []byte {
	t.Helper()
	tree["version"] = "000000000000"
	data, err := json.Marshal(tree)
	if err != nil {
		t.Fatalf("marshal policy: %v", err)
	}
	version, err := policy.CanonicalVersion(data)
	if err != nil {
		t.Fatalf("CanonicalVersion: %v", err)
	}
	tree["version"] = version
	if data, err = json.Marshal(tree); err != nil {
		t.Fatalf("marshal policy: %v", err)
	}
	return data
}

// firstModel returns models[0] of a policy tree.
func firstModel(t *testing.T, tree map[string]any) map[string]any {
	t.Helper()
	models, _ := tree["models"].([]any)
	model, ok := models[0].(map[string]any)
	if !ok {
		t.Fatalf("models[0] is %T, want an object", models[0])
	}
	return model
}

// policyJSON returns the small policy after mutate has edited its tree.
func policyJSON(t *testing.T, mutate func(tree map[string]any)) []byte {
	t.Helper()
	tree := policyTree(t)
	if mutate != nil {
		mutate(tree)
	}
	return renderPolicy(t, tree)
}

// policyPairJSON returns two valid policy documents with different versions.
func policyPairJSON(t *testing.T) (a, b []byte) {
	t.Helper()
	a = policyJSON(t, nil)
	b = policyJSON(t, func(tree map[string]any) {
		predictor, _ := firstModel(t, tree)["predictor"].(map[string]any)
		predictor["init"] = json.Number("0.123")
	})
	return a, b
}

func mustLoadPolicy(t *testing.T, data []byte) *policy.Policy {
	t.Helper()
	p, err := policy.Load(data)
	if err != nil {
		t.Fatalf("policy.Load: %v", err)
	}
	return p
}

// newAdminEnv is a pool server with an admin key and policy a active in a
// registry that accepts the pool models.
func newAdminEnv(t *testing.T, a *policy.Policy, tweaks ...func(*poolEnv)) *poolEnv {
	t.Helper()
	all := append([]func(*poolEnv){func(e *poolEnv) {
		e.cfg.Admin.APIKeySHA256 = []string{auth.HashKey(adminKey)}
		e.reg = policy.NewRegistry(poolModels, nil)
		activate(t, e.reg, a)
	}}, tweaks...)
	return newPoolEnv(t, all...)
}

// admin sends one request with the admin key.
func (e *poolEnv) admin(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, e.handler(), method, path, adminKey, body, nil)
}

type adminActivationBody struct {
	Version  string    `json:"version"`
	Previous *string   `json:"previous"`
	Actor    string    `json:"actor"`
	At       time.Time `json:"at"`
}

type adminListBody struct {
	Active   *string `json:"active"`
	Policies []struct {
		Version       string    `json:"version"`
		CreatedAt     string    `json:"created_at"`
		AddedAt       time.Time `json:"added_at"`
		Status        string    `json:"status"`
		PredictorType string    `json:"predictor_type"`
	} `json:"policies"`
	Activations []adminActivationBody `json:"activations"`
}

func decodeBody[T any](t *testing.T, rec *httptest.ResponseRecorder, wantStatus int) T {
	t.Helper()
	var v T
	if rec.Code != wantStatus {
		t.Fatalf("status = %d, want %d; body %s", rec.Code, wantStatus, rec.Body)
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatalf("body is not the expected JSON: %v; body %s", err, rec.Body)
	}
	return v
}

// chatPolicyVersion sends one auto chat request and returns the policy version
// header of the response.
func (e *poolEnv) chatPolicyVersion(t *testing.T) string {
	t.Helper()
	rec := e.post(t, "/v1/chat/completions", keyA, autoBody, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200; body %s", rec.Code, rec.Body)
	}
	return rec.Header().Get("X-RouteForge-Policy-Version")
}

// adminRoutes are the four admin routes.
var adminRoutes = []struct{ method, path string }{
	{http.MethodGet, "/admin/policies"},
	{http.MethodPost, "/admin/policies"},
	{http.MethodPost, "/admin/policies/aaaaaaaaaaaa/activate"},
	{http.MethodPost, "/admin/policies/rollback"},
}

func TestAdminRequiresKey(t *testing.T) {
	a := mustLoadPolicy(t, policyJSON(t, nil))

	t.Run("wrong or missing key is 401", func(t *testing.T) {
		env := newAdminEnv(t, a)
		for _, key := range []string{"", "test-admin-wrong", keyA} {
			for _, r := range adminRoutes {
				rec := do(t, env.handler(), r.method, r.path, key, "", nil)
				wantError(t, rec, http.StatusUnauthorized, CodeInvalidAPIKey)
			}
		}
	})

	t.Run("without admin config every admin route is 404", func(t *testing.T) {
		env := newPoolEnv(t) // no admin keys
		for _, key := range []string{"", adminKey, keyA} {
			for _, r := range adminRoutes {
				rec := do(t, env.handler(), r.method, r.path, key, "", nil)
				wantError(t, rec, http.StatusNotFound, CodeNotFound)
			}
		}
	})

	t.Run("without a registry every admin route is 404", func(t *testing.T) {
		env := newAdminEnv(t, a, func(e *poolEnv) {
			e.opts = append(e.opts, func(o *Options) { o.Policies = nil })
		})
		for _, r := range adminRoutes {
			wantError(t, env.admin(t, r.method, r.path, ""), http.StatusNotFound, CodeNotFound)
		}
	})

	t.Run("the admin key is not a tenant key", func(t *testing.T) {
		env := newAdminEnv(t, a)
		wantError(t, env.post(t, "/v1/chat/completions", adminKey, autoBody, nil), http.StatusUnauthorized, CodeInvalidAPIKey)
		wantError(t, env.get(t, "/v1/models", adminKey), http.StatusUnauthorized, CodeInvalidAPIKey)
	})
}

func TestAdminList(t *testing.T) {
	aJSON, bJSON := policyPairJSON(t)
	a, b := mustLoadPolicy(t, aJSON), mustLoadPolicy(t, bJSON)

	t.Run("empty registry", func(t *testing.T) {
		env := newAdminEnv(t, a, func(e *poolEnv) { e.reg = policy.NewRegistry(poolModels, nil) })
		rec := env.admin(t, http.MethodGet, "/admin/policies", "")
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body %s", rec.Code, rec.Body)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", ct)
		}
		// null and [] are part of the contract, not omitted.
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
			t.Fatal(err)
		}
		for key, want := range map[string]string{"active": "null", "policies": "[]", "activations": "[]"} {
			if got := string(raw[key]); got != want {
				t.Errorf("%s = %s, want %s", key, got, want)
			}
		}
	})

	t.Run("policies and activations", func(t *testing.T) {
		env := newAdminEnv(t, a)
		if err := env.reg.Add(b); err != nil {
			t.Fatal(err)
		}
		list := decodeBody[adminListBody](t, env.admin(t, http.MethodGet, "/admin/policies", ""), http.StatusOK)

		if list.Active == nil || *list.Active != a.Version {
			t.Errorf("active = %v, want %s", list.Active, a.Version)
		}
		if len(list.Policies) != 2 {
			t.Fatalf("got %d policies, want 2", len(list.Policies))
		}
		for i, want := range []struct {
			p      *policy.Policy
			status string
		}{{a, policy.StatusActive}, {b, policy.StatusInactive}} {
			got := list.Policies[i]
			if got.Version != want.p.Version || got.Status != want.status || got.PredictorType != "gbm" ||
				got.CreatedAt != want.p.CreatedAt || got.AddedAt.IsZero() {
				t.Errorf("policies[%d] = %+v, want version %s status %s predictor gbm created_at %s", i, got, want.p.Version, want.status, want.p.CreatedAt)
			}
		}
		if len(list.Activations) != 1 {
			t.Fatalf("got %d activations, want 1", len(list.Activations))
		}
		if act := list.Activations[0]; act.Version != a.Version || act.Previous != nil || act.Actor != "test" || act.At.IsZero() {
			t.Errorf("activations[0] = %+v, want version %s, no previous, actor test", act, a.Version)
		}
	})
}

func TestAdminUploadActivateRollback(t *testing.T) {
	aJSON, bJSON := policyPairJSON(t)
	a, b := mustLoadPolicy(t, aJSON), mustLoadPolicy(t, bJSON)
	env := newAdminEnv(t, a)

	if got := env.chatPolicyVersion(t); got != a.Version {
		t.Fatalf("chat before upload used %q, want %s", got, a.Version)
	}

	// Upload B: stored, not active.
	rec := env.admin(t, http.MethodPost, "/admin/policies", string(bJSON))
	up := decodeBody[struct {
		Version string `json:"version"`
	}](t, rec, http.StatusCreated)
	if up.Version != b.Version {
		t.Fatalf("upload returned version %q, want %s", up.Version, b.Version)
	}
	if got := env.chatPolicyVersion(t); got != a.Version {
		t.Errorf("chat after upload used %q, want %s (upload must not activate)", got, a.Version)
	}

	// Activate B.
	rec = env.admin(t, http.MethodPost, "/admin/policies/"+b.Version+"/activate", "")
	act := decodeBody[adminActivationBody](t, rec, http.StatusOK)
	if act.Version != b.Version || act.Previous == nil || *act.Previous != a.Version || act.Actor != "admin" || act.At.IsZero() {
		t.Errorf("activation = %+v, want %s after %s by admin", act, b.Version, a.Version)
	}
	if got := env.chatPolicyVersion(t); got != b.Version {
		t.Errorf("chat after activation used %q, want %s", got, b.Version)
	}

	// Roll back to A.
	rec = env.admin(t, http.MethodPost, "/admin/policies/rollback", "")
	act = decodeBody[adminActivationBody](t, rec, http.StatusOK)
	if act.Version != a.Version || act.Previous == nil || *act.Previous != b.Version || act.Actor != "admin" {
		t.Errorf("rollback = %+v, want %s after %s by admin", act, a.Version, b.Version)
	}
	if got := env.chatPolicyVersion(t); got != a.Version {
		t.Errorf("chat after rollback used %q, want %s", got, a.Version)
	}

	list := decodeBody[adminListBody](t, env.admin(t, http.MethodGet, "/admin/policies", ""), http.StatusOK)
	if list.Active == nil || *list.Active != a.Version {
		t.Errorf("active = %v, want %s", list.Active, a.Version)
	}
	if len(list.Activations) != 3 || list.Activations[1].Actor != "admin" || list.Activations[2].Actor != "admin" {
		t.Errorf("activations = %+v, want the startup one then two by admin", list.Activations)
	}
}

func TestAdminReuploadKeepsStatus(t *testing.T) {
	aJSON, _ := policyPairJSON(t)
	a := mustLoadPolicy(t, aJSON)
	env := newAdminEnv(t, a)

	rec := env.admin(t, http.MethodPost, "/admin/policies", string(aJSON))
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body %s", rec.Code, rec.Body)
	}
	list := decodeBody[adminListBody](t, env.admin(t, http.MethodGet, "/admin/policies", ""), http.StatusOK)
	if len(list.Policies) != 1 || list.Policies[0].Status != policy.StatusActive {
		t.Errorf("policies = %+v, want the one active policy unchanged", list.Policies)
	}
}

func TestAdminUploadInvalid(t *testing.T) {
	a := mustLoadPolicy(t, policyJSON(t, nil))

	cases := []struct {
		name     string
		body     string
		wantText string
	}{
		{"first block starts above zero", string(policyJSON(t, func(tree map[string]any) {
			cal, _ := firstModel(t, tree)["calibration"].(map[string]any)
			blocks, _ := cal["blocks"].([]any)
			block, _ := blocks[0].(map[string]any)
			block["lo"] = json.Number("0.5")
		})), "blocks"},
		{"version does not match the content", strings.Replace(string(policyJSON(t, nil)), a.Version, "000000000000", 1), "version"},
		{"not JSON", "{", "invalid JSON"},
		{"empty body", "", "invalid JSON"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := newAdminEnv(t, a)
			rec := env.admin(t, http.MethodPost, "/admin/policies", tc.body)
			msg := wantError(t, rec, http.StatusBadRequest, CodeInvalidRequest)
			if !strings.Contains(msg, tc.wantText) {
				t.Errorf("message %q does not contain %q", msg, tc.wantText)
			}
			if n := len(env.reg.List()); n != 1 {
				t.Errorf("registry holds %d policies after a rejected upload, want 1", n)
			}
		})
	}
}

func TestAdminUploadUnmappedModel(t *testing.T) {
	aJSON, _ := policyPairJSON(t)
	// No configured model has the policy name of the last pool model.
	env := newAdminEnv(t, mustLoadPolicy(t, aJSON), func(e *poolEnv) {
		e.cfg.Models = e.cfg.Models[:5]
		e.reg = policy.NewRegistry(poolModels[:5], nil)
	})
	rec := env.admin(t, http.MethodPost, "/admin/policies", string(aJSON))

	msg := wantError(t, rec, http.StatusBadRequest, CodeInvalidRequest)
	if !strings.Contains(msg, "qwen3-235b-a22b-2507") {
		t.Errorf("message %q does not name the unmapped model", msg)
	}
	if n := len(env.reg.List()); n != 0 {
		t.Errorf("registry holds %d policies after a rejected upload, want 0", n)
	}
}

func TestAdminUploadTooLarge(t *testing.T) {
	a := mustLoadPolicy(t, policyJSON(t, nil))
	env := newAdminEnv(t, a)
	rec := env.admin(t, http.MethodPost, "/admin/policies", strings.Repeat(" ", 8<<20+1))
	wantError(t, rec, http.StatusRequestEntityTooLarge, CodeRequestTooLarge)
}

func TestAdminActivateUnknownVersion(t *testing.T) {
	a := mustLoadPolicy(t, policyJSON(t, nil))
	env := newAdminEnv(t, a)
	rec := env.admin(t, http.MethodPost, "/admin/policies/ffffffffffff/activate", "")

	wantError(t, rec, http.StatusNotFound, CodeNotFound)
	if got := env.chatPolicyVersion(t); got != a.Version {
		t.Errorf("chat used %q after a failed activation, want %s", got, a.Version)
	}
}

func TestAdminRollbackWithoutPrevious(t *testing.T) {
	a := mustLoadPolicy(t, policyJSON(t, nil))
	env := newAdminEnv(t, a)
	rec := env.admin(t, http.MethodPost, "/admin/policies/rollback", "")

	wantError(t, rec, http.StatusConflict, CodeNoPreviousPolicy)
	if got := env.chatPolicyVersion(t); got != a.Version {
		t.Errorf("chat used %q after a failed rollback, want %s", got, a.Version)
	}
}
