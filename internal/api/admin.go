package api

import (
	"cmp"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/policy"
)

const (
	// adminActor is recorded as the actor of every activation made through
	// the admin API.
	adminActor = "admin"
	// maxPolicyBytes bounds an uploaded policy document.
	maxPolicyBytes = 8 << 20
)

// authorizeAdmin admits a request that carries an admin key to the policy
// routes. Without admin keys or a policy registry these routes do not exist,
// so they answer 404 whatever the request carries. A missing or wrong key is
// 401.
func (s *Server) authorizeAdmin(r *http.Request) *Error {
	if s.policies == nil {
		return errAdminNotFound()
	}
	return s.checkAdminKey(r)
}

// checkAdminKey admits a request that carries an admin key. Without admin keys
// the admin routes do not exist and answer 404.
func (s *Server) checkAdminKey(r *http.Request) *Error {
	if len(s.adminKeys) == 0 {
		return errAdminNotFound()
	}
	if token, ok := auth.BearerToken(r.Header); ok {
		if _, found := s.adminKeys[auth.HashKey(token)]; found {
			return nil
		}
	}
	return &Error{Status: http.StatusUnauthorized, Code: CodeInvalidAPIKey, Message: "invalid API key"}
}

func errAdminNotFound() *Error {
	return &Error{Status: http.StatusNotFound, Code: CodeNotFound, Message: "not found"}
}

type adminPolicy struct {
	Version       string    `json:"version"`
	CreatedAt     string    `json:"created_at"`
	AddedAt       time.Time `json:"added_at"`
	Status        string    `json:"status"`
	PredictorType string    `json:"predictor_type"`
}

type adminActivation struct {
	Version  string    `json:"version"`
	Previous *string   `json:"previous"` // null for the first activation
	Actor    string    `json:"actor"`
	At       time.Time `json:"at"`
}

type adminPolicyList struct {
	Active      *string           `json:"active"` // null before the first activation
	Policies    []adminPolicy     `json:"policies"`
	Activations []adminActivation `json:"activations"`
}

func activationBody(a policy.Activation) adminActivation {
	out := adminActivation{Version: a.Version, Actor: a.Actor, At: a.At}
	if a.Previous != "" {
		out.Previous = &a.Previous
	}
	return out
}

// handleAdminList lists the stored policies and the activation log.
func (s *Server) handleAdminList(w http.ResponseWriter, r *http.Request) {
	if aerr := s.authorizeAdmin(r); aerr != nil {
		writeError(w, aerr)
		return
	}
	entries := s.policies.List()
	acts := s.policies.Activations()
	list := adminPolicyList{
		Policies:    make([]adminPolicy, len(entries)),
		Activations: make([]adminActivation, len(acts)),
	}
	if active := s.policies.Active(); active != nil {
		list.Active = &active.Version
	}
	for i, e := range entries {
		list.Policies[i] = adminPolicy{
			Version:       e.Policy.Version,
			CreatedAt:     e.Policy.CreatedAt,
			AddedAt:       e.AddedAt,
			Status:        e.Status,
			PredictorType: e.Policy.PredictorType,
		}
	}
	for i, a := range acts {
		list.Activations[i] = activationBody(a)
	}
	s.writeJSON(w, r, http.StatusOK, list)
}

// handleAdminUpload validates a policy document and stores it, inactive.
func (s *Server) handleAdminUpload(w http.ResponseWriter, r *http.Request) {
	if aerr := s.authorizeAdmin(r); aerr != nil {
		writeError(w, aerr)
		return
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxPolicyBytes))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, &Error{Status: http.StatusRequestEntityTooLarge, Code: CodeRequestTooLarge, Message: "policy document is too large"})
			return
		}
		writeError(w, invalid("failed to read request body"))
		return
	}
	pol, err := policy.Load(data)
	if err != nil {
		writeError(w, invalid("%v", err))
		return
	}
	if err := s.policies.Add(pol); err != nil {
		var ve *policy.ValidationError
		if errors.As(err, &ve) {
			writeError(w, invalid("%v", err))
			return
		}
		s.logger.ErrorContext(r.Context(), "add policy failed", "request_id", infoFrom(r.Context()).requestID, "error", err)
		if errors.Is(err, policy.ErrPersist) {
			writeError(w, errStorageUnavailable())
			return
		}
		writeError(w, &Error{Status: http.StatusInternalServerError, Code: CodeInternalError, Message: "the policy could not be stored"})
		return
	}
	s.writeJSON(w, r, http.StatusCreated, map[string]string{"version": pol.Version})
}

// handleAdminActivate makes a stored policy the active one.
func (s *Server) handleAdminActivate(w http.ResponseWriter, r *http.Request) {
	if aerr := s.authorizeAdmin(r); aerr != nil {
		writeError(w, aerr)
		return
	}
	act, err := s.policies.Activate(r.PathValue("version"), adminActor)
	if err != nil {
		// The registry knows every version it can activate, so an unknown version the
		// policy store reports is the store's failure, not the client's mistake.
		if errors.Is(err, policy.ErrUnknownVersion) && !errors.Is(err, policy.ErrPersist) {
			writeError(w, &Error{Status: http.StatusNotFound, Code: CodeNotFound, Message: "unknown policy version"})
			return
		}
		s.writeAdminFailure(w, r, "activate policy failed", err)
		return
	}
	s.logActivation(r, act)
	s.writeJSON(w, r, http.StatusOK, activationBody(act))
}

// handleAdminRollback re-activates the policy that was active before the
// current one.
func (s *Server) handleAdminRollback(w http.ResponseWriter, r *http.Request) {
	if aerr := s.authorizeAdmin(r); aerr != nil {
		writeError(w, aerr)
		return
	}
	act, err := s.policies.Rollback(adminActor)
	if err != nil {
		if errors.Is(err, policy.ErrNoPrevious) {
			writeError(w, &Error{Status: http.StatusConflict, Code: CodeNoPreviousPolicy, Message: "there is no previous policy to roll back to"})
			return
		}
		s.writeAdminFailure(w, r, "roll back policy failed", err)
		return
	}
	s.logActivation(r, act)
	s.writeJSON(w, r, http.StatusOK, activationBody(act))
}

// writeAdminFailure logs an unexpected registry error and answers without its
// text: 503 when the policy store did not take the change, 500 otherwise.
func (s *Server) writeAdminFailure(w http.ResponseWriter, r *http.Request, msg string, err error) {
	s.logger.ErrorContext(r.Context(), msg, "request_id", infoFrom(r.Context()).requestID, "error", err)
	if errors.Is(err, policy.ErrPersist) {
		writeError(w, errStorageUnavailable())
		return
	}
	writeError(w, &Error{Status: http.StatusInternalServerError, Code: CodeInternalError, Message: "the policy registry failed"})
}

// logActivation records a change of the active policy.
func (s *Server) logActivation(r *http.Request, a policy.Activation) {
	s.logger.InfoContext(r.Context(), "policy activated",
		"request_id", infoFrom(r.Context()).requestID, "version", a.Version, "previous", a.Previous, "actor", a.Actor)
}

// adminAccounting is the answer of GET /admin/accounting: what a tenant spent,
// holds in reservations and went over its budget by on a UTC day, and what its
// usage records add up to. The benchmark checks the accounting invariants
// through it.
type adminAccounting struct {
	Tenant         string  `json:"tenant"`
	Day            string  `json:"day"`
	SpentUSD       float64 `json:"spent_usd"`
	ReservedUSD    float64 `json:"reserved_usd"`
	OvershootUSD   float64 `json:"overshoot_usd"`
	SpentNanos     int64   `json:"spent_nanos"`
	ReservedNanos  int64   `json:"reserved_nanos"`
	OvershootNanos int64   `json:"overshoot_nanos"`
	UsageRecords   int     `json:"usage_records"`
	UsageCostNanos int64   `json:"usage_cost_nanos"`
}

// handleAdminAccounting reports a tenant's accounting for one day. The tenant
// is required and need not exist: an unknown one has spent nothing. The day is
// a yyyymmdd UTC date and defaults to today.
func (s *Server) handleAdminAccounting(w http.ResponseWriter, r *http.Request) {
	if s.acct == nil {
		writeError(w, errAdminNotFound())
		return
	}
	if aerr := s.checkAdminKey(r); aerr != nil {
		writeError(w, aerr)
		return
	}
	q := r.URL.Query()
	tenant := q.Get("tenant")
	if tenant == "" {
		writeError(w, invalid("the tenant query parameter is required"))
		return
	}
	day := cmp.Or(q.Get("day"), accounting.DayKey(s.now()))
	if _, err := time.Parse("20060102", day); err != nil {
		writeError(w, invalid("day must be a UTC date written yyyymmdd"))
		return
	}

	budget, usage, err := s.acct.State(r.Context(), tenant, day)
	if err != nil {
		s.logger.ErrorContext(r.Context(), "read accounting state failed", "request_id", infoFrom(r.Context()).requestID, "error", err)
		writeError(w, errAccountingUnavailable())
		return
	}
	s.writeJSON(w, r, http.StatusOK, adminAccounting{
		Tenant:         tenant,
		Day:            day,
		SpentUSD:       accounting.NanosToUSD(budget.Spent),
		ReservedUSD:    accounting.NanosToUSD(budget.Reserved),
		OvershootUSD:   accounting.NanosToUSD(budget.Overshoot),
		SpentNanos:     budget.Spent,
		ReservedNanos:  budget.Reserved,
		OvershootNanos: budget.Overshoot,
		UsageRecords:   usage.Records,
		UsageCostNanos: usage.CostNanos,
	})
}
