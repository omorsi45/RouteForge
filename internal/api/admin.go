package api

import (
	"errors"
	"io"
	"net/http"
	"time"

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

// authorizeAdmin admits a request that carries an admin key. Without admin
// keys or a policy registry the admin routes do not exist, so they answer
// 404 whatever the request carries. A missing or wrong key is 401.
func (s *Server) authorizeAdmin(r *http.Request) *Error {
	if len(s.adminKeys) == 0 || s.policies == nil {
		return &Error{Status: http.StatusNotFound, Code: CodeNotFound, Message: "not found"}
	}
	if token, ok := auth.BearerToken(r.Header); ok {
		if _, found := s.adminKeys[auth.HashKey(token)]; found {
			return nil
		}
	}
	return &Error{Status: http.StatusUnauthorized, Code: CodeInvalidAPIKey, Message: "invalid API key"}
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
		if errors.Is(err, policy.ErrUnknownVersion) {
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

// writeAdminFailure logs an unexpected registry error and answers 500 without
// its text.
func (s *Server) writeAdminFailure(w http.ResponseWriter, r *http.Request, msg string, err error) {
	s.logger.ErrorContext(r.Context(), msg, "request_id", infoFrom(r.Context()).requestID, "error", err)
	writeError(w, &Error{Status: http.StatusInternalServerError, Code: CodeInternalError, Message: "the policy registry failed"})
}

// logActivation records a change of the active policy.
func (s *Server) logActivation(r *http.Request, a policy.Activation) {
	s.logger.InfoContext(r.Context(), "policy activated",
		"request_id", infoFrom(r.Context()).requestID, "version", a.Version, "previous", a.Previous, "actor", a.Actor)
}
