package api

import (
	"net/http"

	"github.com/omorsi45/routeforge/internal/router"
)

// handleExplain shows how a request would be routed. It reads the request the
// way chat does, with the same authentication, headers and body rules, and
// runs the router, but it never calls a provider and never stores the
// decision: the decision is returned, not recorded.
func (s *Server) handleExplain(w http.ResponseWriter, r *http.Request) {
	info := infoFrom(r.Context())
	call, aerr := s.readChat(w, r)
	if aerr != nil {
		writeError(w, aerr)
		return
	}
	defer call.cancel()

	d, err := s.router.Decide(call.ctx, call.rreq)
	if d != nil {
		info.decisionID = d.ID
		info.model = d.Chosen
	}
	if err != nil {
		// Per the Router contract d is set only for no_eligible_model and
		// overloaded, where the candidates' reasons explain the failure.
		writeErrorWith(w, fromRouterError(err), d)
		return
	}
	s.writeJSON(w, r, http.StatusOK, d)
}

// handleDecision returns a stored decision to the tenant it belongs to. A
// decision of another tenant and an unknown id get the same answer.
func (s *Server) handleDecision(w http.ResponseWriter, r *http.Request) {
	tenant, aerr := s.authenticate(r)
	if aerr != nil {
		writeError(w, aerr)
		return
	}
	d, found, err := s.decisions.Get(r.Context(), tenant.Name, r.PathValue("id"))
	if err != nil {
		s.logger.ErrorContext(r.Context(), "read decision failed", "request_id", infoFrom(r.Context()).requestID, "error", err)
		writeError(w, &Error{Status: http.StatusInternalServerError, Code: CodeInternalError, Message: "the decision could not be read"})
		return
	}
	if !found {
		writeError(w, &Error{Status: http.StatusNotFound, Code: CodeNotFound, Message: "decision not found"})
		return
	}
	s.writeJSON(w, r, http.StatusOK, d)
}

// storeDecision keeps d for GET /v1/routing/decisions/{id}. The response is
// already decided by then, so a failure is logged, not sent.
func (s *Server) storeDecision(r *http.Request, d *router.Decision) {
	if err := s.decisions.Put(d); err != nil {
		s.logger.WarnContext(r.Context(), "store decision failed",
			"request_id", infoFrom(r.Context()).requestID, "decision_id", d.ID, "error", err)
	}
}
