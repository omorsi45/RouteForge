package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"regexp"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routeforge/internal/accounting"
	"github.com/omorsi45/routeforge/internal/auth"
	"github.com/omorsi45/routeforge/internal/config"
	"github.com/omorsi45/routeforge/internal/executor"
	"github.com/omorsi45/routeforge/internal/policy"
	"github.com/omorsi45/routeforge/internal/reliability"
	"github.com/omorsi45/routeforge/internal/router"
	"github.com/omorsi45/routeforge/internal/store"
)

// readyCheckTimeout bounds each readiness check.
const readyCheckTimeout = 2 * time.Second

// defaultDecisionCapacity is how many decisions the default store keeps.
const defaultDecisionCapacity = 10000

var requestIDRE = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)

// ReadyCheck is one readiness probe. A non-nil error marks the gateway not
// ready.
type ReadyCheck struct {
	Name  string
	Check func(context.Context) error // M3 adds "policy", M5 adds "storage" and "limiter"
}

// Options configures a Server.
type Options struct {
	Config      *config.Config
	Keys        *auth.Store
	Router      router.Router
	Executor    *executor.Executor
	Accountant  *accounting.Accountant // admits and meters chat requests; nil means no rate limits, budgets or usage records
	Decisions   store.DecisionStore    // nil: a ring of 10,000 decisions
	Policies    *policy.Registry       // the registry the router reads; may be nil, and then the policy admin routes do not exist
	InFlight    *reliability.InFlight  // limits concurrent chat and explain requests; nil means no limit
	Logger      *slog.Logger           // nil: slog.New(slog.DiscardHandler)
	Now         func() time.Time       // nil: time.Now
	ReadyChecks []ReadyCheck
}

// Server is the gateway's HTTP surface.
type Server struct {
	cfg       *config.Config
	keys      *auth.Store
	adminKeys map[string]struct{} // digests of the admin keys
	router    router.Router
	exec      *executor.Executor
	acct      *accounting.Accountant
	decisions store.DecisionStore
	policies  *policy.Registry
	inflight  *reliability.InFlight
	logger    *slog.Logger
	now       func() time.Time
	checks    []ReadyCheck
	draining  atomic.Bool
}

// New returns a Server over o.
func New(o Options) *Server {
	s := &Server{
		cfg:       o.Config,
		keys:      o.Keys,
		adminKeys: make(map[string]struct{}, len(o.Config.Admin.APIKeySHA256)),
		router:    o.Router,
		exec:      o.Executor,
		acct:      o.Accountant,
		decisions: o.Decisions,
		policies:  o.Policies,
		inflight:  o.InFlight,
		logger:    o.Logger,
		now:       o.Now,
		checks:    o.ReadyChecks,
	}
	for _, digest := range o.Config.Admin.APIKeySHA256 {
		s.adminKeys[digest] = struct{}{}
	}
	if s.decisions == nil {
		s.decisions = store.NewRing(defaultDecisionCapacity)
	}
	if s.logger == nil {
		s.logger = slog.New(slog.DiscardHandler)
	}
	if s.now == nil {
		s.now = time.Now
	}
	return s
}

// Handler returns the routes wrapped by the request-id and access-log
// middleware.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/routing/explain", s.handleExplain)
	mux.HandleFunc("GET /v1/routing/decisions/{id}", s.handleDecision)
	mux.HandleFunc("GET /admin/policies", s.handleAdminList)
	mux.HandleFunc("POST /admin/policies", s.handleAdminUpload)
	mux.HandleFunc("POST /admin/policies/rollback", s.handleAdminRollback)
	mux.HandleFunc("POST /admin/policies/{version}/activate", s.handleAdminActivate)
	mux.HandleFunc("GET /admin/accounting", s.handleAdminAccounting)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /readyz", s.handleReadyz)
	return s.withRequestID(s.withAccessLog(mux))
}

// SetDraining makes /readyz return 503 from now on.
func (s *Server) SetDraining() { s.draining.Store(true) }

// NewHTTPServer returns an http.Server for h. There is no global read or write
// timeout: handlers set per-request deadlines, because streams outlive any
// fixed value.
func NewHTTPServer(c config.Server, h http.Handler) *http.Server {
	return &http.Server{
		Addr:              c.Listen,
		Handler:           h,
		ReadHeaderTimeout: c.ReadHeaderTimeout,
		IdleTimeout:       c.IdleTimeout,
		MaxHeaderBytes:    c.MaxHeaderBytes,
	}
}

// reqInfo carries per-request values from the handlers back to the access log.
type reqInfo struct {
	requestID  string
	tenant     string
	model      string
	decisionID string
	// streamOpen is set while an event stream has been started and has not
	// ended with [DONE] or a terminal error event. A request whose context was
	// canceled in that state left its client with an incomplete response.
	streamOpen bool
}

type reqInfoKey struct{}

func infoFrom(ctx context.Context) *reqInfo {
	if info, ok := ctx.Value(reqInfoKey{}).(*reqInfo); ok {
		return info
	}
	return &reqInfo{} // a handler running outside the middleware writes nowhere
}

// randHex returns 2*n lowercase hex characters from crypto/rand.
func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("api: crypto/rand failed: " + err.Error())
	}
	return hex.EncodeToString(b)
}

// withRequestID keeps a well-formed incoming X-Request-Id, else generates one,
// and always sets it on the response.
func (s *Server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if !requestIDRE.MatchString(id) {
			id = "req_" + randHex(12)
		}
		w.Header().Set("X-Request-Id", id)
		ctx := context.WithValue(r.Context(), reqInfoKey{}, &reqInfo{requestID: id})
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// statusRecorder remembers the response status for the access log. Unwrap
// keeps http.ResponseController working through it.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (w *statusRecorder) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusRecorder) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(p)
}

func (w *statusRecorder) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// statusClientGone is the status logged for a request whose client left before
// the response was complete. It is nginx's convention; net/http has no name
// for it, and nothing is sent to the departed client.
const statusClientGone = 499

// clientGone reports whether the client left before the handler finished its
// response (Ruling C11). The request context must have ended with
// context.Canceled, which net/http does when the connection closes. The
// gateway's own deadline lives on a context derived from this one and never
// ends it, so a timeout is not mistaken for a departure. And the handler must
// have written nothing (written is the status recorded, 0 for none) or left an
// event stream open; a response that was written whole stays what it was.
func clientGone(ctx context.Context, written int, info *reqInfo) bool {
	if !errors.Is(ctx.Err(), context.Canceled) {
		return false
	}
	return written == 0 || info.streamOpen
}

// withAccessLog logs one line per request. It never logs bodies.
func (s *Server) withAccessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := s.now()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		info := infoFrom(r.Context())
		status := rec.status
		if status == 0 {
			status = http.StatusOK
		}
		gone := clientGone(r.Context(), rec.status, info)
		if gone {
			status = statusClientGone
		}
		s.logger.LogAttrs(r.Context(), slog.LevelInfo, "request",
			slog.String("request_id", info.requestID),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", status),
			slog.Bool("client_gone", gone),
			slog.Int64("duration_ms", s.now().Sub(start).Milliseconds()),
			slog.String("tenant", info.tenant),
			slog.String("model", info.model),
			slog.String("decision_id", info.decisionID),
		)
	})
}

// authenticate resolves the bearer token to its tenant and records the tenant
// for the access log. The returned tenant is shared and must not be mutated.
func (s *Server) authenticate(r *http.Request) (*config.Tenant, *Error) {
	if token, ok := auth.BearerToken(r.Header); ok {
		if t, found := s.keys.Lookup(token); found {
			infoFrom(r.Context()).tenant = t.Name
			return t, nil
		}
	}
	return nil, &Error{Status: http.StatusUnauthorized, Code: CodeInvalidAPIKey, Message: "invalid API key"}
}

// writeJSON writes v as a JSON response with the given status.
func (s *Server) writeJSON(w http.ResponseWriter, r *http.Request, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// Headers are already sent, so the only useful action is to record it.
		s.logger.WarnContext(r.Context(), "write response failed", "request_id", infoFrom(r.Context()).requestID, "error", err)
	}
}

type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	tenant, aerr := s.authenticate(r)
	if aerr != nil {
		writeError(w, aerr)
		return
	}
	data := []modelEntry{{ID: config.AutoModel, Object: "model", OwnedBy: "routeforge"}}
	for _, m := range s.cfg.Models {
		if tenant.Allows(m.Name) {
			data = append(data, modelEntry{ID: m.Name, Object: "model", OwnedBy: "routeforge"})
		}
	}
	s.writeJSON(w, r, http.StatusOK, struct {
		Object string       `json:"object"`
		Data   []modelEntry `json:"data"`
	}{Object: "list", Data: data})
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, r, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	if s.draining.Load() {
		s.writeJSON(w, r, http.StatusServiceUnavailable, map[string]string{"status": "draining"})
		return
	}
	failed := map[string]string{}
	for _, c := range s.checks {
		ctx, cancel := context.WithTimeout(r.Context(), readyCheckTimeout)
		err := c.Check(ctx)
		cancel()
		if err != nil {
			failed[c.Name] = err.Error()
		}
	}
	if len(failed) > 0 {
		s.writeJSON(w, r, http.StatusServiceUnavailable, struct {
			Status string            `json:"status"`
			Failed map[string]string `json:"failed"`
		}{Status: "not_ready", Failed: failed})
		return
	}
	s.writeJSON(w, r, http.StatusOK, map[string]string{"status": "ready"})
}
