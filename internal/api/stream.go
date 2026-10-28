package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"

	"github.com/omorsi45/routeforge/internal/provider"
)

type chunkDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *chatUsage    `json:"usage,omitempty"`
}

// chunkWriter renders provider events as SSE chunks that share one id,
// created time and served model name.
type chunkWriter struct {
	id, model    string
	created      int64
	includeUsage bool
	rolePending  bool // the first chunk with a choice carries the assistant role
}

func (c *chunkWriter) frame(ch chatChunk) ([]byte, error) {
	ch.ID, ch.Object, ch.Created, ch.Model = c.id, "chat.completion.chunk", c.created, c.model
	body, err := json.Marshal(ch)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, len("data: ")+len(body)+2)
	out = append(out, "data: "...)
	out = append(out, body...)
	return append(out, '\n', '\n'), nil
}

// choice builds a one-choice chunk, stamping the assistant role on the first.
func (c *chunkWriter) choice(d chunkDelta, finish *string) chatChunk {
	if c.rolePending {
		d.Role = provider.RoleAssistant
		c.rolePending = false
	}
	return chatChunk{Choices: []chunkChoice{{Delta: d, FinishReason: finish}}}
}

// encode renders one provider event as zero to three frames, in the order
// content, finish reason, usage. Usage is dropped unless the client asked for
// it.
func (c *chunkWriter) encode(ev provider.StreamEvent) ([]byte, error) {
	var out []byte
	add := func(ch chatChunk) error {
		b, err := c.frame(ch)
		out = append(out, b...)
		return err
	}
	if ev.Delta != "" {
		if err := add(c.choice(chunkDelta{Content: ev.Delta}, nil)); err != nil {
			return nil, err
		}
	}
	if ev.FinishReason != "" {
		reason := ev.FinishReason
		if err := add(c.choice(chunkDelta{}, &reason)); err != nil {
			return nil, err
		}
	}
	if ev.Usage != nil && c.includeUsage {
		u := &chatUsage{
			PromptTokens:     ev.Usage.PromptTokens,
			CompletionTokens: ev.Usage.CompletionTokens,
			TotalTokens:      ev.Usage.PromptTokens + ev.Usage.CompletionTokens,
		}
		if err := add(chatChunk{Choices: []chunkChoice{}, Usage: u}); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// serveStream runs the streaming call and relays it as server-sent events.
//
// executor.Stream returns with the first event already read, and has retried a
// failure before it, so what still fails before the first byte is an ordinary
// JSON error. After the response is committed an upstream failure is reported
// as a terminal error event and the stream ends without [DONE]; there is no
// retry once a byte has gone out.
func (s *Server) serveStream(w http.ResponseWriter, r *http.Request, st *chatState, preq provider.Request) {
	stream, err := s.exec.Stream(st.deadlineCtx, st.decision, preq, st.execMeter())
	if err != nil {
		s.storeDecision(r, st.decision) // the executor recorded the failed attempt
		if aerr := st.failure(err); aerr != nil {
			writeError(w, aerr)
		}
		return
	}
	// finish closes the stream, which ends the upstream call and records the
	// attempt, and only then stores the decision, so its attempts are final. It
	// runs before the last frame is written, so a client that reads the end of
	// the stream can fetch its decision at once, and from a defer on every other
	// path. The deferred cancel in handleChat then releases the request context.
	finish := sync.OnceFunc(func() {
		if err := stream.Close(); err != nil {
			s.logger.WarnContext(r.Context(), "close provider stream failed", "request_id", st.reqID, "error", err)
		}
		s.storeDecision(r, st.decision)
	})
	defer finish()

	// From here on the access log and routeforge_requests_total name the model
	// that serves the stream, which differs from the chosen one after a
	// failover.
	info := infoFrom(r.Context())
	info.model = stream.Model

	rc := http.NewResponseController(w)
	flush := func() bool {
		err := rc.Flush()
		return err == nil || errors.Is(err, http.ErrNotSupported)
	}

	h := w.Header()
	h.Set("X-RouteForge-Model", stream.Model)
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	// From here until [DONE] or a terminal error event, a client that leaves
	// has an incomplete response (the access log's client_gone).
	info.streamOpen = true
	w.WriteHeader(http.StatusOK)
	if !flush() {
		return
	}
	// The first byte has gone out. The gateway's share of the time to get here
	// excludes what the attempts spent upstream: the failed ones before, and
	// the wait for the first event of this one, which has been read already.
	s.observeOverhead(true, s.now().Sub(info.start)-upstreamTime(st.decision)-stream.TTFT())

	cw := &chunkWriter{
		id:           st.completionID,
		model:        stream.Model,
		created:      st.created,
		includeUsage: st.includeUsage,
		rolePending:  true,
	}
	for {
		ev, err := stream.Recv()
		if err != nil {
			finish() // the stream is over; store before the last frame goes out
			if errors.Is(err, io.EOF) {
				if _, werr := io.WriteString(w, "data: [DONE]\n\n"); werr == nil {
					info.streamOpen = false
					flush()
				}
				return
			}
			if s.writeStreamError(w, st, err, flush) {
				info.streamOpen = false
			}
			return
		}
		frames, err := cw.encode(ev)
		if err != nil {
			// Unreachable: chunks hold only strings and numbers.
			s.logger.ErrorContext(r.Context(), "encode stream chunk failed", "request_id", st.reqID, "error", err)
			return
		}
		if len(frames) == 0 {
			continue
		}
		if _, err := w.Write(frames); err != nil || !flush() {
			return
		}
	}
}

// writeStreamError sends the terminal error event after the response has been
// committed. The failure is classified by chatState.failure, so a client that
// has left (nil) gets nothing and a deadline outranks a cancellation. It
// reports whether a terminal error event was due: false means the client had
// left, so the stream ends without one.
func (s *Server) writeStreamError(w http.ResponseWriter, st *chatState, err error, flush func() bool) bool {
	aerr := st.failure(err)
	if aerr == nil {
		return false
	}
	e := &Error{Status: http.StatusBadGateway, Code: CodeUpstreamStreamInterrupted, Message: "upstream stream interrupted"}
	if aerr.Code == CodeDeadlineExceeded {
		e = &Error{Status: http.StatusGatewayTimeout, Code: CodeDeadlineExceeded, Message: "deadline exceeded"}
	}
	if _, werr := w.Write(streamErrorEvent(e)); werr == nil {
		flush()
	}
	return true
}
