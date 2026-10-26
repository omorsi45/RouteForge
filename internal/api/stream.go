package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"

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
// executor.Stream returns with the first event already read, so a failure
// before the first byte is an ordinary JSON error. After the response is
// committed an upstream failure is reported as a terminal error event and the
// stream ends without [DONE]; there is no retry once a byte has gone out.
func (s *Server) serveStream(w http.ResponseWriter, r *http.Request, st *chatState, preq provider.Request) {
	stream, err := s.exec.Stream(st.deadlineCtx, st.decision, preq)
	if err != nil {
		if aerr := st.failure(err); aerr != nil {
			writeError(w, aerr)
		}
		return
	}
	// Closing ends the upstream call and records the attempt. The deferred
	// cancel in handleChat then releases the request context on every path.
	defer func() {
		if err := stream.Close(); err != nil {
			s.logger.WarnContext(r.Context(), "close provider stream failed", "request_id", st.reqID, "error", err)
		}
	}()

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
	w.WriteHeader(http.StatusOK)
	if !flush() {
		return
	}

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
			if errors.Is(err, io.EOF) {
				if _, werr := io.WriteString(w, "data: [DONE]\n\n"); werr == nil {
					flush()
				}
				return
			}
			s.writeStreamError(w, st, flush)
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
// committed. A client that has left gets nothing.
func (s *Server) writeStreamError(w http.ResponseWriter, st *chatState, flush func() bool) {
	if st.clientCtx.Err() != nil {
		return
	}
	e := &Error{Status: http.StatusBadGateway, Code: CodeUpstreamStreamInterrupted, Message: "upstream stream interrupted"}
	if errors.Is(st.deadlineCtx.Err(), context.DeadlineExceeded) {
		e = &Error{Status: http.StatusGatewayTimeout, Code: CodeDeadlineExceeded, Message: "deadline exceeded"}
	}
	if _, err := w.Write(streamErrorEvent(e)); err == nil {
		flush()
	}
}
