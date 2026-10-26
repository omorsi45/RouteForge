package provider

import (
	"bufio"
	"errors"
	"io"
	"strings"
)

// Size limits for upstream reads.
const (
	MaxSSEBytes      = 1 << 20
	MaxResponseBytes = 16 << 20
)

// Size-limit errors.
var (
	ErrSSETooLarge      = errors.New("sse: line or event exceeds size limit")
	ErrResponseTooLarge = errors.New("provider: response body exceeds size limit")
)

// SSEEvent is one dispatched server-sent event.
type SSEEvent struct {
	Event string
	Data  string
}

// SSEReader parses a server-sent event stream. LF and CRLF line endings
// are both accepted.
type SSEReader struct {
	sc  *bufio.Scanner
	max int
}

// NewSSEReader reads events from r, rejecting any line or event larger than
// maxBytes.
func NewSSEReader(r io.Reader, maxBytes int) *SSEReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), maxBytes)
	return &SSEReader{sc: sc, max: maxBytes}
}

// Next returns the next event with data, or io.EOF at a clean end of stream.
func (r *SSEReader) Next() (SSEEvent, error) {
	var event string
	var data []string
	size, hasData := 0, false
	for r.sc.Scan() {
		line := r.sc.Text()
		if line == "" {
			if hasData {
				return SSEEvent{Event: event, Data: strings.Join(data, "\n")}, nil
			}
			event = ""
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			event = value
		case "data":
			size += len(value) + 1
			if size > r.max {
				return SSEEvent{}, ErrSSETooLarge
			}
			data, hasData = append(data, value), true
		} // id, retry and unknown fields are ignored
	}
	if err := r.sc.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return SSEEvent{}, ErrSSETooLarge
		}
		return SSEEvent{}, err
	}
	if hasData { // lenient: dispatch a final event that lacks its blank line
		return SSEEvent{Event: event, Data: strings.Join(data, "\n")}, nil
	}
	return SSEEvent{}, io.EOF
}

// ReadAllLimited reads all of r, failing with ErrResponseTooLarge when more
// than max bytes arrive.
func ReadAllLimited(r io.Reader, max int64) ([]byte, error) {
	b, err := io.ReadAll(io.LimitReader(r, max+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > max {
		return nil, ErrResponseTooLarge
	}
	return b, nil
}
