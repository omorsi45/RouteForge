package provider_test

import (
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/omorsi45/routeforge/internal/provider"
)

const sseInput = ": comment\nevent: message_start\ndata: {\"a\":1}\n\ndata: line1\ndata: line2\n\ndata:nospace\n\nid: 5\nretry: 10\n\ndata: [DONE]\n\n"

func wantSSESequence(t *testing.T, in string) {
	t.Helper()
	want := []provider.SSEEvent{
		{Event: "message_start", Data: `{"a":1}`},
		{Data: "line1\nline2"},
		{Data: "nospace"},
		{Data: "[DONE]"},
	}
	r := provider.NewSSEReader(strings.NewReader(in), provider.MaxSSEBytes)
	for i, w := range want {
		got, err := r.Next()
		if err != nil {
			t.Fatalf("event %d: %v", i, err)
		}
		if got != w {
			t.Fatalf("event %d: got %#v, want %#v", i, got, w)
		}
	}
	if _, err := r.Next(); err != io.EOF {
		t.Fatalf("want io.EOF, got %v", err)
	}
}

func TestSSEReaderEvents(t *testing.T) {
	wantSSESequence(t, sseInput)
}

func TestSSEReaderCRLF(t *testing.T) {
	wantSSESequence(t, strings.ReplaceAll(sseInput, "\n", "\r\n"))
}

func TestSSEReaderPendingEventAtEOF(t *testing.T) {
	r := provider.NewSSEReader(strings.NewReader("data: x"), provider.MaxSSEBytes)
	got, err := r.Next()
	if err != nil || got != (provider.SSEEvent{Data: "x"}) {
		t.Fatalf("got %#v, %v", got, err)
	}
	if _, err := r.Next(); err != io.EOF {
		t.Fatalf("want io.EOF, got %v", err)
	}
}

func TestSSEReaderLineTooLarge(t *testing.T) {
	in := "data: " + strings.Repeat("a", 2<<20) + "\n\n"
	r := provider.NewSSEReader(strings.NewReader(in), provider.MaxSSEBytes)
	if _, err := r.Next(); !errors.Is(err, provider.ErrSSETooLarge) {
		t.Fatalf("got %v", err)
	}
}

func TestSSEReaderEventTooLarge(t *testing.T) {
	line := "data: " + strings.Repeat("a", 400<<10) + "\n"
	in := line + line + line + "\n"
	r := provider.NewSSEReader(strings.NewReader(in), 1<<20)
	if _, err := r.Next(); !errors.Is(err, provider.ErrSSETooLarge) {
		t.Fatalf("got %v", err)
	}
}

func TestReadAllLimited(t *testing.T) {
	b, err := provider.ReadAllLimited(strings.NewReader(strings.Repeat("a", 10)), 10)
	if err != nil || len(b) != 10 {
		t.Fatalf("got %d bytes, %v", len(b), err)
	}
	if _, err := provider.ReadAllLimited(strings.NewReader(strings.Repeat("a", 11)), 10); !errors.Is(err, provider.ErrResponseTooLarge) {
		t.Fatalf("got %v", err)
	}
}
