package postgres

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestRedact(t *testing.T) {
	const secret = "pa55-word-xyz"
	base := errors.New("dial postgres://user:" + secret + "@db:5432: refused")

	got := redact(base, secret)
	if strings.Contains(got.Error(), secret) {
		t.Errorf("redact left the secret in %q", got)
	}
	if !strings.Contains(got.Error(), "dial postgres://user:xxxxx@db:5432: refused") {
		t.Errorf("redact changed more than the secret: %q", got)
	}
	if !errors.Is(got, base) {
		t.Error("redact broke the error chain")
	}

	plain := errors.New("connection refused")
	if redact(plain, secret) != plain {
		t.Error("an error without the secret was wrapped")
	}
	if redact(base, "") != base {
		t.Error("an empty secret changed the error")
	}
	if redact(nil, secret) != nil {
		t.Error("redact(nil) is not nil")
	}
}

func TestOpenRejectsAnUnparsableDSNWithoutQuotingIt(t *testing.T) {
	const secret = "hunter2-hunter2"
	_, err := Open(context.Background(), "postgres://routeforge:"+secret+"@localhost:notaport/db", Options{})
	if err == nil {
		t.Fatal("Open accepted an unparsable DSN")
	}
	if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), "notaport") {
		t.Errorf("error %q quotes the DSN", err)
	}
}
