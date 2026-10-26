package main

import (
	"bytes"
	"context"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestVersionCommand(t *testing.T) {
	var out, errb bytes.Buffer
	code := run(context.Background(), []string{"version"}, &out, &errb, func(string) string { return "" })
	if code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errb.String())
	}
	if !strings.HasPrefix(out.String(), "routeforge dev ") {
		t.Fatalf("got %q", out.String())
	}
	if !strings.Contains(out.String(), runtime.Version()) {
		t.Fatalf("missing go version: %q", out.String())
	}
}

func TestUnknownCommandExits2(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(context.Background(), []string{"bogus"}, &out, &errb, os.Getenv); code != 2 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(errb.String(), `unknown command "bogus"`) {
		t.Fatalf("stderr %q", errb.String())
	}
}

func TestNoArgsPrintsUsageExits2(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(context.Background(), nil, &out, &errb, os.Getenv); code != 2 {
		t.Fatalf("exit %d", code)
	}
	if !strings.Contains(errb.String(), "Usage: routeforge <command>") {
		t.Fatalf("stderr %q", errb.String())
	}
}

func TestHelpExits0(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(context.Background(), []string{"help"}, &out, &errb, os.Getenv); code != 0 {
		t.Fatalf("exit %d", code)
	}
	for _, c := range []string{"serve", "validate-config", "keygen", "version", "healthcheck"} {
		if !strings.Contains(out.String(), c) {
			t.Fatalf("usage missing %q", c)
		}
	}
}
