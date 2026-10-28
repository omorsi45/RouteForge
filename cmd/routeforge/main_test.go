package main

import (
	"bytes"
	"context"
	"io"
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

// TestHelpFlagsPrintUsageToStdout checks that the flags every command line
// tool answers to behave like the help command: the usage on stdout, nothing
// on stderr, exit 0.
func TestHelpFlagsPrintUsageToStdout(t *testing.T) {
	var want bytes.Buffer
	if code := run(context.Background(), []string{"help"}, &want, io.Discard, os.Getenv); code != 0 {
		t.Fatalf("help: exit %d", code)
	}
	for _, flag := range []string{"-h", "--help", "-help"} {
		t.Run(flag, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(context.Background(), []string{flag}, &out, &errb, os.Getenv); code != 0 {
				t.Fatalf("exit %d, stderr %q", code, errb.String())
			}
			if out.String() != want.String() {
				t.Errorf("stdout = %q, want the usage text %q", out.String(), want.String())
			}
			if errb.Len() != 0 {
				t.Errorf("stderr = %q, want nothing", errb.String())
			}
		})
	}
}
