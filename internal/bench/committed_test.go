package bench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// The tests in this file guard the committed benchmark results: the generated
// documents must be exactly what the report renders from the committed result
// files, and the result files must not hold prompt text.

// committedRoot is the repository root, seen from this package.
const committedRoot = "../.."

// minGuardedPromptBytes is the length under which a prompt is not searched for
// in the result files. A short prompt, such as a single word, is also ordinary
// text of a result file (a key, a model name, a label), so it would flag files
// that hold no prompt.
const minGuardedPromptBytes = 40

// committedRunID is the id of the one run whose results are committed. It is
// found by listing bench/results, so the tests follow a new run without an
// edit; more than one run directory, or none, is an error.
func committedRunID(t *testing.T) string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(committedRoot, resultsPath))
	if err != nil {
		t.Fatal(err)
	}
	var runs []string
	for _, e := range entries {
		if e.IsDir() {
			runs = append(runs, e.Name())
		}
	}
	if len(runs) != 1 {
		t.Fatalf("%s has %d run directories %v, want exactly one", resultsPath, len(runs), runs)
	}
	return runs[0]
}

// committedResultFiles lists the result files of the run, not the raw logs of
// the gitignored raw directory that a local run leaves beside them.
func committedResultFiles(t *testing.T, runID string) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(committedRoot, resultsPath, runID, "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatalf("%s/%s has no result files", resultsPath, runID)
	}
	return files
}

func readCommitted(t *testing.T, rel string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(committedRoot, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestCommittedBenchmarksRegenerate(t *testing.T) {
	runID := committedRunID(t)
	in, err := LoadReportInputs(
		filepath.Join(committedRoot, resultsPath, runID),
		filepath.Join(committedRoot, filepath.FromSlash(reportsPath)),
		runID)
	if err != nil {
		t.Fatal(err)
	}

	doc, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if committed := string(readCommitted(t, "docs/benchmarks.md")); doc != committed {
		t.Errorf("docs/benchmarks.md is not what the report renders from the committed results; first difference at %s\nrun: make report RUN_ID=%s",
			firstDifference(committed, doc), runID)
	}

	block, err := RenderReadmeResults(in)
	if err != nil {
		t.Fatal(err)
	}
	readme := string(readCommitted(t, "README.md"))
	want, err := ReplaceReadmeResults(readme, block)
	if err != nil {
		t.Fatal(err)
	}
	if want != readme {
		t.Errorf("the results block of README.md is not what the report renders from the committed results; first difference at %s\nrun: make report RUN_ID=%s",
			firstDifference(readme, want), runID)
	}
}

func TestCommittedResultsNoRawText(t *testing.T) {
	files := committedResultFiles(t, committedRunID(t))

	t.Run("no prompt key", func(t *testing.T) {
		problems, err := promptKeyProblems(files)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range problems {
			t.Error(p)
		}
	})

	t.Run("no prompt text", func(t *testing.T) {
		promptsFile := filepath.Join(committedRoot, "trainer", "artifacts", "replay", "prompts.jsonl")
		if _, err := os.Stat(promptsFile); os.IsNotExist(err) {
			t.Skip("trainer/artifacts/replay/prompts.jsonl is gitignored and absent here, so prompt text cannot be searched for")
		} else if err != nil {
			t.Fatal(err)
		}
		problems, searched, err := promptTextProblems(files, promptsFile)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range problems {
			t.Error(p)
		}
		if searched == 0 {
			t.Error("no prompt of prompts.jsonl is long enough to search for")
		}
	})
}

// promptKeyProblems lists the result files at paths that have a "prompt" key at
// any depth.
func promptKeyProblems(paths []string) ([]string, error) {
	var problems []string
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var doc any
		if err := json.Unmarshal(data, &doc); err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		if findKey(doc, promptKey) {
			problems = append(problems, fmt.Sprintf("%s has a %q key at some depth, which would hold prompt text", filepath.Base(path), promptKey))
		}
	}
	return problems, nil
}

// promptTextProblems lists the result files at paths that contain the text of a
// prompt of the JSON-lines file promptsFile, and counts the prompts it searched
// for: those of at least minGuardedPromptBytes bytes. The result files are
// JSON, so a prompt with a quote or a line break would sit in them in its
// escaped form: it looks for the text and for its escaped form.
func promptTextProblems(paths []string, promptsFile string) (problems []string, searched int, err error) {
	contents := make(map[string][]byte, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, 0, err
		}
		contents[filepath.Base(path)] = data
	}
	err = readJSONL(promptsFile, func(r promptRecord) error {
		if len(r.Prompt) < minGuardedPromptBytes {
			return nil
		}
		searched++
		escaped, err := escapedJSONText(r.Prompt)
		if err != nil {
			return err
		}
		for _, name := range orderedKeys(contents, nil) {
			if bytes.Contains(contents[name], []byte(r.Prompt)) || bytes.Contains(contents[name], escaped) {
				problems = append(problems, fmt.Sprintf("%s contains the text of prompt %q", name, r.PromptID))
			}
		}
		return nil
	})
	return problems, searched, err
}

// escapedJSONText is s as it is written inside a JSON string, between the
// quotes, the way WriteResult writes it.
func escapedJSONText(s string) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		return nil, err
	}
	quoted := strings.TrimSuffix(buf.String(), "\n")
	return []byte(quoted[1 : len(quoted)-1]), nil
}

// The guards would be worth nothing if they passed on any input, so these tests
// run them on small synthetic files, not on the committed results.

func writeGuardFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestGuardFlagsPromptKeyAtAnyDepth(t *testing.T) {
	dir := t.TempDir()
	clean := writeGuardFile(t, dir, "clean.json", `{"prompt_id":"p:1","results":{"a":[{"prompts":3}]}}`)
	top := writeGuardFile(t, dir, "top.json", `{"prompt":"x"}`)
	deep := writeGuardFile(t, dir, "deep.json", `{"results":{"a":[{"b":{"prompt":"x"}}]}}`)

	problems, err := promptKeyProblems([]string{clean, top, deep})
	if err != nil {
		t.Fatal(err)
	}
	if len(problems) != 2 || !strings.HasPrefix(problems[0], "top.json ") || !strings.HasPrefix(problems[1], "deep.json ") {
		t.Fatalf("problems = %q, want one each for top.json and deep.json and none for clean.json", problems)
	}
}

func TestGuardFlagsPromptText(t *testing.T) {
	dir := t.TempDir()
	plain := "Explain how a binary search tree keeps its keys in order."
	quoted := "Quote the line \"to be, or not to be\" and name the play.\nThen explain it."
	short := "Hello there"
	prompts := writeGuardFile(t, dir, "prompts.jsonl",
		`{"prompt_id":"d:1","prompt":`+jsonString(t, plain)+`}`+"\n"+
			`{"prompt_id":"d:2","prompt":`+jsonString(t, quoted)+`}`+"\n"+
			`{"prompt_id":"d:3","prompt":`+jsonString(t, short)+`}`+"\n")

	clean := writeGuardFile(t, dir, "clean.json", `{"note":"Hello there","prompt_id":"d:1"}`)
	withPlain := writeGuardFile(t, dir, "plain.json", `{"note":`+jsonString(t, "see: "+plain)+`}`)
	withEscaped := writeGuardFile(t, dir, "escaped.json", `{"note":`+jsonString(t, quoted)+`}`)

	problems, searched, err := promptTextProblems([]string{clean, withPlain, withEscaped}, prompts)
	if err != nil {
		t.Fatal(err)
	}
	if searched != 2 {
		t.Errorf("searched for %d prompts, want 2 (the short one is skipped)", searched)
	}
	want := []string{
		`plain.json contains the text of prompt "d:1"`,
		`escaped.json contains the text of prompt "d:2"`,
	}
	if strings.Join(problems, "\n") != strings.Join(want, "\n") {
		t.Fatalf("problems = %q, want %q", problems, want)
	}
}

func jsonString(t *testing.T, s string) string {
	t.Helper()
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// The tests below guard the documents. The hand-written ones state no measured
// result, so what a program can check is checked: no em dash in any document,
// and no claim of a guarantee other than the marginal one that the evaluation
// report states.

func TestDocsHaveNoEmDash(t *testing.T) {
	for _, p := range emDashProblems(committedDocs(t)) {
		t.Error(p)
	}
}

func TestDocsMentionOnlyTheMarginalGuarantee(t *testing.T) {
	for _, p := range guaranteeProblems(committedDocs(t), guaranteeNote(t)) {
		t.Error(p)
	}
}

// requiredDocs are the documents the repository carries. The guards fail when
// one is missing instead of skipping it.
var requiredDocs = []string{
	"README.md",
	"CONTRIBUTING.md",
	"trainer/README.md",
	"docs/architecture.md",
	"docs/routing.md",
	"docs/operations.md",
	"docs/benchmarks.md",
	"docs/adr/0001-offline-trained-certified-router.md",
	"docs/adr/0002-json-policy-artifact-with-parity.md",
	"docs/adr/0003-nano-usd-integer-accounting.md",
	"docs/adr/0004-tenants-in-yaml.md",
	"docs/adr/0005-replay-and-fake-providers-for-evidence.md",
	"examples/demo-output.md",
}

// guaranteeDocs are the documents that must carry the guarantee sentence.
var guaranteeDocs = []string{"README.md", "docs/routing.md"}

// forbiddenGuaranteePhrases claim more than the marginal guarantee. They are
// compared in lower case.
var forbiddenGuaranteePhrases = []string{"per-request guarantee", "guaranteed quality"}

// guaranteeStart begins the guarantee sentence in the notes of the evaluation
// report.
const guaranteeStart = "With probability at least 1 - delta"

// committedDocs reads every document the guards scan, keyed by its path from
// the repository root: the required ones, every Markdown file under docs, and
// the Markdown files directly in examples.
func committedDocs(t *testing.T) map[string]string {
	t.Helper()
	paths := slices.Clone(requiredDocs)
	err := filepath.WalkDir(filepath.Join(committedRoot, "docs"), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".md" {
			rel, err := filepath.Rel(committedRoot, path)
			if err != nil {
				return err
			}
			paths = append(paths, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	examples, err := filepath.Glob(filepath.Join(committedRoot, "examples", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range examples {
		rel, err := filepath.Rel(committedRoot, path)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, filepath.ToSlash(rel))
	}

	docs := make(map[string]string, len(paths))
	for _, rel := range paths {
		data, err := os.ReadFile(filepath.Join(committedRoot, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("document %s cannot be read: %v", rel, err)
		}
		docs[rel] = string(data)
	}
	return docs
}

// guaranteeNote is the guarantee sentence as the evaluation report states it.
// It is found by its first words, not by position, and must be there once.
func guaranteeNote(t *testing.T) string {
	t.Helper()
	var report struct {
		Notes []string `json:"notes"`
	}
	if err := json.Unmarshal(readCommitted(t, reportsPath+"/evaluation.json"), &report); err != nil {
		t.Fatal(err)
	}
	var found []string
	for _, note := range report.Notes {
		if strings.HasPrefix(note, guaranteeStart) {
			found = append(found, note)
		}
	}
	if len(found) != 1 {
		t.Fatalf("evaluation.json has %d notes that start with %q, want exactly one", len(found), guaranteeStart)
	}
	return found[0]
}

// emDashProblems lists the lines of the documents that hold an em dash.
func emDashProblems(docs map[string]string) []string {
	var problems []string
	for _, name := range orderedKeys(docs, nil) {
		for i, line := range strings.Split(docs[name], "\n") {
			if strings.Contains(line, emDash) {
				problems = append(problems, fmt.Sprintf("%s:%d has an em dash", name, i+1))
			}
		}
	}
	return problems
}

// guaranteeProblems lists what is wrong with the way the documents speak of the
// guarantee: a guarantee document that lacks the exact sentence note, and any
// document that holds a phrase claiming more than the marginal guarantee.
func guaranteeProblems(docs map[string]string, note string) []string {
	var problems []string
	for _, name := range guaranteeDocs {
		if !strings.Contains(docs[name], note) {
			problems = append(problems, name+" does not contain the guarantee sentence of the evaluation report")
		}
	}
	for _, name := range orderedKeys(docs, nil) {
		lower := strings.ToLower(docs[name])
		for _, phrase := range forbiddenGuaranteePhrases {
			if strings.Contains(lower, phrase) {
				problems = append(problems, fmt.Sprintf("%s contains %q", name, phrase))
			}
		}
	}
	return problems
}

// The guards would be worth nothing if they passed on any input, so these tests
// run them on small synthetic documents, not on the committed ones.

func TestGuardFlagsEmDash(t *testing.T) {
	docs := map[string]string{
		"clean.md": "A hyphen - is fine.\nSo is a comma, a colon: and a period.\n",
		"dirty.md": "first line\nsecond \u2014 line\nthird \u2014 line\n",
	}
	want := []string{
		"dirty.md:2 has an em dash",
		"dirty.md:3 has an em dash",
	}
	if got := emDashProblems(docs); strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("problems = %q, want %q", got, want)
	}
}

func TestGuardFlagsGuaranteeClaims(t *testing.T) {
	const note = "The marginal guarantee sentence."
	t.Run("sentence in both documents and no claim", func(t *testing.T) {
		docs := map[string]string{
			"README.md":       "Intro. " + note + " More.\n",
			"docs/routing.md": note + "\n",
			"docs/other.md":   "Nothing to see.\n",
		}
		if got := guaranteeProblems(docs, note); len(got) != 0 {
			t.Fatalf("problems = %q, want none", got)
		}
	})
	t.Run("missing sentence and forbidden phrases", func(t *testing.T) {
		docs := map[string]string{
			"README.md":       note + "\n",
			"docs/routing.md": "The sentence is missing here.\n",
			"docs/more.md":    "It offers GUARANTEED QUALITY.\n",
			"docs/other.md":   "This is a Per-Request Guarantee.\n",
		}
		want := []string{
			"docs/routing.md does not contain the guarantee sentence of the evaluation report",
			`docs/more.md contains "guaranteed quality"`,
			`docs/other.md contains "per-request guarantee"`,
		}
		if got := guaranteeProblems(docs, note); strings.Join(got, "\n") != strings.Join(want, "\n") {
			t.Fatalf("problems = %q, want %q", got, want)
		}
	})
	t.Run("absent required document", func(t *testing.T) {
		docs := map[string]string{"docs/routing.md": note + "\n"}
		want := []string{"README.md does not contain the guarantee sentence of the evaluation report"}
		if got := guaranteeProblems(docs, note); strings.Join(got, "\n") != strings.Join(want, "\n") {
			t.Fatalf("problems = %q, want %q", got, want)
		}
	})
}
