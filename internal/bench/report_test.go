package bench

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The inputs of the golden test are synthetic files of the shape the benchmark
// driver and the trainer write. None of their numbers was measured.
const (
	reportTestdata = "testdata/report"
	reportRunID    = "golden-run"
)

var reportInputFiles = []string{
	"replay.json", "load-zero.json", "load-profiled.json", "faults.json", "evaluation.json", "training.json",
}

func readReportTestdata(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(reportTestdata, name))
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func loadGolden(t *testing.T) ReportInputs {
	t.Helper()
	in, err := LoadReportInputs(reportTestdata, reportTestdata, reportRunID)
	if err != nil {
		t.Fatalf("LoadReportInputs: %v", err)
	}
	return in
}

// edit changes the content of a test input file.
type edit func(t *testing.T, content string) string

// editedReportDir copies the golden inputs to a temporary directory, applying
// the edit of each file that has one, and returns the directory.
func editedReportDir(t *testing.T, edits map[string]edit) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range reportInputFiles {
		content := readReportTestdata(t, name)
		if e, ok := edits[name]; ok {
			content = e(t, content)
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// replacing is the edit that replaces the first occurrence of old by repl.
func replacing(old, repl string) edit {
	return func(t *testing.T, s string) string {
		t.Helper()
		if !strings.Contains(s, old) {
			t.Fatalf("the test data does not contain %q", old)
		}
		return strings.Replace(s, old, repl, 1)
	}
}

// firstDifference describes where two texts first differ.
func firstDifference(got, want string) string {
	g, w := strings.Split(got, "\n"), strings.Split(want, "\n")
	for i := 0; i < max(len(g), len(w)); i++ {
		var gl, wl string
		if i < len(g) {
			gl = g[i]
		}
		if i < len(w) {
			wl = w[i]
		}
		if gl != wl {
			return "line " + strconv.Itoa(i+1) + ":\n got: " + gl + "\nwant: " + wl
		}
	}
	return "no difference"
}

func lineWith(t *testing.T, text, prefix string) string {
	t.Helper()
	for _, l := range strings.Split(text, "\n") {
		if strings.HasPrefix(l, prefix) {
			return l
		}
	}
	t.Fatalf("no line starts with %q in:\n%s", prefix, text)
	return ""
}

func TestRenderBenchmarksGolden(t *testing.T) {
	got, err := RenderBenchmarks(loadGolden(t))
	if err != nil {
		t.Fatal(err)
	}
	if want := readReportTestdata(t, "benchmarks.md"); got != want {
		t.Fatalf("benchmarks.md differs from the golden file at %s", firstDifference(got, want))
	}
}

func TestRenderReadmeResultsGolden(t *testing.T) {
	got, err := RenderReadmeResults(loadGolden(t))
	if err != nil {
		t.Fatal(err)
	}
	if want := readReportTestdata(t, "readme-results.md"); got != want {
		t.Fatalf("the README block differs from the golden file at %s", firstDifference(got, want))
	}
}

func TestRenderEndsWithOneNewline(t *testing.T) {
	in := loadGolden(t)
	for name, render := range map[string]func(ReportInputs) (string, error){
		"benchmarks": RenderBenchmarks, "readme": RenderReadmeResults,
	} {
		out, err := render(in)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasSuffix(out, "\n") || strings.HasSuffix(out, "\n\n") || strings.Contains(out, "\r") {
			t.Errorf("%s: want exactly one trailing newline and no carriage return, got tail %q", name, out[max(0, len(out)-20):])
		}
	}
}

func TestRenderIsPure(t *testing.T) {
	in, untouched := loadGolden(t), loadGolden(t)
	for name, render := range map[string]func(ReportInputs) (string, error){
		"benchmarks": RenderBenchmarks, "readme": RenderReadmeResults,
	} {
		first, err := render(in)
		if err != nil {
			t.Fatal(err)
		}
		second, err := render(in)
		if err != nil {
			t.Fatal(err)
		}
		if first != second {
			t.Errorf("%s: two renders of one input differ at %s", name, firstDifference(second, first))
		}
	}
	if !reflect.DeepEqual(in, untouched) {
		t.Error("rendering changed its input")
	}
}

func TestNoEmDashInOutput(t *testing.T) {
	in := loadGolden(t)
	for name, render := range map[string]func(ReportInputs) (string, error){
		"benchmarks": RenderBenchmarks, "readme": RenderReadmeResults,
	} {
		out, err := render(in)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(out, "\u2014") {
			t.Errorf("%s output has an em dash", name)
		}
	}
}

func TestTargetsReportedPassAndFail(t *testing.T) {
	both := func(t *testing.T, in ReportInputs) (docs, readme string) {
		t.Helper()
		docs, err := RenderBenchmarks(in)
		if err != nil {
			t.Fatal(err)
		}
		readme, err = RenderReadmeResults(in)
		if err != nil {
			t.Fatal(err)
		}
		return docs, readme
	}

	t.Run("the golden inputs", func(t *testing.T) {
		docs, readme := both(t, loadGolden(t))
		for _, want := range []string{
			"Target of 25.0% or more: fail.",
			"p95 gateway overhead < 20 ms: pass (12.50 ms, profiled open loop 200 rps)",
			"H1: falsified (static_oracle_family)",
			"Parity: pass",
			"Accounting invariants: holds",
		} {
			if !strings.Contains(docs, want) {
				t.Errorf("benchmarks.md lacks %q", want)
			}
		}
		if !strings.Contains(readme, "Accounting invariants: holds.") {
			t.Errorf("the README block lacks the invariants line:\n%s", readme)
		}
	})

	t.Run("every verdict flipped", func(t *testing.T) {
		in := loadGolden(t)
		in.Evaluation.H1.TargetMet = true
		in.Evaluation.H1.Falsified = false
		in.Evaluation.H1.FalsifiedBy = nil
		in.LoadProfiled.Results.OverheadTarget = &OverheadTarget{
			P95MSBelow: 20, Scenario: "profiled open loop 200 rps", P95MS: 25.5, Pass: false,
		}
		in.Replay.Results.ParityPass = false
		in.Faults.Results.Invariants.OvershootPositive = false
		in.Faults.Results.Invariants.InvariantsHold = false

		docs, readme := both(t, in)
		for _, want := range []string{
			"Target of 25.0% or more: pass.",
			"p95 gateway overhead < 20 ms: fail (25.50 ms, profiled open loop 200 rps)",
			"H1: supported",
			"Parity: fail",
			"| d. Overshoot is positive (bench-over) | broken |",
			"Accounting invariants: broken",
		} {
			if !strings.Contains(docs, want) {
				t.Errorf("benchmarks.md lacks %q", want)
			}
		}
		for _, want := range []string{
			"H1: supported\n",
			"Target of 25.0% or more: pass.",
			"p95 gateway overhead < 20 ms: fail (25.50 ms, profiled open loop 200 rps)",
			"Gateway replay parity: fail",
			"Accounting invariants: broken.",
		} {
			if !strings.Contains(readme, want) {
				t.Errorf("the README block lacks %q:\n%s", want, readme)
			}
		}
	})

	t.Run("each falsification check", func(t *testing.T) {
		in := loadGolden(t)
		in.Evaluation.H1.VsShareMatchedRandom.Beats = false
		in.Evaluation.H1.VsStaticOracleFamily.Beats = true
		in.Evaluation.H1.RegretExceedsAlphaBeyondCI = true
		in.Evaluation.H1.LodoGainsDisappear = true
		docs, _ := both(t, in)
		for prefix, want := range map[string]string{
			"| Beats share_matched_random":  "| fail |",
			"| Beats static_oracle_family":  "| pass |",
			"| Regret rate not above":       "| fail |",
			"| Leave-one-dataset-out gains": "| fail |",
		} {
			if line := lineWith(t, docs, prefix); !strings.Contains(line, want) {
				t.Errorf("line %q lacks %q", line, want)
			}
		}
		if !strings.Contains(docs, "Gains disappear: yes.") {
			t.Error("the leave-one-dataset-out section does not say the gains disappear")
		}
	})
}

func TestNumbersComeFromTheInputs(t *testing.T) {
	in := loadGolden(t)
	in.Evaluation.H1.Target = 0.4
	in.LoadProfiled.Results.OverheadTarget.P95MSBelow = 50
	in.Replay.Params.ParityThreshold = 0.5
	in.Evaluation.Notes = []string{"first note", "second note", "third note"}

	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	readme, err := RenderReadmeResults(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"Target of 40.0% or more", "p95 gateway overhead < 50 ms", "threshold 0.500",
		"first note", "second note", "third note",
	} {
		if !strings.Contains(docs, want) {
			t.Errorf("benchmarks.md lacks %q", want)
		}
	}
	for _, gone := range []string{"Target of 25.0% or more", "< 20 ms", "threshold 0.999", "Guarantee note text."} {
		if strings.Contains(docs, gone) {
			t.Errorf("benchmarks.md still has %q after its input changed", gone)
		}
	}
	for _, want := range []string{"Target of 40.0% or more", "p95 gateway overhead < 50 ms"} {
		if !strings.Contains(readme, want) {
			t.Errorf("the README block lacks %q", want)
		}
	}
}

// The latency note is fixed text about the method, not a measurement. Whatever
// latency_note a result file holds, the report prints ReplayLatencyNote, so the
// committed documents keep reading the same when the wording changes.
func TestReplayLatencyNoteIsRenderedFromTheConstant(t *testing.T) {
	in := loadGolden(t)
	in.Replay.Results.LatencyNote = "a note that an older run wrote"

	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(docs, ReplayLatencyNote) {
		t.Errorf("benchmarks.md lacks the replay latency note %q", ReplayLatencyNote)
	}
	if strings.Contains(docs, in.Replay.Results.LatencyNote) {
		t.Errorf("benchmarks.md prints the result file's latency note %q", in.Replay.Results.LatencyNote)
	}
}

func TestReadmeMarkersReplaced(t *testing.T) {
	const (
		before = "# Title\n\nIntroduction.\n\n"
		after  = "\nTail text.\n"
	)
	readme := before + "<!-- results:begin -->\nold line 1\nold line 2\n<!-- results:end -->\n" + after
	got, err := ReplaceReadmeResults(readme, "new line 1\nnew line 2\n")
	if err != nil {
		t.Fatal(err)
	}
	want := before + "<!-- results:begin -->\nnew line 1\nnew line 2\n<!-- results:end -->\n" + after
	if got != want {
		t.Fatalf("got:\n%s\nwant:\n%s", got, want)
	}
	again, err := ReplaceReadmeResults(got, "new line 1\nnew line 2\n")
	if err != nil {
		t.Fatal(err)
	}
	if again != got {
		t.Errorf("replacing the same block twice changed the README:\n%s", again)
	}

	t.Run("an empty block between the markers", func(t *testing.T) {
		got, err := ReplaceReadmeResults(before+"<!-- results:begin --><!-- results:end -->"+after, "x\n")
		if err != nil {
			t.Fatal(err)
		}
		if want := before + "<!-- results:begin -->\nx\n<!-- results:end -->" + after; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestReadmeMissingMarkers(t *testing.T) {
	const begin, end = "<!-- results:begin -->", "<!-- results:end -->"
	for _, c := range []struct {
		name, readme, want string
	}{
		{"no markers", "# Title\n", begin},
		{"no begin", "text\n" + end + "\n", begin},
		{"no end", "text\n" + begin + "\nold\n", end},
		{"end before begin", end + "\n" + begin + "\n", "before"},
		{"two begins", begin + "\n" + begin + "\n" + end + "\n", "more than once"},
		{"two ends", begin + "\n" + end + "\n" + end + "\n", "more than once"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := ReplaceReadmeResults(c.readme, "x\n")
			if err == nil {
				t.Fatalf("got %q, want an error", got)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
		})
	}
	if _, err := ReplaceReadmeResults(begin+"\n"+end+"\n", ""); err == nil {
		t.Error("an empty block was accepted")
	}
}

func TestRenderMissingInputsNameTheFile(t *testing.T) {
	type input struct {
		file   string
		remove func(*ReportInputs)
		readme bool // the README block reads it too
	}
	for _, c := range []input{
		{"replay.json", func(in *ReportInputs) { in.Replay = nil }, true},
		{"load-zero.json", func(in *ReportInputs) { in.LoadZero = nil }, false},
		{"load-profiled.json", func(in *ReportInputs) { in.LoadProfiled = nil }, true},
		{"faults.json", func(in *ReportInputs) { in.Faults = nil }, true},
		{"evaluation.json", func(in *ReportInputs) { in.Evaluation = nil }, true},
		{"training.json", func(in *ReportInputs) { in.Training = nil }, false},
	} {
		t.Run(c.file, func(t *testing.T) {
			in := loadGolden(t)
			c.remove(&in)
			out, err := RenderBenchmarks(in)
			if err == nil {
				t.Fatalf("rendered without %s:\n%s", c.file, out)
			}
			if !strings.Contains(err.Error(), c.file) {
				t.Errorf("error %q does not name %s", err, c.file)
			}
			if out != "" {
				t.Errorf("a failed render returned text: %q", out)
			}
			out, err = RenderReadmeResults(in)
			switch {
			case c.readme && err == nil:
				t.Errorf("the README block rendered without %s:\n%s", c.file, out)
			case c.readme && !strings.Contains(err.Error(), c.file):
				t.Errorf("error %q does not name %s", err, c.file)
			case !c.readme && err != nil:
				t.Errorf("the README block does not read %s but failed: %v", c.file, err)
			}
		})
	}

	t.Run("all of them", func(t *testing.T) {
		_, err := RenderBenchmarks(ReportInputs{RunID: reportRunID})
		if err == nil {
			t.Fatal("rendered nothing without error")
		}
		for _, f := range reportInputFiles {
			if !strings.Contains(err.Error(), f) {
				t.Errorf("error %q does not name %s", err, f)
			}
		}
	})
}

func TestLoadReportInputs(t *testing.T) {
	in := loadGolden(t)
	if in.RunID != reportRunID {
		t.Errorf("RunID %q", in.RunID)
	}
	switch {
	case in.Replay == nil || in.LoadZero == nil || in.LoadProfiled == nil || in.Faults == nil || in.Evaluation == nil || in.Training == nil:
		t.Fatalf("an input was not loaded: %+v", in)
	case in.Replay.Results.Requests != 700 || in.Replay.Params.ParityThreshold != 0.999:
		t.Errorf("replay: %+v", in.Replay)
	case in.LoadProfiled.Results.OverheadTarget == nil || !in.LoadProfiled.Results.OverheadTarget.Pass:
		t.Errorf("load-profiled: %+v", in.LoadProfiled.Results.OverheadTarget)
	case in.LoadZero.Results.OverheadTarget != nil:
		t.Error("load-zero has an overhead target")
	case in.Faults.Results.Accounting[OverTenant].OvershootNanos != 3500000:
		t.Errorf("faults accounting: %+v", in.Faults.Results.Accounting)
	case in.Evaluation.ServedPredictor != "lr" || len(in.Evaluation.Strategies) != 8:
		t.Errorf("evaluation: %q, %d strategies", in.Evaluation.ServedPredictor, len(in.Evaluation.Strategies))
	case !slices.Equal(in.Training.Data.Pool, []string{"model-a", "model-b", "model-c"}):
		t.Errorf("training pool %v", in.Training.Data.Pool)
	}
}

func TestLoadReportInputsLeavesAbsentFilesNil(t *testing.T) {
	in, err := LoadReportInputs(t.TempDir(), t.TempDir(), reportRunID)
	if err != nil {
		t.Fatal(err)
	}
	if in.Replay != nil || in.LoadZero != nil || in.LoadProfiled != nil || in.Faults != nil || in.Evaluation != nil || in.Training != nil {
		t.Errorf("an absent file was loaded: %+v", in)
	}
}

func TestLoadReportInputsRejects(t *testing.T) {
	remove := func(old string) edit { return replacing(old, "") }
	one := func(file string, e edit) map[string]edit { return map[string]edit{file: e} }
	for _, c := range []struct {
		name  string
		runID string
		edits map[string]edit
		want  []string
	}{
		{"another run", "other-run", nil, []string{"replay.json", "golden-run", "other-run"}},
		{"another kind", reportRunID, one("replay.json", replacing(`"kind": "replay"`, `"kind": "load-zero"`)),
			[]string{"replay.json", "kind", "load-zero"}},
		{"another schema", reportRunID, one("faults.json", replacing(`"schema_version": 1`, `"schema_version": 2`)),
			[]string{"faults.json", "schema_version"}},
		{"a missing field", reportRunID, one("replay.json", remove(`"parity_pass": true,`)),
			[]string{"replay.json", "results", "parity_pass"}},
		{"a missing latency note", reportRunID, one("replay.json", remove(`"latency_note": "The latency group is reported and not checked.",`)),
			[]string{"replay.json", "results", "latency_note"}},
		{"a missing field of a strategy", reportRunID, one("replay.json", remove(`"agreement": 0.97,`)),
			[]string{"replay.json", "latency", "agreement"}},
		{"a missing field of a mismatch", reportRunID, one("replay.json", remove(`"offline": "model-b",`)),
			[]string{"replay.json", "latency", "mismatches[2]", "offline"}},
		{"a missing field of a scenario", reportRunID, one("load-zero.json", remove(`"error_rate": 0.01,`)),
			[]string{"load-zero.json", "closed_loop_c16", "error_rate"}},
		{"a missing field of the books", reportRunID, one("faults.json", remove(`"overshoot_nanos": 3500000,`)),
			[]string{"faults.json", "bench-over", "overshoot_nanos"}},
		{"a missing field of the evaluation", reportRunID, one("evaluation.json", remove(`"served_predictor": "lr",`)),
			[]string{"evaluation.json", "served_predictor"}},
		{"a missing field of a baseline", reportRunID, one("evaluation.json", remove(`"regret_rate": 0.3,`)),
			[]string{"evaluation.json", "latency", "regret_rate"}},
		{"a missing interval bound", reportRunID, one("evaluation.json", remove(`"ci_low": 0.15,`)),
			[]string{"evaluation.json", "ci_low"}},
		{"the wrong evaluation schema", reportRunID, one("evaluation.json", replacing(`"schema_version": 1`, `"schema_version": 9`)),
			[]string{"evaluation.json", "schema_version"}},
		{"the wrong training schema", reportRunID, one("training.json", replacing(`"schema_version": 1`, `"schema_version": 9`)),
			[]string{"training.json", "schema_version"}},
		{"a file that is not JSON", reportRunID, one("load-profiled.json", func(*testing.T, string) string { return "{" }),
			[]string{"load-profiled.json"}},
		{"a number where a string belongs", reportRunID, one("training.json", replacing(`"model-a"`, `7`)),
			[]string{"training.json"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := editedReportDir(t, c.edits)
			in, err := LoadReportInputs(dir, dir, c.runID)
			if err == nil {
				t.Fatalf("loaded %+v", in)
			}
			for _, want := range c.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

func TestLoadReportInputsAcceptsNullBounds(t *testing.T) {
	dir := editedReportDir(t, map[string]edit{
		"evaluation.json": replacing(`"estimate": 0.2`, `"estimate": null`),
	})
	in, err := LoadReportInputs(dir, dir, reportRunID)
	if err != nil {
		t.Fatalf("a null estimate was rejected: %v", err)
	}
	if in.Evaluation.H1.CPSReduction.Estimate != nil {
		t.Error("a null estimate was not kept as absent")
	}
}

func TestNullsReadNotAvailable(t *testing.T) {
	in := loadGolden(t)
	in.Evaluation.H1.CPSReduction.Estimate = nil
	in.Evaluation.H1.VsStaticOracleFamily.CostPerSuccessDiff.CILow = nil
	oracle := in.Evaluation.Strategies["oracle"]
	oracle.CostPerSuccessUSD = nil
	in.Evaluation.Strategies["oracle"] = oracle
	latency := in.Replay.Results.Strategies["latency"]
	latency.CostPerSuccessUSD = nil
	in.Replay.Results.Strategies["latency"] = latency

	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"reduction versus always_strongest: n/a (95% CI 15.0% to 25.0%)",
		"difference $0.0050 (95% CI n/a to $0.0150)",
		"| oracle | 0.900 | $2.7000 | n/a | 0.000 |",
		"| latency | 100 | 97 | 0.970 | no | 2 | 0.600 | $1.5000 | n/a |",
	} {
		if !strings.Contains(docs, want) {
			t.Errorf("benchmarks.md lacks %q", want)
		}
	}
}

func TestEmptySummariesReadNotAvailable(t *testing.T) {
	in := loadGolden(t)
	sc := in.LoadZero.Results.Scenarios["open_loop_50rps"]
	sc.LatencyMS = map[string]float64{"n": 0}
	sc.RSSBytes, sc.CPUPercent, sc.CPUMethod = nil, nil, ""
	sc.SamplingError = "ps failed"
	sc.DecisionsMissing = 3
	in.LoadZero.Results.Scenarios["open_loop_50rps"] = sc

	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if want := "| open_loop_50rps | 750 | 49.5 | n/a | n/a | n/a | 0.75 | 120 | 0.000 | n/a | n/a |"; !strings.Contains(docs, want) {
		t.Errorf("benchmarks.md lacks %q", want)
	}
	for _, want := range []string{
		"Resource sampling of open_loop_50rps failed: ps failed",
		"3 decisions of open_loop_50rps were not stored when asked for, so its overhead covers fewer requests.",
	} {
		if !strings.Contains(docs, want) {
			t.Errorf("benchmarks.md lacks %q", want)
		}
	}
}

func TestNoCPUMeasurementLeavesTheMethodOut(t *testing.T) {
	in := loadGolden(t)
	for name, sc := range in.LoadZero.Results.Scenarios {
		sc.CPUPercent, sc.CPUMethod = nil, ""
		in.LoadZero.Results.Scenarios[name] = sc
	}
	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(docs, "| CPU mean (%) |") {
		t.Error("the CPU header should not name a method when there is none")
	}
	if strings.Contains(docs, "CPU method `proc_stat_interval`") {
		t.Error("benchmarks.md describes a CPU method that no scenario of the file used")
	}
}

func TestMixedCPUMethodsAreRefused(t *testing.T) {
	in := loadGolden(t)
	sc := in.LoadZero.Results.Scenarios["closed_loop_c16"]
	sc.CPUMethod = CPUMethodPSLifetime
	in.LoadZero.Results.Scenarios["closed_loop_c16"] = sc
	out, err := RenderBenchmarks(in)
	if err == nil {
		t.Fatalf("rendered a CPU column that mixes two methods:\n%s", out)
	}
	for _, want := range []string{"load-zero", "CPU", CPUMethodProcStat, CPUMethodPSLifetime} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestScenarioAndGroupOrder(t *testing.T) {
	in := loadGolden(t)
	c64 := in.LoadZero.Results.Scenarios["closed_loop_c16"]
	in.LoadZero.Results.Scenarios["closed_loop_c64"] = c64
	in.LoadZero.Results.Scenarios["zz_custom"] = c64
	in.Replay.Results.Strategies["zz_extra"] = in.Replay.Results.Strategies["static"]

	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	inOrder := func(names ...string) {
		t.Helper()
		last := -1
		for _, n := range names {
			i := strings.Index(docs, "| "+n+" |")
			if i < 0 || i <= last {
				t.Errorf("row %q is missing or out of order (at %d, after %d)", n, i, last)
			}
			last = i
		}
	}
	inOrder("open_loop_50rps", "closed_loop_c16", "closed_loop_c64", "zz_custom")
	inOrder("latency", "strongest", "cheapest", "static", "quality/strict", "quality/balanced", "quality/economy", "zz_extra")
}

func TestSharesMustNameThePoolModels(t *testing.T) {
	for _, c := range []struct {
		name   string
		change func(map[string]float64)
		want   string
	}{
		{"a model outside the pool", func(s map[string]float64) { s["model-z"] = 0 }, "model-z"},
		{"a model of the pool without a share", func(s map[string]float64) { delete(s, "model-c") }, "model-c"},
	} {
		t.Run(c.name, func(t *testing.T) {
			in := loadGolden(t)
			c.change(in.Evaluation.Strategies["oracle"].Shares)
			out, err := RenderBenchmarks(in)
			if err == nil {
				t.Fatalf("rendered:\n%s", out)
			}
			if !strings.Contains(err.Error(), c.want) || !strings.Contains(err.Error(), "oracle") {
				t.Errorf("error %q does not name the strategy and %q", err, c.want)
			}
		})
	}
}

func TestRenderNeedsTheTrainerNotes(t *testing.T) {
	in := loadGolden(t)
	in.Evaluation.Notes = in.Evaluation.Notes[:2]
	if out, err := RenderBenchmarks(in); err == nil || !strings.Contains(err.Error(), "notes") {
		t.Errorf("got %q, %v; want an error about the notes", out, err)
	}
}

func TestRenderNeedsTheStrategiesItTabulates(t *testing.T) {
	for _, key := range []string{"always_strongest", "quality/lr/balanced", "share_matched_random/balanced", "latency", "oracle"} {
		in := loadGolden(t)
		delete(in.Evaluation.Strategies, key)
		out, err := RenderBenchmarks(in)
		if err == nil {
			t.Errorf("rendered without %s:\n%s", key, out)
			continue
		}
		if !strings.Contains(err.Error(), key) {
			t.Errorf("error %q does not name %s", err, key)
		}
	}
}

func TestTablesEscapePipes(t *testing.T) {
	in := loadGolden(t)
	in.Replay.Env.CPUModel = "Test | CPU"
	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(docs, `| CPU | Test \| CPU |`) {
		t.Errorf("a pipe in a cell is not escaped:\n%s", lineWith(t, docs, "| CPU |"))
	}
}

func TestEnvironmentReadsUnknownForMissingFacts(t *testing.T) {
	in := loadGolden(t)
	in.Replay.Env.GitSHA = ""
	in.Replay.Env.RAMBytes = 0
	docs, err := RenderBenchmarks(in)
	if err != nil {
		t.Fatal(err)
	}
	if line := lineWith(t, docs, "| Git SHA |"); !strings.HasPrefix(line, "| Git SHA | unknown |") {
		t.Errorf("git sha row: %s", line)
	}
	if line := lineWith(t, docs, "| RAM |"); !strings.HasPrefix(line, "| RAM | unknown |") {
		t.Errorf("RAM row: %s", line)
	}
}

func TestFormattersNeverPrintNegativeZero(t *testing.T) {
	for _, c := range []struct{ got, want string }{
		{fixed(-0.00001, 4), "0.0000"},
		{usd(-0.00001), "$0.0000"},
		{usd(-0.01), "-$0.0100"},
		{usd(2), "$2.0000"},
		{pct(-0.00001), "0.0%"},
		{pct(-0.032), "-3.2%"},
		{pct(0.2), "20.0%"},
		{rate(-0.0004), "0.000"},
		{rate(0.9996), "1.000"},
		{ms(-0.001), "0.00"},
		{nanoUSD(3500000), "$0.0035 (3500000 nano-USD)"},
		{nanoUSD(0), "$0.0000 (0 nano-USD)"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}
