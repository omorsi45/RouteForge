package bench

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"math"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
)

// The files the report reads: result files under bench/results/<run-id> and
// the reports of the trainer under trainer/artifacts/reports.
const (
	replayFileName       = "replay.json"
	loadZeroFileName     = KindLoadZero + ".json"
	loadProfiledFileName = KindLoadProfiled + ".json"
	faultsFileName       = "faults.json"
	evaluationFileName   = "evaluation.json"
	trainingFileName     = "training.json"

	resultsPath = "bench/results"
	reportsPath = "trainer/artifacts/reports"
)

// trainerSchemaVersion is the schema_version of the trainer reports that the
// report reads.
const trainerSchemaVersion = 1

// The markers that enclose the results block of the README.
const (
	ReadmeBeginMarker = "<!-- results:begin -->"
	ReadmeEndMarker   = "<!-- results:end -->"
)

// ciLabel names the confidence level of the trainer's intervals.
const ciLabel = "95% CI"

// unavailable is what a value the files leave out reads as. It is a word, not a
// number: the report never makes up a figure.
const unavailable = "n/a"

// ResultFile is a result file as WriteResult writes it, with the types of the
// params and results of its kind.
type ResultFile[P, R any] struct {
	SchemaVersion int    `json:"schema_version"`
	Kind          string `json:"kind"`
	RunID         string `json:"run_id"`
	CreatedAt     string `json:"created_at"`
	Env           Env    `json:"env"`
	ConfigPath    string `json:"config_path"`
	Params        P      `json:"params"`
	Results       R      `json:"results"`
}

// ReplayParams are the params of a replay result that the report reads.
type ReplayParams struct {
	ParityThreshold float64 `json:"parity_threshold"`
}

// The result files by kind.
type (
	ReplayFile = ResultFile[ReplayParams, ReplayResults]
	LoadFile   = ResultFile[map[string]any, LoadResults]
	FaultsFile = ResultFile[map[string]any, FaultsResults]
)

// Interval is an estimate with its confidence interval. A part the trainer
// could not define is null, and nil here.
type Interval struct {
	Estimate *float64 `json:"estimate"`
	CILow    *float64 `json:"ci_low"`
	CIHigh   *float64 `json:"ci_high"`
}

// StrategySummary is what the trainer's evaluation reports about one strategy.
type StrategySummary struct {
	N                 int                `json:"n"`
	Accuracy          float64            `json:"accuracy"`
	TotalCostUSD      float64            `json:"total_cost_usd"`
	CostPerSuccessUSD *float64           `json:"cost_per_success_usd"`
	RegretRate        float64            `json:"regret_rate"`
	Shares            map[string]float64 `json:"shares"` // of requests served, by model
}

// H1Comparison is the comparison of the quality router with a baseline on cost
// per success.
type H1Comparison struct {
	Beats              bool     `json:"beats"`
	CostPerSuccessDiff Interval `json:"cost_per_success_diff"`
}

// H1Regret is the realized regret rate of the quality router with its interval
// and the alpha it is held to.
type H1Regret struct {
	Rate   float64  `json:"rate"`
	Count  int      `json:"count"`
	N      int      `json:"n"`
	CILow  *float64 `json:"ci_low"`
	CIHigh *float64 `json:"ci_high"`
	Alpha  float64  `json:"alpha"`
}

// H1Verdict is the trainer's verdict on hypothesis H1.
type H1Verdict struct {
	Profile                    string       `json:"profile"`
	Predictor                  string       `json:"predictor"`
	CPSReduction               Interval     `json:"cps_reduction_vs_strongest"`
	Target                     float64      `json:"target"`
	TargetMet                  bool         `json:"target_met"`
	Regret                     H1Regret     `json:"regret"`
	RegretExceedsAlphaBeyondCI bool         `json:"regret_exceeds_alpha_beyond_ci"`
	VsShareMatchedRandom       H1Comparison `json:"vs_share_matched_random"`
	VsStaticOracleFamily       H1Comparison `json:"vs_static_oracle_family"`
	LodoPooledReduction        Interval     `json:"lodo_pooled_reduction"`
	LodoGainsDisappear         bool         `json:"lodo_gains_disappear"`
	Falsified                  bool         `json:"falsified"`
	FalsifiedBy                []string     `json:"falsified_by"`
}

// LodoPooled is the leave-one-dataset-out result pooled over the datasets: the
// reduction of cost per success against always_strongest.
type LodoPooled struct {
	Interval
	N                   int `json:"n"`
	NResamples          int `json:"n_resamples"`
	DegenerateResamples int `json:"degenerate_resamples"`
}

// Lodo is the leave-one-dataset-out section of the evaluation.
type Lodo struct {
	Pooled LodoPooled `json:"pooled"`
}

// Evaluation is the part of trainer/artifacts/reports/evaluation.json that the
// report reads.
type Evaluation struct {
	SchemaVersion   int                        `json:"schema_version"`
	PolicyVersion   string                     `json:"policy_version"`
	ServedPredictor string                     `json:"served_predictor"`
	Strategies      map[string]StrategySummary `json:"strategies"`
	H1              H1Verdict                  `json:"h1"`
	Lodo            Lodo                       `json:"lodo"`
	// Notes are the fixed texts of the evaluation: the guarantee, the note on
	// benchmark templates, the note on the latency router, then the cost note.
	Notes []string `json:"notes"`
}

// TrainingData is the data section of the training report.
type TrainingData struct {
	Pool []string `json:"pool"` // the models, in pool order
}

// Training is the part of trainer/artifacts/reports/training.json that the
// report reads.
type Training struct {
	SchemaVersion int          `json:"schema_version"`
	Data          TrainingData `json:"data"`
}

// ReportInputs are the parsed files the report is rendered from. A nil input is
// a file that does not exist; the renderers refuse to run without the inputs
// they read, and say which file is missing.
type ReportInputs struct {
	RunID                  string
	Replay                 *ReplayFile
	LoadZero, LoadProfiled *LoadFile
	Faults                 *FaultsFile
	Evaluation             *Evaluation
	Training               *Training
}

// LoadReportInputs reads the result files of run runID from resultsDir and the
// trainer reports from reportsDir. A file that does not exist leaves its input
// nil. A file that exists must be of the run and kind asked for, of a known
// schema version, and carry every field the benchmark driver or the trainer
// always writes and the report reads: a missing field is an error naming the
// file and the field, not a zero.
func LoadReportInputs(resultsDir, reportsDir, runID string) (ReportInputs, error) {
	in := ReportInputs{RunID: runID}
	var err error
	if in.Replay, err = loadResultFile[ReplayParams, ReplayResults](resultsDir, replayFileName, "replay", runID); err != nil {
		return ReportInputs{}, err
	}
	if in.LoadZero, err = loadResultFile[map[string]any, LoadResults](resultsDir, loadZeroFileName, KindLoadZero, runID); err != nil {
		return ReportInputs{}, err
	}
	if in.LoadProfiled, err = loadResultFile[map[string]any, LoadResults](resultsDir, loadProfiledFileName, KindLoadProfiled, runID); err != nil {
		return ReportInputs{}, err
	}
	if in.Faults, err = loadResultFile[map[string]any, FaultsResults](resultsDir, faultsFileName, "faults", runID); err != nil {
		return ReportInputs{}, err
	}
	if in.Evaluation, err = loadTrainerFile[Evaluation](reportsDir, evaluationFileName, func(e *Evaluation) int { return e.SchemaVersion }); err != nil {
		return ReportInputs{}, err
	}
	if in.Training, err = loadTrainerFile[Training](reportsDir, trainingFileName, func(t *Training) int { return t.SchemaVersion }); err != nil {
		return ReportInputs{}, err
	}
	return in, nil
}

// readReportFile decodes the JSON file dir/name into v, a pointer, and checks
// that it has every field v's type always carries. found is false when the file
// does not exist.
func readReportFile(dir, name string, v any) (found bool, err error) {
	data, err := os.ReadFile(filepath.Join(dir, name))
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return true, fmt.Errorf("read %s: %w", name, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		return true, fmt.Errorf("%s: %w", name, err)
	}
	if err := requireFields(data, reflect.TypeOf(v).Elem(), ""); err != nil {
		return true, fmt.Errorf("%s: %w", name, err)
	}
	return true, nil
}

func loadResultFile[P, R any](dir, name, kind, runID string) (*ResultFile[P, R], error) {
	var f ResultFile[P, R]
	found, err := readReportFile(dir, name, &f)
	if err != nil || !found {
		return nil, err
	}
	switch {
	case f.SchemaVersion != SchemaVersion:
		return nil, fmt.Errorf("%s: schema_version is %d, want %d", name, f.SchemaVersion, SchemaVersion)
	case f.Kind != kind:
		return nil, fmt.Errorf("%s: kind is %q, want %q", name, f.Kind, kind)
	case f.RunID != runID:
		return nil, fmt.Errorf("%s: run_id is %q, want %q", name, f.RunID, runID)
	}
	return &f, nil
}

func loadTrainerFile[T any](dir, name string, schema func(*T) int) (*T, error) {
	var v T
	found, err := readReportFile(dir, name, &v)
	if err != nil || !found {
		return nil, err
	}
	if got := schema(&v); got != trainerSchemaVersion {
		return nil, fmt.Errorf("%s: schema_version is %d, want %d", name, got, trainerSchemaVersion)
	}
	return &v, nil
}

// requireFields checks that the JSON value raw has, for the type t, every field
// its writer always emits: the fields with a json tag that is not omitempty. A
// null is fine where t is a pointer. It looks inside objects, maps of objects
// and lists of objects, and reports the path of the first missing field.
func requireFields(raw json.RawMessage, t reflect.Type, at string) error {
	switch t.Kind() {
	case reflect.Pointer:
		if string(bytes.TrimSpace(raw)) == "null" {
			return nil
		}
		return requireFields(raw, t.Elem(), at)
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
			return fmt.Errorf("%s is not an object", pathName(at))
		}
		return requireStructFields(fields, t, at)
	case reflect.Map:
		if !holdsObjects(t.Elem()) {
			return nil
		}
		var entries map[string]json.RawMessage
		if err := json.Unmarshal(raw, &entries); err != nil || entries == nil {
			return fmt.Errorf("%s is not an object", pathName(at))
		}
		for _, key := range slices.Sorted(maps.Keys(entries)) {
			if err := requireFields(entries[key], t.Elem(), joinPath(at, key)); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if !holdsObjects(t.Elem()) {
			return nil
		}
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil || items == nil {
			return fmt.Errorf("%s is not a list", pathName(at))
		}
		for i, item := range items {
			if err := requireFields(item, t.Elem(), fmt.Sprintf("%s[%d]", at, i)); err != nil {
				return err
			}
		}
	}
	return nil
}

func requireStructFields(fields map[string]json.RawMessage, t reflect.Type, at string) error {
	for i := range t.NumField() {
		f := t.Field(i)
		tag, tagged := f.Tag.Lookup("json")
		name, opts, _ := strings.Cut(tag, ",")
		switch {
		case f.Anonymous && !tagged && f.Type.Kind() == reflect.Struct:
			// An embedded struct's fields sit in the same object.
			if err := requireStructFields(fields, f.Type, at); err != nil {
				return err
			}
			continue
		case !f.IsExported() || name == "-":
			continue
		case name == "":
			name = f.Name
		}
		raw, ok := fields[name]
		if !ok {
			if strings.Contains(opts, "omitempty") {
				continue
			}
			return fmt.Errorf("%s is missing %q", pathName(at), name)
		}
		if err := requireFields(raw, f.Type, joinPath(at, name)); err != nil {
			return err
		}
	}
	return nil
}

// holdsObjects reports whether values of type t are, or contain, objects whose
// fields requireFields checks.
func holdsObjects(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice:
		return holdsObjects(t.Elem())
	case reflect.Struct:
		return true
	}
	return false
}

func joinPath(at, name string) string {
	if at == "" {
		return name
	}
	return at + "." + name
}

func pathName(at string) string {
	if at == "" {
		return "the document"
	}
	return at
}

// inputFile identifies an input of the report.
type inputFile int

const (
	inReplay inputFile = iota
	inLoadZero
	inLoadProfiled
	inFaults
	inEvaluation
	inTraining
)

// require fails, naming the files, unless every one of files is loaded.
func (in ReportInputs) require(files ...inputFile) error {
	var missing []string
	result := func(name string) string { return path.Join(resultsPath, in.RunID, name) }
	for _, f := range files {
		switch {
		case f == inReplay && in.Replay == nil:
			missing = append(missing, result(replayFileName))
		case f == inLoadZero && in.LoadZero == nil:
			missing = append(missing, result(loadZeroFileName))
		case f == inLoadProfiled && in.LoadProfiled == nil:
			missing = append(missing, result(loadProfiledFileName))
		case f == inFaults && in.Faults == nil:
			missing = append(missing, result(faultsFileName))
		case f == inEvaluation && in.Evaluation == nil:
			missing = append(missing, path.Join(reportsPath, evaluationFileName))
		case f == inTraining && in.Training == nil:
			missing = append(missing, path.Join(reportsPath, trainingFileName))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing input: %s", strings.Join(missing, ", "))
	}
	return nil
}

// RenderBenchmarks renders docs/benchmarks.md from the inputs. Every number in
// it is formatted from them. It needs all six inputs and returns an error
// naming the ones that are missing.
func RenderBenchmarks(in ReportInputs) (string, error) {
	if err := in.require(inReplay, inLoadZero, inLoadProfiled, inFaults, inEvaluation, inTraining); err != nil {
		return "", err
	}
	doc := blocks{
		"# Benchmarks",
		"Generated by `routeforge-bench report` from `" + path.Join(resultsPath, in.RunID) + "/` and `" +
			path.Join(reportsPath, evaluationFileName) + "`. Do not edit by hand.",
	}
	doc.add("## Environment")
	doc.add(environmentTable(in))

	offline, err := offlineBlocks(in.Evaluation, in.Training)
	if err != nil {
		return "", err
	}
	doc = append(doc, offline...)
	doc = append(doc, lodoBlocks(in.Evaluation)...)
	doc = append(doc, replayBlocks(in.Replay)...)
	load, err := loadBlocks(in.LoadZero, in.LoadProfiled)
	if err != nil {
		return "", err
	}
	doc = append(doc, load...)
	faults, err := faultsBlocks(in.Faults)
	if err != nil {
		return "", err
	}
	doc = append(doc, faults...)
	doc = append(doc, reproduceBlocks(in.RunID)...)
	return doc.String(), nil
}

// RenderReadmeResults renders the results block of the README, the text between
// its markers: the headline numbers of the benchmarks, each with its verdict on
// its target, and a link to docs/benchmarks.md. It needs the replay, the
// profiled load and the fault results and the evaluation.
func RenderReadmeResults(in ReportInputs) (string, error) {
	if err := in.require(inReplay, inLoadProfiled, inFaults, inEvaluation); err != nil {
		return "", err
	}
	target := in.LoadProfiled.Results.OverheadTarget
	if target == nil {
		return "", fmt.Errorf("%s has no overhead_target", loadProfiledFileName)
	}
	h1, replay, faults := in.Evaluation.H1, in.Replay.Results, in.Faults.Results
	regret := h1.Regret
	return blocks{
		h1Verdict(h1),
		strings.Join([]string{
			"- " + reductionLine(h1),
			fmt.Sprintf("- Regret rate: %s against alpha %s (%s %s to %s).", rate(regret.Rate), rate(regret.Alpha),
				ciLabel, opt(regret.CILow, rate), opt(regret.CIHigh, rate)),
			fmt.Sprintf("- Gateway replay parity: %s (deterministic agreement %s).", passFail(replay.ParityPass), rate(replay.DeterministicAgreement)),
			"- " + overheadLine(*target),
			fmt.Sprintf("- Accounting invariants: %s.", holdsBroken(faults.Invariants.InvariantsHold)),
		}, "\n"),
		fmt.Sprintf("Run `%s`. Full results, environment and commands: [docs/benchmarks.md](docs/benchmarks.md).", in.RunID),
	}.String(), nil
}

// ReplaceReadmeResults returns readme with the text between its results
// markers replaced by block. Each marker must appear exactly once, the begin
// marker first. Nothing outside the markers changes.
func ReplaceReadmeResults(readme, block string) (string, error) {
	block = strings.TrimRight(block, "\n")
	if block == "" {
		return "", errors.New("the results block is empty")
	}
	begin, err := markerIndex(readme, ReadmeBeginMarker)
	if err != nil {
		return "", err
	}
	end, err := markerIndex(readme, ReadmeEndMarker)
	if err != nil {
		return "", err
	}
	if end < begin+len(ReadmeBeginMarker) {
		return "", fmt.Errorf("the end marker %s comes before the begin marker %s", ReadmeEndMarker, ReadmeBeginMarker)
	}
	return readme[:begin+len(ReadmeBeginMarker)] + "\n" + block + "\n" + readme[end:], nil
}

func markerIndex(text, marker string) (int, error) {
	switch strings.Count(text, marker) {
	case 0:
		return 0, fmt.Errorf("the marker %s is missing", marker)
	case 1:
		return strings.Index(text, marker), nil
	}
	return 0, fmt.Errorf("the marker %s appears more than once", marker)
}

// blocks are the paragraphs, headings, tables and code blocks of a Markdown
// document, which are set apart by one blank line.
type blocks []string

func (b *blocks) add(lines ...string) { *b = append(*b, strings.Join(lines, "\n")) }

func (b blocks) String() string { return strings.Join(b, "\n\n") + "\n" }

// table renders a Markdown table. Pipes and line breaks in a cell are escaped.
func table(header []string, rows [][]string) string {
	line := func(cells []string) string {
		var sb strings.Builder
		sb.WriteString("|")
		for _, c := range cells {
			c = strings.NewReplacer("|", `\|`, "\r\n", " ", "\n", " ", "\r", " ").Replace(c)
			sb.WriteString(" " + c + " |")
		}
		return sb.String()
	}
	lines := []string{line(header), line(slices.Repeat([]string{"---"}, len(header)))}
	for _, row := range rows {
		lines = append(lines, line(row))
	}
	return strings.Join(lines, "\n")
}

// fixed formats x with prec decimals. A value that rounds to zero has no sign.
func fixed(x float64, prec int) string {
	s := strconv.FormatFloat(x, 'f', prec, 64)
	if strings.HasPrefix(s, "-") && strings.Trim(s, "-0.") == "" {
		return s[1:]
	}
	return s
}

func rate(x float64) string { return fixed(x, 3) }           // a rate or share between 0 and 1
func pct(x float64) string  { return fixed(100*x, 1) + "%" } // a fraction as a percentage
func ms(x float64) string   { return fixed(x, 2) }

func usd(x float64) string {
	s := fixed(math.Abs(x), 4)
	if x < 0 && s != "0.0000" {
		return "-$" + s
	}
	return "$" + s
}

// nanoUSD renders an amount of nano-USD as USD with the exact count beside it.
func nanoUSD(n int64) string { return fmt.Sprintf("%s (%d nano-USD)", usd(float64(n)/1e9), n) }

func mib(b float64) string { return fixed(b/(1<<20), 1) }

// opt formats *p with f, or reads n/a for a value the file leaves null.
func opt(p *float64, f func(float64) string) string {
	if p == nil {
		return unavailable
	}
	return f(*p)
}

// stat formats the named figure of a summary, or reads n/a when the summary
// lacks it, as the summary of no values does.
func stat(summary map[string]float64, key string, f func(float64) string) string {
	v, ok := summary[key]
	if !ok {
		return unavailable
	}
	return f(v)
}

// text renders the interval as `estimate (95% CI low to high)`.
func (iv Interval) text(f func(float64) string) string {
	return fmt.Sprintf("%s (%s %s to %s)", opt(iv.Estimate, f), ciLabel, opt(iv.CILow, f), opt(iv.CIHigh, f))
}

func passFail(ok bool) string {
	if ok {
		return "pass"
	}
	return "fail"
}

func holdsBroken(ok bool) string {
	if ok {
		return "holds"
	}
	return "broken"
}

func yesNo(ok bool) string {
	if ok {
		return "yes"
	}
	return "no"
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}

func code(s string) string {
	if s == "" {
		return "unknown"
	}
	return "`" + s + "`"
}

// orderedKeys lists the keys of m: the preferred ones that m has, in the order
// given, then any others in alphabetical order.
func orderedKeys[V any](m map[string]V, preferred []string) []string {
	keys := make([]string, 0, len(m))
	for _, k := range preferred {
		if _, ok := m[k]; ok {
			keys = append(keys, k)
		}
	}
	for _, k := range slices.Sorted(maps.Keys(m)) {
		if !slices.Contains(preferred, k) {
			keys = append(keys, k)
		}
	}
	return keys
}

// environmentTable lists the environment of each result file, one column per
// file.
func environmentTable(in ReportInputs) string {
	type entry struct {
		name, createdAt, config string
		env                     Env
	}
	entries := []entry{
		{"replay", in.Replay.CreatedAt, in.Replay.ConfigPath, in.Replay.Env},
		{KindLoadZero, in.LoadZero.CreatedAt, in.LoadZero.ConfigPath, in.LoadZero.Env},
		{KindLoadProfiled, in.LoadProfiled.CreatedAt, in.LoadProfiled.ConfigPath, in.LoadProfiled.Env},
		{"faults", in.Faults.CreatedAt, in.Faults.ConfigPath, in.Faults.Env},
	}
	fields := []struct {
		name  string
		value func(e entry) string
	}{
		{"Created (UTC)", func(e entry) string { return orUnknown(e.createdAt) }},
		{"Config", func(e entry) string { return code(e.config) }},
		{"Config SHA-256", func(e entry) string { return code(e.env.ConfigSHA256) }},
		{"Git SHA", func(e entry) string { return code(e.env.GitSHA) }},
		{"Go", func(e entry) string { return orUnknown(e.env.GoVersion) }},
		{"Python (trainer)", func(e entry) string { return orUnknown(e.env.PythonVersion) }},
		{"OS", func(e entry) string { return orUnknown(e.env.OS) }},
		{"Kernel", func(e entry) string { return orUnknown(e.env.Kernel) }},
		{"CPU", func(e entry) string { return orUnknown(e.env.CPUModel) }},
		{"CPUs", func(e entry) string {
			if e.env.CPUs <= 0 {
				return "unknown"
			}
			return strconv.Itoa(e.env.CPUs)
		}},
		{"RAM", func(e entry) string {
			if e.env.RAMBytes <= 0 {
				return "unknown"
			}
			return mib(float64(e.env.RAMBytes)) + " MiB"
		}},
		{"Dataset revision", func(e entry) string { return code(e.env.DatasetRevision) }},
		{"Policy version", func(e entry) string { return code(e.env.PolicyVersion) }},
	}
	header := []string{"Field"}
	for _, e := range entries {
		header = append(header, e.name)
	}
	rows := make([][]string, len(fields))
	for i, f := range fields {
		rows[i] = []string{f.name}
		for _, e := range entries {
			rows[i] = append(rows[i], f.value(e))
		}
	}
	return table(header, rows)
}

// offlineStrategy is a row of the offline tables: the key of the strategy in
// the evaluation and how the row is named.
type offlineStrategy struct{ key, label string }

// offlineStrategies are the strategies the offline tables list, in order, for
// the profile and the served predictor of the H1 test.
func offlineStrategies(ev *Evaluation) []offlineStrategy {
	profile := ev.H1.Profile
	quality := "quality/" + ev.ServedPredictor + "/" + profile
	matched := "share_matched_random/" + profile
	return []offlineStrategy{
		{"always_strongest", "always_strongest"},
		{"always_cheapest", "always_cheapest"},
		{"static_keyword", "static_keyword"},
		{"static_oracle_family", "static_oracle_family"},
		{matched, matched},
		{quality, quality},
		{"latency", "latency (reference only)"},
		{"oracle", "oracle"},
	}
}

func offlineBlocks(ev *Evaluation, tr *Training) ([]string, error) {
	if len(ev.Notes) < 3 {
		return nil, fmt.Errorf("%s: notes has %d entries, want the guarantee, template and latency notes", evaluationFileName, len(ev.Notes))
	}
	strategies := offlineStrategies(ev)
	var qualityN int
	costRows := make([][]string, len(strategies))
	shareRows := make([][]string, len(strategies))
	for i, s := range strategies {
		sum, ok := ev.Strategies[s.key]
		if !ok {
			return nil, fmt.Errorf("%s has no strategy %q", evaluationFileName, s.key)
		}
		if s.key == "quality/"+ev.ServedPredictor+"/"+ev.H1.Profile {
			qualityN = sum.N
		}
		costRows[i] = []string{s.label, rate(sum.Accuracy), usd(sum.TotalCostUSD), opt(sum.CostPerSuccessUSD, usd), rate(sum.RegretRate)}

		for model := range sum.Shares {
			if !slices.Contains(tr.Data.Pool, model) {
				return nil, fmt.Errorf("%s: strategy %q has a share for model %q, which is not in the pool of %s", evaluationFileName, s.key, model, trainingFileName)
			}
		}
		shareRows[i] = []string{s.label}
		for _, model := range tr.Data.Pool {
			share, ok := sum.Shares[model]
			if !ok {
				return nil, fmt.Errorf("%s: strategy %q has no share for pool model %q", evaluationFileName, s.key, model)
			}
			shareRows[i] = append(shareRows[i], pct(share))
		}
	}

	var out blocks
	out.add("## Offline routing quality and cost")
	out.add(fmt.Sprintf("Policy version `%s`, served predictor `%s`, %s profile, %d test prompts.",
		ev.PolicyVersion, ev.ServedPredictor, ev.H1.Profile, qualityN))
	out.add(table([]string{"Strategy", "Accuracy", "Total cost", "Cost per success", "Regret rate"}, costRows))
	out.add("Share of requests served by each model:")
	out.add(table(append([]string{"Strategy"}, tr.Data.Pool...), shareRows))
	out = append(out, h1Blocks(ev)...)
	return out, nil
}

// reductionLine is the sentence on the cost-per-success target of H1.
func reductionLine(h1 H1Verdict) string {
	return fmt.Sprintf("Cost per success reduction versus always_strongest: %s. Target of %s or more: %s.",
		h1.CPSReduction.text(pct), pct(h1.Target), passFail(h1.TargetMet))
}

func h1Verdict(h1 H1Verdict) string {
	if h1.Falsified {
		return "H1: falsified (" + strings.Join(h1.FalsifiedBy, ", ") + ")"
	}
	return "H1: supported"
}

func h1Blocks(ev *Evaluation) []string {
	h1 := ev.H1
	regret := h1.Regret
	checks := [][]string{
		{"Beats share_matched_random on cost per success", passFail(h1.VsShareMatchedRandom.Beats),
			"difference " + h1.VsShareMatchedRandom.CostPerSuccessDiff.text(usd)},
		{"Beats static_oracle_family on cost per success", passFail(h1.VsStaticOracleFamily.Beats),
			"difference " + h1.VsStaticOracleFamily.CostPerSuccessDiff.text(usd)},
		{"Regret rate not above alpha beyond its CI", passFail(!h1.RegretExceedsAlphaBeyondCI),
			fmt.Sprintf("CI lower bound %s against alpha %s", opt(regret.CILow, rate), rate(regret.Alpha))},
		{"Leave-one-dataset-out gains persist", passFail(!h1.LodoGainsDisappear),
			"pooled reduction " + h1.LodoPooledReduction.text(pct)},
	}
	var out blocks
	out.add(fmt.Sprintf("### H1 (%s, %s)", h1.Profile, h1.Predictor))
	out.add(
		"- "+reductionLine(h1),
		fmt.Sprintf("- Regret rate: %s (%d of %d), %s %s to %s, alpha %s.", rate(regret.Rate), regret.Count, regret.N,
			ciLabel, opt(regret.CILow, rate), opt(regret.CIHigh, rate), rate(regret.Alpha)),
	)
	out.add("H1 is falsified when any check below fails. The cost target above is reported separately and does not decide it.")
	out.add(table([]string{"Check", "Result", "Evidence"}, checks))
	out.add(h1Verdict(h1))
	out.add("### Guarantee and notes")
	out.add(ev.Notes[0])
	out.add("- "+ev.Notes[1], "- "+ev.Notes[2])
	return out
}

func lodoBlocks(ev *Evaluation) []string {
	p := ev.Lodo.Pooled
	return blocks{
		"## Leave-one-dataset-out",
		fmt.Sprintf("Pooled over the held-out datasets (%d prompts): cost per success reduction versus always_strongest %s, %d resamples, %d degenerate. Gains disappear: %s.",
			p.N, p.Interval.text(pct), p.NResamples, p.DegenerateResamples, yesNo(ev.H1.LodoGainsDisappear)),
	}
}

func replayBlocks(f *ReplayFile) []string {
	r := f.Results
	labels := make([]string, len(replayGroups))
	for i, g := range replayGroups {
		labels[i] = g.label()
	}
	rows := [][]string{}
	for _, name := range orderedKeys(r.Strategies, labels) {
		s := r.Strategies[name]
		errs := 0
		for _, n := range s.Errors {
			errs += n
		}
		rows = append(rows, []string{
			name, strconv.Itoa(s.N), strconv.Itoa(s.Agree), rate(s.Agreement), yesNo(s.Deterministic), strconv.Itoa(errs),
			rate(s.Accuracy), usd(s.TotalCostUSD), opt(s.CostPerSuccessUSD, usd),
		})
	}
	var out blocks
	out.add("## Gateway replay parity")
	out.add(fmt.Sprintf("Each prompt of the test split (%d prompts) is sent through the gateway once for every offline decision about it, "+
		"and the model that served it is compared with the one the trainer chose: %d requests.", r.Prompts, r.Requests))
	out.add(table([]string{"Strategy", "Requests", "Agreeing", "Agreement", "Deterministic", "Errors", "Accuracy", "Total cost", "Cost per success"}, rows))
	out.add(
		fmt.Sprintf("- Deterministic agreement: %s (threshold %s for every deterministic strategy)", rate(r.DeterministicAgreement), rate(f.Params.ParityThreshold)),
		fmt.Sprintf("- Overall agreement: %s", rate(r.OverallAgreement)),
		fmt.Sprintf("- Parity: %s", passFail(r.ParityPass)),
	)
	out.add(r.LatencyNote)
	return out
}

// loadKindNotes say what the providers of each kind of load run do.
var loadKindNotes = map[string]string{
	KindLoadZero:     "Providers answer at once.",
	KindLoadProfiled: "Providers take the time of the simulated latency profiles.",
}

// cpuMethodNotes say what each way of measuring CPU use covers.
var cpuMethodNotes = map[string]string{
	CPUMethodProcStat: "the mean of the CPU time the gateway process used between consecutive samples " +
		"(utime and stime from /proc/<pid>/stat), as a percentage of one CPU.",
	CPUMethodPSLifetime: "the mean of the ps pcpu readings, each the CPU time of the gateway process over its whole life so far " +
		"and not over the last sampling interval, as a percentage of one CPU.",
}

func loadBlocks(zero, profiled *LoadFile) ([]string, error) {
	target := profiled.Results.OverheadTarget
	if target == nil {
		return nil, fmt.Errorf("%s has no overhead_target", loadProfiledFileName)
	}
	out := blocks{
		"## Gateway load",
		"Each scenario runs against a gateway started for it. Overhead is the end-to-end time of a request less the time of its provider attempts, " +
			"as the stored decision records it, queueing included. Decision time is the time routing took.",
	}
	for _, f := range []struct {
		kind string
		file *LoadFile
		name string
	}{{KindLoadZero, zero, loadZeroFileName}, {KindLoadProfiled, profiled, loadProfiledFileName}} {
		section, err := loadKindBlocks(f.kind, f.file, f.name)
		if err != nil {
			return nil, err
		}
		out = append(out, section...)
	}
	out.add(overheadLine(*target))
	return out, nil
}

func loadKindBlocks(kind string, f *LoadFile, fileName string) ([]string, error) {
	scenarios := f.Results.Scenarios
	preferred := make([]string, 0, 6)
	for _, sc := range DefaultScenarios() {
		preferred = append(preferred, sc.Name)
	}
	names := orderedKeys(scenarios, preferred)

	var methods []string
	for _, name := range names {
		if m := scenarios[name].CPUMethod; m != "" && !slices.Contains(methods, m) {
			methods = append(methods, m)
		}
	}
	if len(methods) > 1 {
		slices.Sort(methods)
		return nil, fmt.Errorf("%s: the scenarios measured CPU use in more than one way (%s), so one CPU column cannot describe them", fileName, strings.Join(methods, ", "))
	}
	cpuHeader := "CPU mean (%)"
	if len(methods) == 1 {
		cpuHeader = "CPU mean (%, " + methods[0] + ")"
	}

	rows := [][]string{}
	var notes []string
	for _, name := range names {
		sc := scenarios[name]
		rows = append(rows, []string{
			name,
			strconv.Itoa(sc.Requests),
			fixed(sc.ThroughputRPS, 1),
			stat(sc.LatencyMS, "p50", ms), stat(sc.LatencyMS, "p95", ms), stat(sc.LatencyMS, "p99", ms),
			stat(sc.OverheadMS, "p95", ms),
			stat(sc.DecisionUS, "p95", func(x float64) string { return fixed(x, 0) }),
			rate(sc.ErrorRate),
			stat(sc.RSSBytes, "max", mib),
			stat(sc.CPUPercent, "mean", func(x float64) string { return fixed(x, 1) }),
		})
		if sc.SamplingError != "" {
			notes = append(notes, fmt.Sprintf("- Resource sampling of %s failed: %s", name, sc.SamplingError))
		}
		if sc.DecisionsMissing > 0 {
			notes = append(notes, fmt.Sprintf("- %d decisions of %s were not stored when asked for, so its overhead covers fewer requests.", sc.DecisionsMissing, name))
		}
	}

	var out blocks
	out.add("### " + kind)
	if about, ok := loadKindNotes[kind]; ok {
		out.add(about)
	}
	out.add(table([]string{
		"Scenario", "Requests", "Throughput (req/s)", "Latency p50 (ms)", "Latency p95 (ms)", "Latency p99 (ms)",
		"Overhead p95 (ms)", "Decision p95 (us)", "Error rate", "RSS max (MiB)", cpuHeader,
	}, rows))
	if len(methods) == 1 {
		if about, ok := cpuMethodNotes[methods[0]]; ok {
			out.add("CPU method `" + methods[0] + "`: " + about)
		} else {
			out.add("CPU method `" + methods[0] + "`.")
		}
	}
	if len(notes) > 0 {
		out.add(notes...)
	}
	return out, nil
}

// overheadLine is the verdict on the overhead target.
func overheadLine(t OverheadTarget) string {
	return fmt.Sprintf("p95 gateway overhead < %s ms: %s (%s ms, %s)",
		strconv.FormatFloat(t.P95MSBelow, 'f', -1, 64), passFail(t.Pass), ms(t.P95MS), t.Scenario)
}

func faultsBlocks(f *FaultsFile) ([]string, error) {
	r := f.Results
	benchBooks, ok := r.Accounting[BenchTenant]
	if !ok {
		return nil, fmt.Errorf("%s has no accounting for tenant %q", faultsFileName, BenchTenant)
	}
	overBooks, ok := r.Accounting[OverTenant]
	if !ok {
		return nil, fmt.Errorf("%s has no accounting for tenant %q", faultsFileName, OverTenant)
	}

	phaseRows := [][]string{}
	for _, p := range []PhaseResult{r.Bench, r.BenchOver} {
		phaseRows = append(phaseRows, []string{
			p.Tenant, strconv.Itoa(p.Requests), strconv.Itoa(p.StreamRequests), rate(p.SuccessRate),
			strconv.Itoa(p.PartialStreams), strconv.Itoa(p.Retries), strconv.Itoa(p.Failovers),
		})
	}
	inv := r.Invariants
	invRows := [][]string{
		{"a. Spend equals the sum of usage records", holdsBroken(inv.SpentEqualsUsage),
			fmt.Sprintf("%s: spent %s, usage %s; %s: spent %s, usage %s",
				BenchTenant, nanoUSD(benchBooks.SpentNanos), nanoUSD(benchBooks.UsageCostNanos),
				OverTenant, nanoUSD(overBooks.SpentNanos), nanoUSD(overBooks.UsageCostNanos))},
		{"b. Reservations return to zero", holdsBroken(inv.ReservedReturnsToZero),
			fmt.Sprintf("%s: reserved %s; %s: reserved %s",
				BenchTenant, nanoUSD(benchBooks.ReservedNanos), OverTenant, nanoUSD(overBooks.ReservedNanos))},
		{"c. Spend stays within budget (" + BenchTenant + ")", holdsBroken(inv.SpendWithinBudget),
			fmt.Sprintf("spent %s, budget %s", nanoUSD(benchBooks.SpentNanos), nanoUSD(benchBooks.BudgetNanos))},
		{"d. Overshoot equals spend beyond budget (" + OverTenant + ")", holdsBroken(inv.OvershootEqualsExcess),
			fmt.Sprintf("overshoot %s, spent %s, budget %s",
				nanoUSD(overBooks.OvershootNanos), nanoUSD(overBooks.SpentNanos), nanoUSD(overBooks.BudgetNanos))},
		{"d. Overshoot is positive (" + OverTenant + ")", holdsBroken(inv.OvershootPositive),
			"overshoot " + nanoUSD(overBooks.OvershootNanos)},
	}

	var out blocks
	out.add("## Fault injection")
	out.add("The providers fail, time out, cut streams short and leave usage out. " +
		"The accounting invariants are checked on the books the gateway reports after the traffic.")
	out.add(table([]string{"Tenant", "Requests", "Streaming", "Success rate", "Partial streams", "Retries", "Failovers"}, phaseRows))
	out.add(table([]string{"Invariant", "Status", "Books"}, invRows))
	out.add("Accounting invariants: " + holdsBroken(inv.InvariantsHold))
	return out, nil
}

func reproduceBlocks(runID string) []string {
	return blocks{
		"## Reproduce",
		"Run from the repository root. The offline numbers come from the trainer and the gateway numbers from the benchmark driver. " +
			"The last command regenerates this file and the results block of the README.",
		strings.Join([]string{
			"```sh",
			"make data",
			"make train",
			"make evaluate",
			"make bench-replay RUN_ID=" + runID,
			"make bench-load RUN_ID=" + runID,
			"make bench-faults RUN_ID=" + runID,
			"make report RUN_ID=" + runID,
			"```",
		}, "\n"),
	}
}
