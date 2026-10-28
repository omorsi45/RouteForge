package accounting

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const priceHeader = `effective_date: "2026-10-26"
source: "test table"
currency: USD
`

func writePrices(t *testing.T, doc string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "prices.yaml")
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadPricesCommitted(t *testing.T) {
	tab, err := LoadPrices("../../deploy/prices.yaml")
	if err != nil {
		t.Fatalf("LoadPrices: %v", err)
	}
	if tab.EffectiveDate != "2026-10-26" || tab.Currency != "USD" || tab.Source == "" {
		t.Errorf("header = %q, %q, %q", tab.EffectiveDate, tab.Currency, tab.Source)
	}
	if len(tab.byName) != 6 {
		t.Fatalf("table has %d models, want 6", len(tab.byName))
	}
	want := []struct {
		name              string
		in, out           float64
		inNanos, outNanos int64
	}{
		{"gpt-5", 1.25, 10, 1250, 10000},
		{"claude-sonnet-4", 3, 15, 3000, 15000},
		{"gemini-2.5-pro", 1.25, 10, 1250, 10000},
		{"gemini-2.5-flash", 0.30, 2.50, 300, 2500},
		{"deepseek-v3.1-terminus", 0.27, 1, 270, 1000},
		{"qwen3-235b-a22b-2507", 0.09, 0.60, 90, 600},
	}
	for _, w := range want {
		p, ok := tab.Price(w.name)
		if !ok {
			t.Errorf("%s: not found", w.name)
			continue
		}
		if p.Name != w.name || p.InputPerMTok != w.in || p.OutputPerMTok != w.out {
			t.Errorf("%s: price = %+v", w.name, p)
		}
		if p.InNanos != w.inNanos || p.OutNanos != w.outNanos {
			t.Errorf("%s: nanos = %d / %d, want %d / %d", w.name, p.InNanos, p.OutNanos, w.inNanos, w.outNanos)
		}
	}
	if _, ok := tab.Price("nope"); ok {
		t.Error("Price(nope) found")
	}
}

func TestLoadPricesRejectsSubNanoRate(t *testing.T) {
	path := writePrices(t, priceHeader+`models:
  - name: tiny-model
    input_per_mtok: 0.0001
    output_per_mtok: 1.0
`)
	_, err := LoadPrices(path)
	if err == nil || !strings.Contains(err.Error(), "tiny-model") {
		t.Fatalf("err = %v, want one naming tiny-model", err)
	}
	if !strings.Contains(err.Error(), "input_per_mtok") {
		t.Errorf("err = %v, want it to name input_per_mtok", err)
	}
}

func TestLoadPricesRejectsCurrency(t *testing.T) {
	path := writePrices(t, strings.Replace(priceHeader, "USD", "EUR", 1)+`models:
  - {name: m, input_per_mtok: 1.0, output_per_mtok: 2.0}
`)
	_, err := LoadPrices(path)
	if err == nil || !strings.Contains(err.Error(), "currency") || !strings.Contains(err.Error(), "EUR") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadPricesRejects(t *testing.T) {
	tests := []struct {
		name string
		doc  string
		want string
	}{
		{"empty", "", "price table is empty"},
		{"multi document", priceHeader + "models:\n  - {name: m, input_per_mtok: 1, output_per_mtok: 1}\n---\n" + priceHeader, "multiple YAML documents"},
		{"unknown field", priceHeader + "models:\n  - {name: m, input_per_mtok: 1, output_per_mtok: 1, cached: 1}\n", "cached"},
		{"unknown top level field", priceHeader + "region: us\nmodels:\n  - {name: m, input_per_mtok: 1, output_per_mtok: 1}\n", "region"},
		{"no models", priceHeader + "models: []\n", "at least one model"},
		{"empty name", priceHeader + "models:\n  - {name: \"\", input_per_mtok: 1, output_per_mtok: 1}\n", "models[0].name: must not be empty"},
		{"duplicate", priceHeader + "models:\n  - {name: m, input_per_mtok: 1, output_per_mtok: 1}\n  - {name: m, input_per_mtok: 2, output_per_mtok: 2}\n", `"m"`},
		{"missing input", priceHeader + "models:\n  - {name: m, output_per_mtok: 1}\n", `model "m": input_per_mtok is required`},
		{"missing output", priceHeader + "models:\n  - {name: m, input_per_mtok: 1}\n", `model "m": output_per_mtok is required`},
		{"negative", priceHeader + "models:\n  - {name: m, input_per_mtok: -1, output_per_mtok: 1}\n", `model "m": input_per_mtok`},
		{"nan", priceHeader + "models:\n  - {name: m, input_per_mtok: 1, output_per_mtok: .nan}\n", `model "m": output_per_mtok`},
		{"inf", priceHeader + "models:\n  - {name: m, input_per_mtok: .inf, output_per_mtok: 1}\n", `model "m": input_per_mtok`},
		{"too large for exact nanos", priceHeader + "models:\n  - {name: m, input_per_mtok: 1e15, output_per_mtok: 1}\n", `model "m": input_per_mtok`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadPrices(writePrices(t, tc.doc))
			if err == nil {
				t.Fatal("LoadPrices returned nil")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestLoadPricesReportsEveryBadModel(t *testing.T) {
	path := writePrices(t, priceHeader+`models:
  - {name: first, input_per_mtok: 0.0001, output_per_mtok: 1}
  - {name: second, input_per_mtok: 1, output_per_mtok: 0.0002}
`)
	_, err := LoadPrices(path)
	if err == nil {
		t.Fatal("LoadPrices returned nil")
	}
	for _, name := range []string{"first", "second"} {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("error %q does not name %s", err, name)
		}
	}
}

func TestLoadPricesMissingFile(t *testing.T) {
	_, err := LoadPrices(filepath.Join(t.TempDir(), "absent.yaml"))
	if err == nil || !strings.Contains(err.Error(), "read prices") {
		t.Fatalf("err = %v", err)
	}
}

func TestNewPrice(t *testing.T) {
	p, err := NewPrice("m", 0.27, 1)
	if err != nil {
		t.Fatalf("NewPrice: %v", err)
	}
	if p.Name != "m" || p.InputPerMTok != 0.27 || p.OutputPerMTok != 1 || p.InNanos != 270 || p.OutNanos != 1000 {
		t.Errorf("price = %+v", p)
	}
	free, err := NewPrice("local", 0, 0)
	if err != nil || free.InNanos != 0 || free.OutNanos != 0 {
		t.Errorf("free price = %+v, %v", free, err)
	}
	// Three decimals is the finest rate that is a whole nano-USD per token.
	fine, err := NewPrice("m", 0.001, 0.003)
	if err != nil || fine.InNanos != 1 || fine.OutNanos != 3 {
		t.Errorf("fine price = %+v, %v", fine, err)
	}
	if _, err := NewPrice("m", 1.0005, 1); err == nil || !strings.Contains(err.Error(), `"m"`) {
		t.Errorf("sub-nano err = %v", err)
	}
}
