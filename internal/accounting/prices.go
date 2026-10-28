// Package accounting prices requests and tracks spend in exact integer money.
package accounting

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"go.yaml.in/yaml/v3"
)

// priceCurrency is the only currency the price table may state.
const priceCurrency = "USD"

// nanosPerUnitRate converts a price in USD per 1M tokens into nano-USD per
// token: x USD / 1e6 tokens = x * 1e3 nano-USD per token.
const nanosPerUnitRate = 1000

// maxExactNanos is the largest per-token price (2^53 nano-USD) at which a
// float64 still represents every whole number, so the conversion stays exact.
const maxExactNanos = 1 << 53

// nanoTolerance is how far rate*1000 may sit from a whole number before the
// rate is rejected as finer than one nano-USD per token.
const nanoTolerance = 1e-9

// Price is one model's list price. Money is an int64 count of nano-USD
// (1e-9 USD); InNanos and OutNanos are nano-USD per token.
type Price struct {
	Name          string
	InputPerMTok  float64
	OutputPerMTok float64
	InNanos       int64
	OutNanos      int64
}

// NewPrice builds a Price from rates in USD per 1M tokens. Each rate must be
// finite, not negative and a whole number of nano-USD per token, which means
// at most three decimals. Errors name the model.
func NewPrice(name string, in, out float64) (Price, error) {
	inNanos, err := nanosPerToken(in)
	if err != nil {
		return Price{}, fmt.Errorf("model %q: input_per_mtok %v: %w", name, in, err)
	}
	outNanos, err := nanosPerToken(out)
	if err != nil {
		return Price{}, fmt.Errorf("model %q: output_per_mtok %v: %w", name, out, err)
	}
	return Price{Name: name, InputPerMTok: in, OutputPerMTok: out, InNanos: inNanos, OutNanos: outNanos}, nil
}

func nanosPerToken(rate float64) (int64, error) {
	if math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 0 {
		return 0, errors.New("must be finite and not negative")
	}
	scaled := rate * nanosPerUnitRate
	whole := math.Round(scaled)
	if math.Abs(scaled-whole) > nanoTolerance {
		return 0, errors.New("must be a whole number of nano-USD per token (at most 3 decimals)")
	}
	if whole > maxExactNanos {
		return 0, errors.New("too large to convert to nano-USD exactly")
	}
	return int64(whole), nil
}

// PriceTable is the loaded price file.
type PriceTable struct {
	EffectiveDate string
	Source        string
	Currency      string
	byName        map[string]Price
}

// Price returns the price of the named model.
func (t *PriceTable) Price(name string) (Price, bool) {
	p, ok := t.byName[name]
	return p, ok
}

type priceFile struct {
	EffectiveDate string       `yaml:"effective_date"`
	Source        string       `yaml:"source"`
	Currency      string       `yaml:"currency"`
	Models        []priceEntry `yaml:"models"`
}

// priceEntry uses pointers so a missing rate is an error, not a free model.
type priceEntry struct {
	Name          string   `yaml:"name"`
	InputPerMTok  *float64 `yaml:"input_per_mtok"`
	OutputPerMTok *float64 `yaml:"output_per_mtok"`
}

// LoadPrices reads and strictly decodes the price file at path. It rejects
// unknown fields, a currency other than USD, duplicate or unnamed models,
// missing rates and rates that are not a whole number of nano-USD per token.
// Errors name the file and every bad model.
func LoadPrices(path string) (*PriceTable, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read prices: %w", err)
	}
	t, err := parsePrices(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

func parsePrices(data []byte) (*PriceTable, error) {
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)

	var f priceFile
	if err := dec.Decode(&f); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, errors.New("price table is empty")
		}
		return nil, fmt.Errorf("decode price table: %w", err)
	}
	var extra yaml.Node
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, errors.New("multiple YAML documents are not supported")
	}

	var errs []error
	if f.Currency != priceCurrency {
		errs = append(errs, fmt.Errorf("currency %q: must be %s", f.Currency, priceCurrency))
	}
	if len(f.Models) == 0 {
		errs = append(errs, errors.New("models: at least one model is required"))
	}

	t := &PriceTable{
		EffectiveDate: f.EffectiveDate,
		Source:        f.Source,
		Currency:      f.Currency,
		byName:        make(map[string]Price, len(f.Models)),
	}
	for i, e := range f.Models {
		if e.Name == "" {
			errs = append(errs, fmt.Errorf("models[%d].name: must not be empty", i))
			continue
		}
		if _, dup := t.byName[e.Name]; dup {
			errs = append(errs, fmt.Errorf("models[%d]: duplicate model %q", i, e.Name))
			continue
		}
		switch {
		case e.InputPerMTok == nil:
			errs = append(errs, fmt.Errorf("model %q: input_per_mtok is required", e.Name))
		case e.OutputPerMTok == nil:
			errs = append(errs, fmt.Errorf("model %q: output_per_mtok is required", e.Name))
		default:
			p, err := NewPrice(e.Name, *e.InputPerMTok, *e.OutputPerMTok)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			t.byName[e.Name] = p
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return t, nil
}
