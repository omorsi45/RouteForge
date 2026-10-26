package provider

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// MaxErrorBodyBytes caps how much of an upstream error body an adapter reads.
const MaxErrorBodyBytes = 64 << 10

// DecodeJSON decodes exactly one JSON value from data into v, rejecting any
// trailing data other than whitespace.
func DecodeJSON(data []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data after JSON value")
	}
	return nil
}
