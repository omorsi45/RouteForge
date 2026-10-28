package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
)

// unversionedKeys are the top-level keys left out of the version hash.
var unversionedKeys = []string{"version", "created_at"}

const hexDigits = "0123456789abcdef"

// decodeTree decodes one JSON value into a generic tree. Numbers stay
// json.Number so their text survives to the hash, and trailing data other
// than whitespace is an error.
func decodeTree(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("trailing data after JSON value")
	}
	return tree, nil
}

// canonicalBytes encodes a tree from decodeTree as the trainer's
// canonical_bytes does: keys sorted by code point, separators "," and ":",
// strings escaped like Python json.dumps(ensure_ascii=False), and numbers
// copied from their source text.
func canonicalBytes(tree any) ([]byte, error) {
	var buf bytes.Buffer
	if err := writeCanonical(&buf, tree); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeCanonical(buf *bytes.Buffer, v any) error {
	switch x := v.(type) {
	case nil:
		buf.WriteString("null")
	case bool:
		if x {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case json.Number:
		buf.WriteString(x.String())
	case string:
		writeCanonicalString(buf, x)
	case []any:
		buf.WriteByte('[')
		for i, item := range x {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := writeCanonical(buf, item); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case map[string]any:
		buf.WriteByte('{')
		// UTF-8 byte order is code point order, which is Python's key order.
		for i, key := range slices.Sorted(maps.Keys(x)) {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeCanonicalString(buf, key)
			buf.WriteByte(':')
			if err := writeCanonical(buf, x[key]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		return fmt.Errorf("unsupported value of type %T", v)
	}
	return nil
}

// writeCanonicalString writes s like Python json.dumps(ensure_ascii=False):
// the quote, the backslash and \b \f \n \r \t get two-character escapes, any
// other byte below 0x20 becomes \u00xx in lowercase hex, and everything else
// is written raw (including DEL and U+2028).
func writeCanonicalString(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; c {
		case '"':
			buf.WriteString(`\"`)
		case '\\':
			buf.WriteString(`\\`)
		case '\b':
			buf.WriteString(`\b`)
		case '\f':
			buf.WriteString(`\f`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		default:
			if c < 0x20 {
				buf.WriteString(`\u00`)
				buf.WriteByte(hexDigits[c>>4])
				buf.WriteByte(hexDigits[c&0xf])
			} else {
				buf.WriteByte(c)
			}
		}
	}
	buf.WriteByte('"')
}

// versionOf hashes the canonical form of the document without its version
// and created_at: the first 12 lowercase hex characters of the sha256.
func versionOf(tree any) (string, error) {
	doc, ok := tree.(map[string]any)
	if !ok {
		return "", errors.New("document is not a JSON object")
	}
	body := make(map[string]any, len(doc))
	for key, value := range doc {
		if !slices.Contains(unversionedKeys, key) {
			body[key] = value
		}
	}
	canonical, err := canonicalBytes(body)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:6]), nil
}

// CanonicalVersion computes the version of a policy document the way the
// trainer's policy_version does. Numbers are hashed from their source text,
// so the document must be in the trainer's canonical number form.
func CanonicalVersion(data []byte) (string, error) {
	tree, err := decodeTree(data)
	if err != nil {
		return "", fmt.Errorf("decode policy document: %w", err)
	}
	version, err := versionOf(tree)
	if err != nil {
		return "", fmt.Errorf("compute policy version: %w", err)
	}
	return version, nil
}
