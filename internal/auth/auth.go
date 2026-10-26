// Package auth hashes API keys, generates new ones and maps a presented key
// to its tenant. Keys are never stored: only their SHA-256 digests are.
package auth

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/omorsi45/routeforge/internal/config"
)

// KeyPrefix starts every generated API key.
const KeyPrefix = "rf_"

const keyBytes = 32

// HashKey returns the lowercase hex SHA-256 digest of the exact bytes of key.
func HashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// GenerateKey returns KeyPrefix followed by the unpadded base64url encoding of
// 32 bytes read from r.
func GenerateKey(r io.Reader) (string, error) {
	buf := make([]byte, keyBytes)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("read random bytes: %w", err)
	}
	return KeyPrefix + base64.RawURLEncoding.EncodeToString(buf), nil
}

// Store maps key digests to tenants.
type Store struct {
	byDigest map[string]*config.Tenant
}

// NewStore indexes tenants by each of their key digests. It assumes
// config.Validate has passed, so digests are well formed and unique. The store
// holds its own copies of the tenants.
func NewStore(tenants []config.Tenant) *Store {
	s := &Store{byDigest: make(map[string]*config.Tenant)}
	for _, t := range tenants {
		cp := t
		cp.APIKeySHA256 = slices.Clone(t.APIKeySHA256)
		cp.AllowedModels = slices.Clone(t.AllowedModels)
		for _, digest := range cp.APIKeySHA256 {
			s.byDigest[digest] = &cp
		}
	}
	return s
}

// Lookup returns the tenant owning key. The match is by digest only, so the
// plaintext key is never compared and its shape is never inspected. An empty
// key never matches.
func (s *Store) Lookup(key string) (*config.Tenant, bool) {
	if key == "" {
		return nil, false
	}
	t, ok := s.byDigest[HashKey(key)]
	return t, ok
}

// BearerToken extracts the token from an "Authorization: Bearer <token>"
// header. The scheme is case-insensitive and surrounding spaces are trimmed.
// A missing header, another scheme or an empty token returns false.
func BearerToken(h http.Header) (string, bool) {
	scheme, token, found := strings.Cut(strings.TrimSpace(h.Get("Authorization")), " ")
	if !found || !strings.EqualFold(scheme, "Bearer") {
		return "", false
	}
	token = strings.TrimSpace(token)
	if token == "" {
		return "", false
	}
	return token, true
}
