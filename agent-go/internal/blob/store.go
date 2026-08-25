package blob

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

type Store struct { root string }

func New(root string) (*Store, error) {
	if err := os.MkdirAll(root, 0o755); err != nil { return nil, err }
	return &Store{root: root}, nil
}

func (s *Store) path(blobID string) string { return filepath.Join(s.root, blobID) }

func (s *Store) Put(blobID string, data []byte) (int, error) {
	want := sha256.Sum256(data)
	if blobID != hex.EncodeToString(want[:]) { return 0, fmt.Errorf("blob id mismatch") }
	if err := os.WriteFile(s.path(blobID), data, 0o644); err != nil { return 0, err }
	return len(data), nil
}

func (s *Store) Get(blobID string) ([]byte, bool, error) {
	b, err := os.ReadFile(s.path(blobID))
	if err != nil {
		if os.IsNotExist(err) { return nil, false, nil }
		return nil, false, err
	}
	return b, true, nil
}

func (s *Store) Has(blobID string) bool { _, err := os.Stat(s.path(blobID)); return err == nil }

func (s *Store) Delete(blobID string) error {
	err := os.Remove(s.path(blobID))
	if os.IsNotExist(err) { return nil }
	return err
}
