package identity

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base32"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

type Identity struct {
	DeviceID      string `json:"device_id"`
	PublicKey     string `json:"public_key"`
	PrivateKeyPEM string `json:"private_key_pem"`
	privateKey    ed25519.PrivateKey
}

func LoadOrCreate(dataDir string) (Identity, error) {
	path := filepath.Join(dataDir, "identity.json")
	if raw, err := os.ReadFile(path); err == nil {
		var id Identity
		if err := json.Unmarshal(raw, &id); err == nil {
			priv, err := parsePrivateKeyPEM(id.PrivateKeyPEM)
			if err == nil {
				id.privateKey = priv
				return id, nil
			}
		}
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return Identity{}, err
	}
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return Identity{}, err
	}
	privDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return Identity{}, err
	}
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privDER})
	id := Identity{
		DeviceID:      DeviceIDFromPublicKey(pub),
		PublicKey:     base64.StdEncoding.EncodeToString(pub),
		PrivateKeyPEM: string(pemBytes),
		privateKey:    priv,
	}
	raw, err := json.MarshalIndent(id, "", "  ")
	if err != nil {
		return Identity{}, err
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return Identity{}, err
	}
	return id, nil
}

func DeviceIDFromPublicKey(pub ed25519.PublicKey) string {
	sum := sha256.Sum256(pub)
	enc := base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(sum[:15])
	return "dev_" + lower(enc)
}

func EnrollmentChallenge(deviceID string, tsMS int64) []byte {
	return []byte(fmt.Sprintf("meshpool-enroll:%s:%d:", deviceID, tsMS))
}

func (i Identity) SignEnrollment(tsMS int64) (string, error) {
	return i.Sign(EnrollmentChallenge(i.DeviceID, tsMS))
}

func (i Identity) Sign(data []byte) (string, error) {
	if len(i.privateKey) == 0 {
		priv, err := parsePrivateKeyPEM(i.PrivateKeyPEM)
		if err != nil {
			return "", err
		}
		i.privateKey = priv
	}
	sig := ed25519.Sign(i.privateKey, data)
	return base64.StdEncoding.EncodeToString(sig), nil
}

func VerifyEnrollment(publicKeyB64, deviceID, signature string, tsMS int64) bool {
	pub, err := base64.StdEncoding.DecodeString(publicKeyB64)
	if err != nil {
		return false
	}
	if DeviceIDFromPublicKey(ed25519.PublicKey(pub)) != deviceID {
		return false
	}
	sig, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		return false
	}
	return ed25519.Verify(ed25519.PublicKey(pub), EnrollmentChallenge(deviceID, tsMS), sig)
}

func ChallengeIsFresh(tsMS, nowMS int64) bool {
	delta := nowMS - tsMS
	if delta < 0 {
		delta = -delta
	}
	return delta <= int64(5*time.Minute/time.Millisecond)
}

func parsePrivateKeyPEM(txt string) (ed25519.PrivateKey, error) {
	block, _ := pem.Decode([]byte(txt))
	if block == nil {
		return nil, fmt.Errorf("decode pem")
	}
	key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	priv, ok := key.(ed25519.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("not ed25519")
	}
	return priv, nil
}

func lower(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + 32
		}
	}
	return string(b)
}
