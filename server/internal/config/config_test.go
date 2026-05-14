package config

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTestConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cyberverse_config.yaml")
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestAvatarEnabledDefaultsTrue(t *testing.T) {
	cfg, err := Load(writeTestConfig(t, "inference: {}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.AvatarEnabled() {
		t.Fatal("expected avatar to default enabled")
	}
	if cfg.Pipeline.AvatarEnabled == nil || !*cfg.Pipeline.AvatarEnabled {
		t.Fatalf("expected pipeline avatar enabled pointer to be true, got %#v", cfg.Pipeline.AvatarEnabled)
	}
}

func TestAvatarEnabledCanBeDisabled(t *testing.T) {
	cfg, err := Load(writeTestConfig(t, `
inference:
  avatar:
    enabled: false
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.AvatarEnabled() {
		t.Fatal("expected avatar to be disabled")
	}
	if cfg.Pipeline.AvatarEnabled == nil || *cfg.Pipeline.AvatarEnabled {
		t.Fatalf("expected pipeline avatar enabled pointer to be false, got %#v", cfg.Pipeline.AvatarEnabled)
	}
}
