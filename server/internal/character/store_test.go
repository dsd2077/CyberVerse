package character

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadLegacyCharacterStripsAvatarModelOnSave(t *testing.T) {
	baseDir := t.TempDir()
	charID := "123e4567-e89b-12d3-a456-426614174000"
	charDir := filepath.Join(baseDir, charDirName("Legacy", charID))
	if err := os.MkdirAll(charDir, 0755); err != nil {
		t.Fatal(err)
	}

	legacy := map[string]any{
		"id":              charID,
		"name":            "Legacy",
		"description":     "legacy payload",
		"avatar_image":    "",
		"use_face_crop":   false,
		"voice_provider":  "doubao",
		"voice_type":      "温柔文雅",
		"avatar_model":    "flash_head",
		"speaking_style":  "平静",
		"personality":     "稳定",
		"welcome_message": "你好",
		"system_prompt":   "legacy system prompt",
		"tags":            []string{"legacy"},
		"images":          []any{},
		"active_image":    "",
		"image_mode":      "fixed",
		"created_at":      "2026-04-18T00:00:00Z",
		"updated_at":      "2026-04-18T00:00:00Z",
	}
	data, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(charDir, "character.json"), data, 0644); err != nil {
		t.Fatal(err)
	}

	store, err := NewStore(baseDir)
	if err != nil {
		t.Fatal(err)
	}

	char, err := store.Get(charID)
	if err != nil {
		t.Fatal(err)
	}
	if char.Name != "Legacy" {
		t.Fatalf("expected Legacy, got %q", char.Name)
	}

	updated := &Character{
		Name:           char.Name,
		Description:    "updated legacy payload",
		AvatarImage:    char.AvatarImage,
		UseFaceCrop:    char.UseFaceCrop,
		VoiceProvider:  char.VoiceProvider,
		VoiceType:      char.VoiceType,
		SpeakingStyle:  char.SpeakingStyle,
		Personality:    char.Personality,
		WelcomeMessage: char.WelcomeMessage,
		SystemPrompt:   char.SystemPrompt,
		Tags:           append([]string(nil), char.Tags...),
		ImageMode:      char.ImageMode,
	}
	if _, err := store.Update(charID, updated); err != nil {
		t.Fatal(err)
	}

	saved, err := os.ReadFile(filepath.Join(charDir, "character.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(saved), "\"avatar_model\"") {
		t.Fatalf("expected saved character.json to omit avatar_model, got %s", string(saved))
	}

	var savedJSON map[string]any
	if err := json.Unmarshal(saved, &savedJSON); err != nil {
		t.Fatal(err)
	}
	if _, ok := savedJSON["avatar_model"]; ok {
		t.Fatalf("expected avatar_model to be removed from saved JSON")
	}
}

func TestIdleVideoFilenameIncludesResolutionVariant(t *testing.T) {
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	got := store.IdleVideoFilename("img_003.png", DefaultIdleVideoProfile)
	want := "img_003__breathing10s_v1.mp4"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}
