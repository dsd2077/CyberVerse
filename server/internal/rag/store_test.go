package rag

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cyberverse/server/internal/character"
)

func newTestStore(t *testing.T) (*Store, string) {
	t.Helper()
	charStore, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	created, err := charStore.Create(&character.Character{Name: "RAG Test"})
	if err != nil {
		t.Fatalf("Create character: %v", err)
	}
	return NewStore(charStore), created.ID
}

func TestStoreSaveFilePreservesRelativePathAndStatusLifecycle(t *testing.T) {
	store, characterID := newTestStore(t)

	result, err := store.SaveFile(characterID, "角色 资料/profile v1.md", "text/markdown", strings.NewReader("出生在海边。后来成为工程师。"))
	if err != nil {
		t.Fatalf("SaveFile: %v", err)
	}
	source := result.Source
	if source.Status != SourceStatusIndexing {
		t.Fatalf("expected indexing status, got %q", source.Status)
	}
	if !source.Indexable {
		t.Fatal("expected markdown source to be indexable")
	}
	if source.Type != "" {
		t.Fatalf("expected new sources to omit type, got %q", source.Type)
	}
	if source.Title != "profile v1" {
		t.Fatalf("expected title from filename, got %q", source.Title)
	}
	if source.RelativePath != "角色 资料/profile v1.md" || source.StoredPath != "sources/角色 资料/profile v1.md" {
		t.Fatalf("unexpected source paths: %+v", source)
	}
	if !strings.HasSuffix(result.Path, filepath.Join("knowledge", "sources", "角色 资料", "profile v1.md")) {
		t.Fatalf("unexpected source path: %s", result.Path)
	}

	ready, err := store.MarkReady(characterID, source.ID, 3)
	if err != nil {
		t.Fatalf("MarkReady: %v", err)
	}
	if ready.Status != SourceStatusReady || ready.ChunkCount != 3 || ready.IndexedAt == "" {
		t.Fatalf("unexpected ready source: %+v", ready)
	}

	failed, err := store.MarkFailed(characterID, source.ID, errTestFailure{})
	if err != nil {
		t.Fatalf("MarkFailed: %v", err)
	}
	if failed.Status != SourceStatusFailed || failed.Error == "" {
		t.Fatalf("unexpected failed source: %+v", failed)
	}
}

func TestStoreSavesNonIndexableFileAsReady(t *testing.T) {
	store, characterID := newTestStore(t)

	result, err := store.SaveFile(characterID, "images/avatar.png", "image/png", strings.NewReader("png"))
	if err != nil {
		t.Fatalf("SaveFile: %v", err)
	}
	if result.Source.Indexable {
		t.Fatalf("expected image source to be stored only, got %+v", result.Source)
	}
	if result.Source.Status != SourceStatusReady || result.Source.ChunkCount != 0 {
		t.Fatalf("expected ready/0 chunks for image, got %+v", result.Source)
	}
}

func TestStoreRejectsUnsafeRelativePath(t *testing.T) {
	store, characterID := newTestStore(t)

	if _, err := store.SaveFile(characterID, "../secret.txt", "text/plain", strings.NewReader("nope")); err == nil {
		t.Fatal("expected unsafe relative path error")
	}
}

func TestStoreDeleteRemovesSource(t *testing.T) {
	store, characterID := newTestStore(t)

	result, err := store.SaveFile(characterID, "folder/rules.txt", "text/plain", strings.NewReader("只回答事实。"))
	if err != nil {
		t.Fatalf("SaveFile: %v", err)
	}
	source := result.Source
	if err := store.Delete(characterID, source.ID); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	sources, err := store.List(characterID)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(sources) != 0 {
		t.Fatalf("expected no sources after delete, got %+v", sources)
	}
	if _, err := os.Stat(result.Path); !os.IsNotExist(err) {
		t.Fatalf("expected stored file to be removed, stat err=%v", err)
	}
}

func TestStoreOverwriteSameRelativePathReusesSource(t *testing.T) {
	store, characterID := newTestStore(t)

	first, err := store.SaveFile(characterID, "docs/profile.md", "text/markdown", strings.NewReader("first"))
	if err != nil {
		t.Fatalf("SaveFile first: %v", err)
	}
	second, err := store.SaveFile(characterID, "docs/profile.md", "text/markdown", strings.NewReader("second"))
	if err != nil {
		t.Fatalf("SaveFile second: %v", err)
	}
	if second.Created || first.Source.ID != second.Source.ID {
		t.Fatalf("expected same source to be updated, first=%+v second=%+v", first.Source, second.Source)
	}
	data, err := os.ReadFile(second.Path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "second" {
		t.Fatalf("expected overwritten file content, got %q", string(data))
	}
	sources, err := store.List(characterID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 1 {
		t.Fatalf("expected one source after overwrite, got %+v", sources)
	}
}

func TestStoreSameBasenameDifferentDirectoriesDoNotConflict(t *testing.T) {
	store, characterID := newTestStore(t)

	first, err := store.SaveFile(characterID, "a/profile.md", "text/markdown", strings.NewReader("a"))
	if err != nil {
		t.Fatalf("SaveFile first: %v", err)
	}
	second, err := store.SaveFile(characterID, "b/profile.md", "text/markdown", strings.NewReader("b"))
	if err != nil {
		t.Fatalf("SaveFile second: %v", err)
	}
	if first.Source.ID == second.Source.ID {
		t.Fatalf("expected different source IDs for different relative paths")
	}
	sources, err := store.List(characterID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 2 {
		t.Fatalf("expected two sources, got %+v", sources)
	}
}

func TestStoreReadsLegacySourceType(t *testing.T) {
	store, characterID := newTestStore(t)
	knowledgeDir, err := store.KnowledgeDir(characterID)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(knowledgeDir, 0755); err != nil {
		t.Fatal(err)
	}
	legacy := []Source{{
		ID:             "legacy-source",
		Type:           SourceType("biography"),
		Title:          "旧人物生平",
		Filename:       "bio.txt",
		Status:         SourceStatusReady,
		CreatedAt:      nowString(),
		UpdatedAt:      nowString(),
		StoredFilename: "bio.txt",
	}}
	data, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(knowledgeDir, "sources.json"), data, 0644); err != nil {
		t.Fatal(err)
	}

	sources, err := store.List(characterID)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) != 1 || sources[0].Type != SourceType("biography") {
		t.Fatalf("expected legacy type to round-trip, got %+v", sources)
	}
}

func TestStoreLegacySourcePathAndDelete(t *testing.T) {
	store, characterID := newTestStore(t)
	knowledgeDir, err := store.KnowledgeDir(characterID)
	if err != nil {
		t.Fatal(err)
	}
	legacyDir := filepath.Join(knowledgeDir, "sources", "legacy-source")
	if err := os.MkdirAll(legacyDir, 0755); err != nil {
		t.Fatal(err)
	}
	legacyPath := filepath.Join(legacyDir, "bio.txt")
	if err := os.WriteFile(legacyPath, []byte("legacy"), 0644); err != nil {
		t.Fatal(err)
	}
	legacy := []Source{{
		ID:             "legacy-source",
		Title:          "旧素材",
		Filename:       "bio.txt",
		Status:         SourceStatusReady,
		CreatedAt:      nowString(),
		UpdatedAt:      nowString(),
		StoredFilename: "bio.txt",
	}}
	data, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(knowledgeDir, "sources.json"), data, 0644); err != nil {
		t.Fatal(err)
	}

	sources, err := store.List(characterID)
	if err != nil {
		t.Fatal(err)
	}
	path, err := store.SourcePath(characterID, &sources[0])
	if err != nil {
		t.Fatal(err)
	}
	if path != legacyPath {
		t.Fatalf("expected legacy source path %q, got %q", legacyPath, path)
	}
	if err := store.Delete(characterID, "legacy-source"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(legacyDir); !os.IsNotExist(err) {
		t.Fatalf("expected legacy source dir to be removed, stat err=%v", err)
	}
}

func TestStoreSourcePathPrefersRelativePathForOldMetadata(t *testing.T) {
	store, characterID := newTestStore(t)
	knowledgeDir, err := store.KnowledgeDir(characterID)
	if err != nil {
		t.Fatal(err)
	}
	newPath := filepath.Join(knowledgeDir, "sources", "资料", "九州.md")
	if err := os.MkdirAll(filepath.Dir(newPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newPath, []byte("new path"), 0644); err != nil {
		t.Fatal(err)
	}
	legacy := []Source{{
		ID:             "legacy-source",
		Title:          "九州",
		Filename:       "九州.md",
		RelativePath:   "资料/九州.md",
		Status:         SourceStatusReady,
		CreatedAt:      nowString(),
		UpdatedAt:      nowString(),
		StoredFilename: "九州.md",
	}}
	data, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(knowledgeDir, "sources.json"), data, 0644); err != nil {
		t.Fatal(err)
	}

	sources, err := store.List(characterID)
	if err != nil {
		t.Fatal(err)
	}
	if !sources[0].Indexable {
		t.Fatalf("expected old markdown metadata to normalize as indexable: %+v", sources[0])
	}
	path, err := store.SourcePath(characterID, &sources[0])
	if err != nil {
		t.Fatal(err)
	}
	if path != newPath {
		t.Fatalf("expected source path %q, got %q", newPath, path)
	}
}

type errTestFailure struct{}

func (errTestFailure) Error() string { return "index failed" }
