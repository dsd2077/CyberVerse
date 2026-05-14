package rag

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/cyberverse/server/internal/character"
)

type SourceType string

type SourceStatus string

const (
	SourceStatusIndexing SourceStatus = "indexing"
	SourceStatusReady    SourceStatus = "ready"
	SourceStatusFailed   SourceStatus = "failed"
)

type Source struct {
	ID             string       `json:"id"`
	Type           SourceType   `json:"type,omitempty"`
	Title          string       `json:"title"`
	Filename       string       `json:"filename"`
	MimeType       string       `json:"mime_type"`
	RelativePath   string       `json:"relative_path,omitempty"`
	StoredPath     string       `json:"stored_path,omitempty"`
	Indexable      bool         `json:"indexable"`
	Status         SourceStatus `json:"status"`
	ChunkCount     int          `json:"chunk_count"`
	Error          string       `json:"error,omitempty"`
	CreatedAt      string       `json:"created_at"`
	UpdatedAt      string       `json:"updated_at"`
	IndexedAt      string       `json:"indexed_at,omitempty"`
	StoredFilename string       `json:"stored_filename,omitempty"`
}

type Store struct {
	mu        sync.Mutex
	charStore *character.Store
}

type FileSaveResult struct {
	Source            *Source
	Path              string
	Created           bool
	PreviousIndexable bool
}

func NewStore(charStore *character.Store) *Store {
	return &Store{charStore: charStore}
}

func nowString() string {
	return time.Now().UTC().Format(time.RFC3339)
}

func (s *Store) characterDir(characterID string) (string, error) {
	if s == nil || s.charStore == nil {
		return "", errors.New("character store is not configured")
	}
	if _, err := s.charStore.Get(characterID); err != nil {
		return "", err
	}
	dir := s.charStore.CharDir(characterID)
	if dir == "" {
		return "", fmt.Errorf("character directory not found: %s", characterID)
	}
	return dir, nil
}

func (s *Store) KnowledgeDir(characterID string) (string, error) {
	dir, err := s.characterDir(characterID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "knowledge"), nil
}

func (s *Store) SourcesDir(characterID string) (string, error) {
	dir, err := s.KnowledgeDir(characterID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "sources"), nil
}

func (s *Store) legacySourceDir(characterID, sourceID string) (string, error) {
	if strings.TrimSpace(sourceID) == "" || sourceID != filepath.Base(sourceID) || strings.Contains(sourceID, "..") {
		return "", fmt.Errorf("invalid source id")
	}
	dir, err := s.SourcesDir(characterID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, sourceID), nil
}

func (s *Store) sourcesFile(characterID string) (string, error) {
	dir, err := s.KnowledgeDir(characterID)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "sources.json"), nil
}

func (s *Store) readSourcesLocked(characterID string) ([]Source, error) {
	path, err := s.sourcesFile(characterID)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []Source{}, nil
		}
		return nil, err
	}
	var sources []Source
	if err := json.Unmarshal(data, &sources); err != nil {
		return nil, err
	}
	if sources == nil {
		sources = []Source{}
	}
	for i := range sources {
		sources[i] = normalizeSource(sources[i])
	}
	return sources, nil
}

func (s *Store) writeSourcesLocked(characterID string, sources []Source) error {
	path, err := s.sourcesFile(characterID)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	sort.Slice(sources, func(i, j int) bool {
		return sources[i].CreatedAt > sources[j].CreatedAt
	})
	data, err := json.MarshalIndent(sources, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func sanitizePathSegment(segment, fallback string) string {
	segment = strings.TrimSpace(segment)
	if segment == "" {
		segment = fallback
	}
	var b strings.Builder
	for _, r := range segment {
		switch {
		case r == 0 || r < 32 || r == 127:
			b.WriteRune('_')
		case strings.ContainsRune(`<>:"|?*`, r):
			b.WriteRune('_')
		default:
			b.WriteRune(r)
		}
	}
	segment = strings.TrimSpace(b.String())
	if strings.Trim(segment, "._- ") == "" {
		segment = fallback
	}
	rs := []rune(segment)
	if len(rs) > 120 {
		segment = string(rs[:120])
	}
	return segment
}

func cleanRelativePath(value string) (string, string, error) {
	value = strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
	if value == "" || strings.HasPrefix(value, "/") || filepath.IsAbs(value) {
		return "", "", fmt.Errorf("invalid relative path")
	}
	rawParts := strings.Split(value, "/")
	parts := make([]string, 0, len(rawParts))
	for _, part := range rawParts {
		part = strings.TrimSpace(part)
		if part == "" || part == "." {
			continue
		}
		if part == ".." {
			return "", "", fmt.Errorf("invalid relative path")
		}
		parts = append(parts, sanitizePathSegment(part, "item"))
	}
	if len(parts) == 0 {
		return "", "", fmt.Errorf("invalid relative path")
	}
	filename := parts[len(parts)-1]
	return strings.Join(parts, "/"), filename, nil
}

func storedPathFor(relativePath string) string {
	return filepath.ToSlash(filepath.Join("sources", filepath.FromSlash(relativePath)))
}

func defaultTitle(title, filename string) string {
	title = strings.TrimSpace(title)
	if title != "" {
		return title
	}
	filename = strings.TrimSpace(filename)
	if filename != "" {
		return strings.TrimSuffix(filename, filepath.Ext(filename))
	}
	return "素材"
}

func supportedExt(filename string) bool {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".txt", ".md", ".json", ".pdf", ".docx":
		return true
	default:
		return false
	}
}

func IndexableFilename(filename string) bool {
	return supportedExt(filename)
}

func normalizeSource(source Source) Source {
	if source.Filename == "" && source.StoredFilename != "" {
		source.Filename = source.StoredFilename
	}
	if source.RelativePath == "" && source.StoredPath != "" {
		stored := filepath.ToSlash(source.StoredPath)
		if strings.HasPrefix(stored, "sources/") {
			source.RelativePath = strings.TrimPrefix(stored, "sources/")
		}
	}
	if source.RelativePath == "" {
		source.RelativePath = source.Filename
	}
	if source.StoredPath == "" && source.ID != "" && source.StoredFilename == "" {
		source.StoredFilename = source.Filename
	}
	source.Indexable = supportedExt(source.Filename)
	return source
}

func mimeTypeFor(filename, provided string) string {
	if provided = strings.TrimSpace(provided); provided != "" {
		return provided
	}
	if typ := mime.TypeByExtension(filepath.Ext(filename)); typ != "" {
		return typ
	}
	return "application/octet-stream"
}

func (s *Store) List(characterID string) ([]Source, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sources, err := s.readSourcesLocked(characterID)
	if err != nil {
		return nil, err
	}
	sort.Slice(sources, func(i, j int) bool {
		return sources[i].CreatedAt > sources[j].CreatedAt
	})
	return sources, nil
}

func (s *Store) Get(characterID, sourceID string) (*Source, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sources, err := s.readSourcesLocked(characterID)
	if err != nil {
		return nil, err
	}
	for _, src := range sources {
		if src.ID == sourceID {
			copied := src
			return &copied, nil
		}
	}
	return nil, fmt.Errorf("knowledge source not found: %s", sourceID)
}

func (s *Store) pathFromStoredPath(characterID, storedPath string) (string, error) {
	storedPath = filepath.ToSlash(strings.TrimSpace(storedPath))
	if storedPath == "" || strings.HasPrefix(storedPath, "/") {
		return "", fmt.Errorf("invalid stored path")
	}
	cleaned := filepath.ToSlash(filepath.Clean(filepath.FromSlash(storedPath)))
	if cleaned == "." || !strings.HasPrefix(cleaned, "sources/") {
		return "", fmt.Errorf("invalid stored path")
	}
	for _, part := range strings.Split(cleaned, "/") {
		if part == ".." {
			return "", fmt.Errorf("invalid stored path")
		}
	}
	knowledgeDir, err := s.KnowledgeDir(characterID)
	if err != nil {
		return "", err
	}
	path := filepath.Join(knowledgeDir, filepath.FromSlash(cleaned))
	absKnowledgeDir, err := filepath.Abs(knowledgeDir)
	if err != nil {
		return "", err
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if absPath != absKnowledgeDir && !strings.HasPrefix(absPath, absKnowledgeDir+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid stored path")
	}
	return path, nil
}

func (s *Store) SourcePath(characterID string, source *Source) (string, error) {
	if source == nil {
		return "", errors.New("source is nil")
	}
	if strings.TrimSpace(source.StoredPath) != "" {
		return s.pathFromStoredPath(characterID, source.StoredPath)
	}
	if strings.TrimSpace(source.RelativePath) != "" {
		relativePath, _, err := cleanRelativePath(source.RelativePath)
		if err == nil {
			if path, pathErr := s.pathFromStoredPath(characterID, storedPathFor(relativePath)); pathErr == nil {
				if stat, statErr := os.Stat(path); statErr == nil && !stat.IsDir() {
					return path, nil
				}
			}
		}
	}

	dir, err := s.legacySourceDir(characterID, source.ID)
	if err != nil {
		return "", err
	}
	filename := source.StoredFilename
	if filename == "" {
		filename = source.Filename
	}
	if filename == "" || filename != filepath.Base(filename) || strings.Contains(filename, "..") {
		return "", fmt.Errorf("invalid stored filename")
	}
	return filepath.Join(dir, filename), nil
}

func (s *Store) SaveFile(characterID, relativePath, mimeType string, reader io.Reader) (*FileSaveResult, error) {
	relativePath, filename, err := cleanRelativePath(relativePath)
	if err != nil {
		return nil, err
	}
	path, err := s.pathFromStoredPath(characterID, storedPathFor(relativePath))
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	dest, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	defer dest.Close()
	if _, err := io.Copy(dest, reader); err != nil {
		return nil, err
	}
	return s.upsertSourceRecord(characterID, relativePath, filename, mimeTypeFor(filename, mimeType), path)
}

func (s *Store) upsertSourceRecord(characterID string, relativePath, filename, mimeType, path string) (*FileSaveResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sources, err := s.readSourcesLocked(characterID)
	if err != nil {
		return nil, err
	}
	now := nowString()
	indexable := supportedExt(filename)
	status := SourceStatusReady
	if indexable {
		status = SourceStatusIndexing
	}
	storedPath := storedPathFor(relativePath)
	for i := range sources {
		if sources[i].RelativePath != relativePath && sources[i].StoredPath != storedPath {
			continue
		}
		previousIndexable := sources[i].Indexable
		sources[i].Title = defaultTitle("", filename)
		sources[i].Filename = filename
		sources[i].MimeType = mimeType
		sources[i].RelativePath = relativePath
		sources[i].StoredPath = storedPath
		sources[i].Indexable = indexable
		sources[i].Status = status
		sources[i].ChunkCount = 0
		sources[i].Error = ""
		sources[i].UpdatedAt = now
		sources[i].IndexedAt = ""
		sources[i].StoredFilename = filename
		if err := s.writeSourcesLocked(characterID, sources); err != nil {
			return nil, err
		}
		copied := sources[i]
		return &FileSaveResult{Source: &copied, Path: path, Created: false, PreviousIndexable: previousIndexable}, nil
	}

	src := Source{
		ID:             uuid.NewString(),
		Title:          defaultTitle("", filename),
		Filename:       filename,
		MimeType:       mimeType,
		RelativePath:   relativePath,
		StoredPath:     storedPath,
		Indexable:      indexable,
		Status:         status,
		ChunkCount:     0,
		CreatedAt:      now,
		UpdatedAt:      now,
		StoredFilename: filename,
	}
	sources = append(sources, src)
	if err := s.writeSourcesLocked(characterID, sources); err != nil {
		return nil, err
	}
	return &FileSaveResult{Source: &src, Path: path, Created: true, PreviousIndexable: false}, nil
}

func (s *Store) MarkIndexing(characterID, sourceID string) (*Source, error) {
	return s.updateSource(characterID, sourceID, func(src *Source) {
		src.Status = SourceStatusIndexing
		src.Error = ""
		src.ChunkCount = 0
		src.IndexedAt = ""
		src.UpdatedAt = nowString()
	})
}

func (s *Store) MarkStoredReady(characterID, sourceID string) (*Source, error) {
	return s.updateSource(characterID, sourceID, func(src *Source) {
		src.Status = SourceStatusReady
		src.Error = ""
		src.ChunkCount = 0
		src.IndexedAt = ""
		src.UpdatedAt = nowString()
	})
}

func (s *Store) MarkReady(characterID, sourceID string, chunkCount int) (*Source, error) {
	return s.updateSource(characterID, sourceID, func(src *Source) {
		src.Status = SourceStatusReady
		src.Error = ""
		src.ChunkCount = chunkCount
		now := nowString()
		src.UpdatedAt = now
		src.IndexedAt = now
	})
}

func (s *Store) MarkFailed(characterID, sourceID string, indexErr error) (*Source, error) {
	msg := ""
	if indexErr != nil {
		msg = indexErr.Error()
	}
	return s.updateSource(characterID, sourceID, func(src *Source) {
		src.Status = SourceStatusFailed
		src.Error = msg
		src.UpdatedAt = nowString()
	})
}

func (s *Store) updateSource(characterID, sourceID string, mutate func(*Source)) (*Source, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	sources, err := s.readSourcesLocked(characterID)
	if err != nil {
		return nil, err
	}
	for i := range sources {
		if sources[i].ID != sourceID {
			continue
		}
		mutate(&sources[i])
		if err := s.writeSourcesLocked(characterID, sources); err != nil {
			return nil, err
		}
		copied := sources[i]
		return &copied, nil
	}
	return nil, fmt.Errorf("knowledge source not found: %s", sourceID)
}

func (s *Store) Delete(characterID, sourceID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	sources, err := s.readSourcesLocked(characterID)
	if err != nil {
		return err
	}
	next := sources[:0]
	found := false
	var removed Source
	for _, src := range sources {
		if src.ID == sourceID {
			found = true
			removed = src
			continue
		}
		next = append(next, src)
	}
	if !found {
		return fmt.Errorf("knowledge source not found: %s", sourceID)
	}
	if err := s.writeSourcesLocked(characterID, next); err != nil {
		return err
	}
	if found {
		if removed.StoredPath != "" {
			if path, err := s.pathFromStoredPath(characterID, removed.StoredPath); err == nil {
				_ = os.Remove(path)
				s.removeEmptySourceDirs(characterID, filepath.Dir(path))
			}
			return nil
		}
		sourceDir, err := s.legacySourceDir(characterID, sourceID)
		if err == nil {
			_ = os.RemoveAll(sourceDir)
		}
	}
	return nil
}

func (s *Store) removeEmptySourceDirs(characterID, startDir string) {
	root, err := s.SourcesDir(characterID)
	if err != nil {
		return
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return
	}
	dir, err := filepath.Abs(startDir)
	if err != nil {
		return
	}
	for dir != root && strings.HasPrefix(dir, root+string(filepath.Separator)) {
		if err := os.Remove(dir); err != nil {
			return
		}
		dir = filepath.Dir(dir)
	}
}
