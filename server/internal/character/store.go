package character

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ImageInfo describes one image file inside a character's images/ directory.
type ImageInfo struct {
	Filename string `json:"filename"`
	OrigName string `json:"orig_name"`
	AddedAt  string `json:"added_at"`
}

type Character struct {
	ID             string      `json:"id"`
	Name           string      `json:"name"`
	Description    string      `json:"description"`
	AvatarImage    string      `json:"avatar_image"`
	UseFaceCrop    bool        `json:"use_face_crop"`
	VoiceProvider  string      `json:"voice_provider"`
	VoiceType      string      `json:"voice_type"`
	SpeakingStyle  string      `json:"speaking_style"`
	Personality    string      `json:"personality"`
	WelcomeMessage string      `json:"welcome_message"`
	SystemPrompt   string      `json:"system_prompt"`
	Tags           []string    `json:"tags"`
	Images         []ImageInfo `json:"images"`
	ActiveImage    string      `json:"active_image"`
	ImageMode      string      `json:"image_mode"`
	CreatedAt      string      `json:"created_at"`
	UpdatedAt      string      `json:"updated_at"`
}

const DefaultIdleVideoProfile = "breathing10s_v1"

type Store struct {
	mu      sync.RWMutex
	baseDir string
	chars   map[string]*Character
	// dirNames caches id → directory name (e.g. "程欣_8981e0a1")
	dirNames map[string]string
}

// NewStore creates a store backed by per-character directories under baseDir.
func NewStore(baseDir string) (*Store, error) {
	s := &Store{
		baseDir:  baseDir,
		chars:    make(map[string]*Character),
		dirNames: make(map[string]string),
	}
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return nil, fmt.Errorf("create characters dir: %w", err)
	}
	if err := s.load(); err != nil {
		return nil, fmt.Errorf("load characters: %w", err)
	}
	return s, nil
}

func (s *Store) BaseDir() string {
	return s.baseDir
}

func (s *Store) List() []*Character {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]*Character, 0, len(s.chars))
	for _, c := range s.chars {
		result = append(result, c)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt < result[j].CreatedAt
	})
	return result
}

func (s *Store) Get(id string) (*Character, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c, ok := s.chars[id]
	if !ok {
		return nil, fmt.Errorf("character not found: %s", id)
	}
	return c, nil
}

func (s *Store) Create(c *Character) (*Character, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	c.ID = uuid.New().String()
	now := time.Now().UTC().Format(time.RFC3339)
	c.CreatedAt = now
	c.UpdatedAt = now
	if c.Tags == nil {
		c.Tags = []string{}
	}
	if c.Images == nil {
		c.Images = []ImageInfo{}
	}

	dirName := charDirName(c.Name, c.ID)
	charDir := filepath.Join(s.baseDir, dirName)

	// Create directory structure
	for _, sub := range []string{"", "images", "sessions"} {
		if err := os.MkdirAll(filepath.Join(charDir, sub), 0755); err != nil {
			return nil, fmt.Errorf("create character dir: %w", err)
		}
	}

	s.chars[c.ID] = c
	s.dirNames[c.ID] = dirName

	if err := s.saveOne(c); err != nil {
		// Cleanup on failure
		os.RemoveAll(charDir)
		delete(s.chars, c.ID)
		delete(s.dirNames, c.ID)
		return nil, err
	}
	return c, nil
}

func (s *Store) Update(id string, c *Character) (*Character, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	existing, ok := s.chars[id]
	if !ok {
		return nil, fmt.Errorf("character not found: %s", id)
	}

	oldDirName := s.dirNames[id]

	c.ID = id
	c.CreatedAt = existing.CreatedAt
	c.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	if c.Tags == nil {
		c.Tags = []string{}
	}
	// Preserve images/active_image from existing if not provided
	if c.Images == nil {
		c.Images = existing.Images
	}
	if c.ActiveImage == "" {
		c.ActiveImage = existing.ActiveImage
	}
	if c.ImageMode == "" {
		c.ImageMode = existing.ImageMode
	}
	// Preserve avatar_image if caller sent empty (e.g. frontend strips blob: URLs)
	if c.AvatarImage == "" && existing.AvatarImage != "" {
		c.AvatarImage = existing.AvatarImage
	}

	newDirName := charDirName(c.Name, c.ID)

	// Rename directory if name changed
	if oldDirName != newDirName {
		oldPath := filepath.Join(s.baseDir, oldDirName)
		newPath := filepath.Join(s.baseDir, newDirName)
		if err := os.Rename(oldPath, newPath); err != nil {
			return nil, fmt.Errorf("rename character dir: %w", err)
		}
		s.dirNames[id] = newDirName
	}

	s.chars[id] = c
	if err := s.saveOne(c); err != nil {
		return nil, err
	}
	return c, nil
}

func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.chars[id]; !ok {
		return fmt.Errorf("character not found: %s", id)
	}

	dirName := s.dirNames[id]
	if dirName != "" {
		os.RemoveAll(filepath.Join(s.baseDir, dirName))
	}

	delete(s.chars, id)
	delete(s.dirNames, id)
	return nil
}

// CharDir returns the full path to a character's directory.
func (s *Store) CharDir(id string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	dirName, ok := s.dirNames[id]
	if !ok {
		return ""
	}
	return filepath.Join(s.baseDir, dirName)
}

// ImagesDir returns the full path to a character's images directory.
func (s *Store) ImagesDir(id string) string {
	d := s.CharDir(id)
	if d == "" {
		return ""
	}
	return filepath.Join(d, "images")
}

// SessionsDir returns the full path to a character's sessions directory.
func (s *Store) SessionsDir(id string) string {
	d := s.CharDir(id)
	if d == "" {
		return ""
	}
	return filepath.Join(d, "sessions")
}

// IdleVideosDir returns the full path to a character's idle video cache directory.
func (s *Store) IdleVideosDir(id string) string {
	d := s.CharDir(id)
	if d == "" {
		return ""
	}
	return filepath.Join(d, "idle_videos")
}

// IdleVideosForImageDir returns the per-image subdirectory under idle_videos/.
// e.g. {charDir}/idle_videos/img_003/
func (s *Store) IdleVideosForImageDir(id, imageFilename string) string {
	dir := s.IdleVideosDir(id)
	if dir == "" {
		return ""
	}
	base := strings.TrimSuffix(filepath.Base(imageFilename), filepath.Ext(imageFilename))
	if base == "" {
		base = "avatar"
	}
	return filepath.Join(dir, base)
}

func idleVideoBaseName(imageFilename string) string {
	base := strings.TrimSuffix(filepath.Base(imageFilename), filepath.Ext(imageFilename))
	if base == "" {
		base = "avatar"
	}
	return base
}

func idleVideoProfileName(profile string) string {
	if profile == "" {
		return DefaultIdleVideoProfile
	}
	return profile
}

func idleVideoSizeDirName(width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	return fmt.Sprintf("%dx%d", width, height)
}

// IdleVideosForSizeDir returns the per-image, per-resolution cache directory.
// e.g. {charDir}/idle_videos/img_003/320x480/
func (s *Store) IdleVideosForSizeDir(id, imageFilename string, width, height int) string {
	dir := s.IdleVideosForImageDir(id, imageFilename)
	sizeDir := idleVideoSizeDirName(width, height)
	if dir == "" || sizeDir == "" {
		return ""
	}
	return filepath.Join(dir, sizeDir)
}

// IdleVideoFilename returns a stable MP4 filename for one source image + profile.
func (s *Store) IdleVideoFilename(imageFilename, profile string) string {
	return fmt.Sprintf("%s__%s.mp4", idleVideoBaseName(imageFilename), idleVideoProfileName(profile))
}

// IdleVideoPath returns the absolute path for a cached idle video inside the
// per-image, per-resolution subdirectory.
func (s *Store) IdleVideoPath(id, imageFilename, profile string, width, height int) string {
	dir := s.IdleVideosForSizeDir(id, imageFilename, width, height)
	if dir == "" {
		return ""
	}
	return filepath.Join(dir, s.IdleVideoFilename(imageFilename, profile))
}

// ListIdleVideos returns all .mp4 filenames in the target resolution directory, sorted.
func (s *Store) ListIdleVideos(id, imageFilename string, width, height int) ([]string, error) {
	dir := s.IdleVideosForSizeDir(id, imageFilename, width, height)
	if dir == "" {
		return nil, fmt.Errorf("idle video dir unavailable for character %s", id)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".mp4") {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)
	return files, nil
}

// NextImageFilename scans the images/ directory and returns the next sequential filename
// (without extension), e.g. "img_003".
func (s *Store) NextImageFilename(id string) string {
	imgDir := s.ImagesDir(id)
	if imgDir == "" {
		return "img_001"
	}

	entries, err := os.ReadDir(imgDir)
	if err != nil {
		return "img_001"
	}

	maxNum := 0
	re := regexp.MustCompile(`^img_(\d+)`)
	for _, e := range entries {
		if m := re.FindStringSubmatch(e.Name()); m != nil {
			if n, err := strconv.Atoi(m[1]); err == nil && n > maxNum {
				maxNum = n
			}
		}
	}
	return fmt.Sprintf("img_%03d", maxNum+1)
}

// SaveConversation persists a session's data (messages + metadata) into the character's sessions dir.
func (s *Store) SaveConversation(characterID, sessionID string, startedAt, endedAt time.Time, messages []map[string]any) error {
	sessDir := s.SessionsDir(characterID)
	if sessDir == "" {
		return fmt.Errorf("character not found: %s", characterID)
	}

	dirName := startedAt.Format("20060102-150405") + "_" + shortID(sessionID)
	fullDir := filepath.Join(sessDir, dirName)
	os.MkdirAll(fullDir, 0755)

	record := map[string]any{
		"session_id":   sessionID,
		"character_id": characterID,
		"started_at":   startedAt.UTC().Format(time.RFC3339),
		"ended_at":     endedAt.UTC().Format(time.RFC3339),
		"messages":     messages,
	}

	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(fullDir, "session.json"), data, 0644)
}

// LoadRecentMessages loads recent conversation messages for a character, paginated by cursor.
// before: cursor (session directory name) — empty string means start from newest.
// limit: max number of messages to return.
// Returns: messages (chronological order), next cursor, hasMore.
func (s *Store) LoadRecentMessages(characterID string, before string, limit int) ([]map[string]any, string, bool, error) {
	sessDir := s.SessionsDir(characterID)
	if sessDir == "" {
		return nil, "", false, fmt.Errorf("character not found: %s", characterID)
	}

	entries, err := os.ReadDir(sessDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, "", false, nil
		}
		return nil, "", false, err
	}

	// Sort descending by name (timestamp prefix ensures chronological order)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() > entries[j].Name()
	})

	// Apply cursor: skip entries >= before
	if before != "" {
		idx := 0
		for idx < len(entries) && entries[idx].Name() >= before {
			idx++
		}
		entries = entries[idx:]
	}

	var allMessages []map[string]any
	var nextCursor string
	collected := 0

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		sessionFile := filepath.Join(sessDir, entry.Name(), "session.json")
		data, err := os.ReadFile(sessionFile)
		if err != nil {
			continue // skip sessions without session.json
		}

		var record struct {
			SessionID string           `json:"session_id"`
			StartedAt string           `json:"started_at"`
			Messages  []map[string]any `json:"messages"`
		}
		if err := json.Unmarshal(data, &record); err != nil {
			continue
		}
		if strings.TrimSpace(record.SessionID) == "" {
			record.SessionID = entry.Name()
		}
		if len(record.Messages) == 0 {
			continue
		}

		// Annotate messages with session metadata
		for _, msg := range record.Messages {
			msg["session_id"] = record.SessionID
			if msg["timestamp"] == nil {
				msg["timestamp"] = record.StartedAt
			} else if ts, ok := msg["timestamp"].(string); ok && strings.TrimSpace(ts) == "" {
				msg["timestamp"] = record.StartedAt
			}
		}

		// These messages are in chronological order within the session,
		// but we're iterating sessions newest-first, so prepend
		allMessages = append(record.Messages, allMessages...)
		collected += len(record.Messages)
		nextCursor = entry.Name()

		if collected >= limit {
			break
		}
	}

	// Trim to limit (keep the most recent messages)
	if len(allMessages) > limit {
		allMessages = allMessages[len(allMessages)-limit:]
	}

	// Check if there are more sessions after the cursor
	hasMore := false
	if nextCursor != "" {
		for _, entry := range entries {
			if entry.IsDir() && entry.Name() < nextCursor {
				// Check if this directory has a session.json
				sessionFile := filepath.Join(sessDir, entry.Name(), "session.json")
				if _, err := os.Stat(sessionFile); err == nil {
					hasMore = true
					break
				}
			}
		}
	}

	return allMessages, nextCursor, hasMore, nil
}

// SessionRecordingDir returns the full path for a session's recording directory,
// creating it if needed. Format: {charDir}/sessions/{timestamp}_{sessionID8}/
// Uses createdAt so that recordings land in the same directory as SaveConversation.
func (s *Store) SessionRecordingDir(characterID, sessionID string, createdAt time.Time) string {
	sessDir := s.SessionsDir(characterID)
	if sessDir == "" {
		return ""
	}
	dirName := createdAt.Format("20060102-150405") + "_" + shortID(sessionID)
	fullDir := filepath.Join(sessDir, dirName)
	// Resolve to absolute path so that recording functions (BeginTurn/SaveRawAudio)
	// don't prepend cfg.OutputDir when they see a relative path.
	if abs, err := filepath.Abs(fullDir); err == nil {
		fullDir = abs
	}
	os.MkdirAll(fullDir, 0755)
	return fullDir
}

// ── internal helpers ──

func (s *Store) load() error {
	entries, err := os.ReadDir(s.baseDir)
	if err != nil {
		return err
	}

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		jsonPath := filepath.Join(s.baseDir, e.Name(), "character.json")
		data, err := os.ReadFile(jsonPath)
		if err != nil {
			log.Printf("skip character dir %s: %v", e.Name(), err)
			continue
		}
		var c Character
		if err := json.Unmarshal(data, &c); err != nil {
			log.Printf("skip character dir %s: bad JSON: %v", e.Name(), err)
			continue
		}
		if c.ID == "" {
			log.Printf("skip character dir %s: no id", e.Name())
			continue
		}
		if c.Tags == nil {
			c.Tags = []string{}
		}
		if c.Images == nil {
			c.Images = []ImageInfo{}
		}
		if c.ImageMode == "" {
			c.ImageMode = "fixed"
		}
		s.chars[c.ID] = &c
		s.dirNames[c.ID] = e.Name()
	}
	return nil
}

func (s *Store) saveOne(c *Character) error {
	dirName, ok := s.dirNames[c.ID]
	if !ok {
		return fmt.Errorf("no directory mapping for character %s", c.ID)
	}
	jsonPath := filepath.Join(s.baseDir, dirName, "character.json")
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(jsonPath, data, 0644)
}

// sanitizeName replaces filesystem-unsafe characters and truncates to 50 chars.
func sanitizeName(name string) string {
	unsafe := regexp.MustCompile(`[/\\:*?"<>|]`)
	s := unsafe.ReplaceAllString(name, "_")
	s = strings.TrimSpace(s)
	// Truncate by rune to avoid breaking multi-byte chars
	runes := []rune(s)
	if len(runes) > 50 {
		runes = runes[:50]
	}
	s = string(runes)
	if s == "" {
		s = "unnamed"
	}
	return s
}

// charDirName returns the directory name for a character: "{sanitizedName}_{id[:8]}"
func charDirName(name, id string) string {
	return sanitizeName(name) + "_" + shortID(id)
}

func shortID(id string) string {
	clean := strings.ReplaceAll(id, "-", "")
	if len(clean) > 8 {
		clean = clean[:8]
	}
	return clean
}

// ListImages returns images sorted by filename for a character.
func (s *Store) ListImages(id string) ([]ImageInfo, error) {
	s.mu.RLock()
	c, ok := s.chars[id]
	s.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("character not found: %s", id)
	}
	imgs := make([]ImageInfo, len(c.Images))
	copy(imgs, c.Images)
	sort.Slice(imgs, func(i, j int) bool {
		return imgs[i].Filename < imgs[j].Filename
	})
	return imgs, nil
}

// AddImage adds an image entry to the character and persists.
func (s *Store) AddImage(id string, info ImageInfo) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	c, ok := s.chars[id]
	if !ok {
		return fmt.Errorf("character not found: %s", id)
	}

	c.Images = append(c.Images, info)

	// Auto-activate first image
	if c.ActiveImage == "" {
		c.ActiveImage = info.Filename
		c.AvatarImage = fmt.Sprintf("/api/v1/characters/%s/images/%s", id, info.Filename)
	}

	c.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return s.saveOne(c)
}

// RemoveImage removes an image entry. If it was active, switches to the first remaining.
func (s *Store) RemoveImage(id, filename string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	c, ok := s.chars[id]
	if !ok {
		return fmt.Errorf("character not found: %s", id)
	}

	found := false
	newImages := make([]ImageInfo, 0, len(c.Images))
	for _, img := range c.Images {
		if img.Filename == filename {
			found = true
			continue
		}
		newImages = append(newImages, img)
	}
	if !found {
		return fmt.Errorf("image not found: %s", filename)
	}

	c.Images = newImages

	if c.ActiveImage == filename {
		if len(c.Images) > 0 {
			c.ActiveImage = c.Images[0].Filename
			c.AvatarImage = fmt.Sprintf("/api/v1/characters/%s/images/%s", id, c.ActiveImage)
		} else {
			c.ActiveImage = ""
			c.AvatarImage = ""
		}
	}

	c.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return s.saveOne(c)
}

// ActivateImage sets a specific image as the active avatar.
func (s *Store) ActivateImage(id, filename string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	c, ok := s.chars[id]
	if !ok {
		return fmt.Errorf("character not found: %s", id)
	}

	found := false
	for _, img := range c.Images {
		if img.Filename == filename {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("image not found: %s", filename)
	}

	c.ActiveImage = filename
	c.AvatarImage = fmt.Sprintf("/api/v1/characters/%s/images/%s", id, filename)
	c.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return s.saveOne(c)
}

// RandomizeImage picks a random image and sets it as the active avatar.
func (s *Store) RandomizeImage(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	c, ok := s.chars[id]
	if !ok {
		return fmt.Errorf("character not found: %s", id)
	}
	if len(c.Images) == 0 {
		return nil
	}

	idx := rand.Intn(len(c.Images))
	c.ActiveImage = c.Images[idx].Filename
	c.AvatarImage = fmt.Sprintf("/api/v1/characters/%s/images/%s", id, c.ActiveImage)
	c.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
	return s.saveOne(c)
}
