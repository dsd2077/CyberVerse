package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/cyberverse/server/internal/character"
)

type characterResponse struct {
	*character.Character
	IdleVideoURL  string   `json:"idle_video_url,omitempty"`
	IdleVideoURLs []string `json:"idle_video_urls,omitempty"`
}

type testCharacterVoiceRequest struct {
	VoiceProvider string `json:"voice_provider"`
	VoiceType     string `json:"voice_type"`
}

type idleVideoTarget struct {
	width  int
	height int
}

func (t idleVideoTarget) valid() bool {
	return t.width > 0 && t.height > 0
}

func characterIdleVideoSizeDir(target idleVideoTarget) string {
	if !target.valid() {
		return ""
	}
	return fmt.Sprintf("%dx%d", target.width, target.height)
}

func (r *Router) currentIdleVideoTarget(ctx context.Context) idleVideoTarget {
	if r == nil || r.orch == nil {
		return idleVideoTarget{}
	}

	info, err := r.orch.AvatarInfo(ctx)
	if err != nil {
		return idleVideoTarget{}
	}

	target := idleVideoTarget{
		width:  int(info.GetOutputWidth()),
		height: int(info.GetOutputHeight()),
	}
	if !target.valid() {
		return idleVideoTarget{}
	}
	return target
}

// idleVideoURLs returns idle video URLs for the current output resolution.
func (r *Router) idleVideoURLs(characterID, imageFilename string, target idleVideoTarget) []string {
	if r.charStore == nil || characterID == "" || imageFilename == "" {
		return nil
	}
	if !target.valid() {
		return nil
	}
	imgBase := strings.TrimSuffix(imageFilename, filepath.Ext(imageFilename))
	sizeDir := characterIdleVideoSizeDir(target)
	files, err := r.charStore.ListIdleVideos(characterID, imageFilename, target.width, target.height)
	if err != nil || len(files) == 0 {
		return nil
	}

	urls := make([]string, 0, len(files))
	for _, filename := range files {
		urls = append(urls, fmt.Sprintf("/api/v1/characters/%s/idle-videos/%s/%s/%s", characterID, imgBase, sizeDir, filename))
	}
	return urls
}

// idleVideoURL returns the first idle video URL (backward compatibility).
func (r *Router) idleVideoURL(characterID, imageFilename string, target idleVideoTarget) string {
	urls := r.idleVideoURLs(characterID, imageFilename, target)
	if len(urls) == 0 {
		return ""
	}
	return urls[0]
}

func (r *Router) buildCharacterResponse(c *character.Character, target idleVideoTarget) characterResponse {
	if c == nil {
		return characterResponse{}
	}
	urls := r.idleVideoURLs(c.ID, c.ActiveImage, target)
	firstURL := ""
	if len(urls) > 0 {
		firstURL = urls[0]
	}
	return characterResponse{
		Character:     c,
		IdleVideoURL:  firstURL,
		IdleVideoURLs: urls,
	}
}

func (r *Router) handleListCharacters(w http.ResponseWriter, req *http.Request) {
	chars := r.charStore.List()
	target := r.currentIdleVideoTarget(req.Context())
	result := make([]characterResponse, 0, len(chars))
	for _, c := range chars {
		result = append(result, r.buildCharacterResponse(c, target))
	}
	writeJSON(w, http.StatusOK, result)
}

func (r *Router) handleGetCharacter(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	c, err := r.charStore.Get(id)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, r.buildCharacterResponse(c, r.currentIdleVideoTarget(req.Context())))
}

func (r *Router) handleCreateCharacter(w http.ResponseWriter, req *http.Request) {
	var c character.Character
	if err := json.NewDecoder(req.Body).Decode(&c); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	if c.Name == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "name is required"})
		return
	}

	created, err := r.charStore.Create(&c)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, r.buildCharacterResponse(created, r.currentIdleVideoTarget(req.Context())))
}

func (r *Router) handleUpdateCharacter(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	var c character.Character
	if err := json.NewDecoder(req.Body).Decode(&c); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}

	updated, err := r.charStore.Update(id, &c)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, r.buildCharacterResponse(updated, r.currentIdleVideoTarget(req.Context())))
}

func (r *Router) handleDeleteCharacter(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if err := r.charStore.Delete(id); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (r *Router) handleTestCharacterVoice(w http.ResponseWriter, req *http.Request) {
	var body testCharacterVoiceRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}

	provider := strings.ToLower(strings.TrimSpace(body.VoiceProvider))
	voiceType := strings.TrimSpace(body.VoiceType)

	if provider == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "voice_provider is required"})
		return
	}
	if voiceType == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "voice_type is required"})
		return
	}
	if provider != "doubao" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "unsupported voice_provider: " + provider})
		return
	}
	if r.orch == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: errInferenceUnavailable.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(req.Context(), 5*time.Second)
	defer cancel()

	providerError, err := r.orch.CheckVoice(ctx, voiceType)
	if providerError != "" {
		writeJSON(w, http.StatusBadGateway, ErrorResponse{Error: providerError})
		return
	}
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "voice check timed out"})
			return
		}
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleUploadAvatar uploads an image to the character's images/ directory.
// Kept at POST /api/v1/characters/{id}/avatar for frontend compatibility.
func (r *Router) handleUploadAvatar(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	if _, err := r.charStore.Get(id); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	if err := req.ParseMultipartForm(10 << 20); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "file too large"})
		return
	}

	file, handler, err := req.FormFile("avatar")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "avatar file required"})
		return
	}
	defer file.Close()

	ext := filepath.Ext(handler.Filename)
	if ext == "" {
		ext = ".png"
	}

	imgDir := r.charStore.ImagesDir(id)
	if imgDir == "" {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "character directory not found"})
		return
	}
	if err := os.MkdirAll(imgDir, 0755); err != nil {
		log.Printf("Failed to create images dir: %v", err)
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "server error"})
		return
	}

	baseName := r.charStore.NextImageFilename(id)
	filename := baseName + ext
	destPath := filepath.Join(imgDir, filename)

	dest, err := os.Create(destPath)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "failed to save file"})
		return
	}
	defer dest.Close()

	if _, err := io.Copy(dest, file); err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "failed to write file"})
		return
	}

	// Add image to character's image list
	info := character.ImageInfo{
		Filename: filename,
		OrigName: handler.Filename,
		AddedAt:  fmt.Sprintf("%d", handler.Size),
	}
	if err := r.charStore.AddImage(id, info); err != nil {
		log.Printf("Failed to add image record: %v", err)
	}

	c, _ := r.charStore.Get(id)
	writeJSON(w, http.StatusOK, map[string]string{"path": c.AvatarImage})
}

// handleListImages returns all images for a character.
func (r *Router) handleListImages(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	imgs, err := r.charStore.ListImages(id)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	// Add URL field for each image
	type imageResp struct {
		character.ImageInfo
		URL string `json:"url"`
	}
	result := make([]imageResp, len(imgs))
	for i, img := range imgs {
		result[i] = imageResp{
			ImageInfo: img,
			URL:       fmt.Sprintf("/api/v1/characters/%s/images/%s", id, img.Filename),
		}
	}
	writeJSON(w, http.StatusOK, result)
}

// handleGetCharacterImage serves an image file from the character's images/ directory.
func (r *Router) handleGetCharacterImage(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	filename := req.PathValue("filename")

	if filename == "" || filename != filepath.Base(filename) || strings.Contains(filename, "..") {
		http.NotFound(w, req)
		return
	}

	imgDir := r.charStore.ImagesDir(id)
	if imgDir == "" {
		http.NotFound(w, req)
		return
	}

	imgPath := filepath.Join(imgDir, filename)
	if _, err := os.Stat(imgPath); err != nil {
		http.NotFound(w, req)
		return
	}

	http.ServeFile(w, req, imgPath)
}

// handleGetIdleVideo serves a cached idle MP4 from the character's idle_videos/{imgbase}/ directory.
func (r *Router) handleGetIdleVideo(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	imgbase := req.PathValue("imgbase")
	variant := req.PathValue("variant")
	filename := req.PathValue("filename")

	// Validate path components to prevent traversal
	parts := []string{imgbase, filename}
	if variant != "" {
		parts = append(parts, variant)
	}
	for _, part := range parts {
		if part == "" || part != filepath.Base(part) || strings.Contains(part, "..") {
			http.NotFound(w, req)
			return
		}
	}

	videoDir := r.charStore.IdleVideosDir(id)
	if videoDir == "" {
		http.NotFound(w, req)
		return
	}

	videoPath := filepath.Join(videoDir, imgbase)
	if variant != "" {
		videoPath = filepath.Join(videoPath, variant)
	}
	videoPath = filepath.Join(videoPath, filename)
	if _, err := os.Stat(videoPath); err != nil {
		http.NotFound(w, req)
		return
	}

	http.ServeFile(w, req, videoPath)
}

// handleDeleteImage removes an image from a character.
func (r *Router) handleDeleteImage(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	filename := req.PathValue("filename")

	if filename == "" || filename != filepath.Base(filename) || strings.Contains(filename, "..") {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid filename"})
		return
	}

	// Delete file from disk
	imgDir := r.charStore.ImagesDir(id)
	if imgDir == "" {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "character not found"})
		return
	}
	imgPath := filepath.Join(imgDir, filename)
	os.Remove(imgPath)

	// Remove from character record
	if err := r.charStore.RemoveImage(id, filename); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// handleActivateImage sets a specific image as the character's active avatar.
func (r *Router) handleActivateImage(w http.ResponseWriter, req *http.Request) {
	id := req.PathValue("id")
	filename := req.PathValue("filename")

	if err := r.charStore.ActivateImage(id, filename); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}

	c, _ := r.charStore.Get(id)
	target := r.currentIdleVideoTarget(req.Context())
	writeJSON(w, http.StatusOK, map[string]any{
		"active_image":    c.ActiveImage,
		"avatar_image":    c.AvatarImage,
		"idle_video_url":  r.idleVideoURL(c.ID, c.ActiveImage, target),
		"idle_video_urls": r.idleVideoURLs(c.ID, c.ActiveImage, target),
	})
}

// handleGetAvatar serves avatar files (backward compatibility for old /api/v1/avatars/{filename} URLs).
func (r *Router) handleGetAvatar(w http.ResponseWriter, req *http.Request) {
	filename := req.PathValue("filename")
	if filename == "" || filename != filepath.Base(filename) || strings.Contains(filename, "..") {
		http.NotFound(w, req)
		return
	}

	// Legacy path: look in old data/avatars/ directory
	avatarPath := filepath.Join(filepath.Dir(r.charStore.BaseDir()), "avatars", filename)
	if _, err := os.Stat(avatarPath); err != nil {
		http.NotFound(w, req)
		return
	}

	http.ServeFile(w, req, avatarPath)
}
