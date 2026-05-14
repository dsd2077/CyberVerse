package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"strings"
	"time"

	ragstore "github.com/cyberverse/server/internal/rag"
)

type internalKnowledgeSearchRequest struct {
	Query string `json:"query"`
}

type skippedKnowledgeFile struct {
	Filename string `json:"filename"`
	Reason   string `json:"reason"`
}

type uploadKnowledgeFilesResponse struct {
	Sources []ragstore.Source      `json:"sources"`
	Skipped []skippedKnowledgeFile `json:"skipped,omitempty"`
}

type uploadedKnowledgeFile struct {
	Filename string
	MimeType string
	TempPath string
}

func (r *Router) handleListKnowledgeSources(w http.ResponseWriter, req *http.Request) {
	if r.ragStore == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "knowledge store is disabled"})
		return
	}
	sources, err := r.ragStore.List(req.PathValue("id"))
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"sources": sources})
}

func (r *Router) handleUploadKnowledgeFiles(w http.ResponseWriter, req *http.Request) {
	if r.ragStore == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "knowledge store is disabled"})
		return
	}
	reader, err := req.MultipartReader()
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid multipart form: " + err.Error()})
		return
	}

	characterID := req.PathValue("id")
	files, relativePaths, skipped, err := readKnowledgeMultipart(reader)
	defer cleanupUploadedKnowledgeFiles(files)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid multipart form: " + err.Error()})
		return
	}
	if len(files) == 0 {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "file is required"})
		return
	}

	sources := make([]ragstore.Source, 0, len(files))
	for i, upload := range files {
		file, err := os.Open(upload.TempPath)
		if err != nil {
			skipped = append(skipped, skippedKnowledgeFile{Filename: upload.Filename, Reason: err.Error()})
			continue
		}
		relativePath := upload.Filename
		if i < len(relativePaths) && strings.TrimSpace(relativePaths[i]) != "" {
			relativePath = relativePaths[i]
		}
		result, createErr := r.ragStore.SaveFile(characterID, relativePath, upload.MimeType, file)
		closeErr := file.Close()
		if createErr != nil {
			skipped = append(skipped, skippedKnowledgeFile{Filename: upload.Filename, Reason: createErr.Error()})
			continue
		}
		if closeErr != nil {
			skipped = append(skipped, skippedKnowledgeFile{Filename: upload.Filename, Reason: closeErr.Error()})
			continue
		}
		if result.Source.Indexable {
			r.scheduleKnowledgeIndex(characterID, result.Source.ID)
		} else if result.PreviousIndexable && r.orch != nil {
			go func(sourceID string) {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
				if err := r.orch.DeleteKnowledgeSource(ctx, characterID, r.charStore.CharDir(characterID), sourceID); err != nil {
					log.Printf("knowledge delete stale index failed character=%s source=%s: %v", characterID, sourceID, err)
				}
			}(result.Source.ID)
		}
		sources = append(sources, *result.Source)
	}

	resp := uploadKnowledgeFilesResponse{Sources: sources, Skipped: skipped}
	if len(sources) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error":   "no files uploaded",
			"skipped": skipped,
		})
		return
	}
	writeJSON(w, http.StatusCreated, resp)
}

func readKnowledgeMultipart(reader *multipart.Reader) ([]uploadedKnowledgeFile, []string, []skippedKnowledgeFile, error) {
	files := make([]uploadedKnowledgeFile, 0)
	relativePaths := make([]string, 0)
	skipped := make([]skippedKnowledgeFile, 0)

	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return files, relativePaths, skipped, err
		}

		switch part.FormName() {
		case "file", "files":
			filename := part.FileName()
			if strings.TrimSpace(filename) == "" {
				skipped = append(skipped, skippedKnowledgeFile{Filename: "", Reason: "filename is required"})
				_ = part.Close()
				continue
			}
			tmp, err := os.CreateTemp("", "cyberverse-knowledge-upload-*")
			if err != nil {
				_ = part.Close()
				return files, relativePaths, skipped, err
			}
			tempPath := tmp.Name()
			_, copyErr := io.Copy(tmp, part)
			closeErr := tmp.Close()
			_ = part.Close()
			if copyErr != nil {
				_ = os.Remove(tempPath)
				skipped = append(skipped, skippedKnowledgeFile{Filename: filename, Reason: copyErr.Error()})
				continue
			}
			if closeErr != nil {
				_ = os.Remove(tempPath)
				skipped = append(skipped, skippedKnowledgeFile{Filename: filename, Reason: closeErr.Error()})
				continue
			}
			files = append(files, uploadedKnowledgeFile{
				Filename: filename,
				MimeType: part.Header.Get("Content-Type"),
				TempPath: tempPath,
			})
		case "relative_paths":
			data, err := io.ReadAll(io.LimitReader(part, 64*1024))
			_ = part.Close()
			if err != nil {
				return files, relativePaths, skipped, err
			}
			relativePaths = append(relativePaths, strings.TrimSpace(string(data)))
		default:
			_, _ = io.Copy(io.Discard, part)
			_ = part.Close()
		}
	}

	return files, relativePaths, skipped, nil
}

func cleanupUploadedKnowledgeFiles(files []uploadedKnowledgeFile) {
	for _, file := range files {
		if file.TempPath != "" {
			_ = os.Remove(file.TempPath)
		}
	}
}

func (r *Router) handleDeleteKnowledgeSource(w http.ResponseWriter, req *http.Request) {
	if r.ragStore == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "knowledge store is disabled"})
		return
	}
	characterID := req.PathValue("id")
	sourceID := req.PathValue("source_id")
	source, err := r.ragStore.Get(characterID, sourceID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	charDir := r.charStore.CharDir(characterID)
	if err := r.ragStore.Delete(characterID, source.ID); err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	if r.orch != nil {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if err := r.orch.DeleteKnowledgeSource(ctx, characterID, charDir, source.ID); err != nil {
				log.Printf("knowledge delete index failed character=%s source=%s: %v", characterID, source.ID, err)
			}
		}()
	}
	w.WriteHeader(http.StatusNoContent)
}

func (r *Router) handleReindexKnowledgeSource(w http.ResponseWriter, req *http.Request) {
	if r.ragStore == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "knowledge store is disabled"})
		return
	}
	characterID := req.PathValue("id")
	sourceID := req.PathValue("source_id")
	source, err := r.ragStore.MarkIndexing(characterID, sourceID)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	if !source.Indexable {
		source, err = r.ragStore.MarkStoredReady(characterID, sourceID)
		if err != nil {
			writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
			return
		}
		writeJSON(w, http.StatusAccepted, source)
		return
	}
	r.scheduleKnowledgeIndex(characterID, source.ID)
	writeJSON(w, http.StatusAccepted, source)
}

func (r *Router) handleInternalKnowledgeSearch(w http.ResponseWriter, req *http.Request) {
	if !authorizeInternalRequest(w, req) {
		return
	}
	if r.orch == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "inference service is not configured"})
		return
	}
	var body internalKnowledgeSearchRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	if strings.TrimSpace(body.Query) == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "query is required"})
		return
	}
	results, err := r.orch.SearchKnowledge(req.Context(), req.PathValue("id"), body.Query)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

func authorizeInternalRequest(w http.ResponseWriter, req *http.Request) bool {
	expected := strings.TrimSpace(os.Getenv("AGENT_INTERNAL_TOKEN"))
	if expected == "" {
		return true
	}
	got := strings.TrimSpace(req.Header.Get("Authorization"))
	if got != "Bearer "+expected {
		writeJSON(w, http.StatusUnauthorized, ErrorResponse{Error: "invalid internal token"})
		return false
	}
	return true
}

func (r *Router) scheduleKnowledgeIndex(characterID, sourceID string) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := r.indexKnowledgeSource(ctx, characterID, sourceID); err != nil {
			log.Printf("knowledge index failed character=%s source=%s: %v", characterID, sourceID, err)
		}
	}()
}

func (r *Router) indexKnowledgeSource(ctx context.Context, characterID, sourceID string) error {
	if r == nil || r.ragStore == nil {
		return errors.New("knowledge store is disabled")
	}
	source, err := r.ragStore.Get(characterID, sourceID)
	if err != nil {
		return err
	}
	if !source.Indexable {
		_, err := r.ragStore.MarkStoredReady(characterID, sourceID)
		return err
	}
	path, err := r.ragStore.SourcePath(characterID, source)
	if err != nil {
		_, _ = r.ragStore.MarkFailed(characterID, sourceID, err)
		return err
	}
	if r.cfg != nil && !r.cfg.Pipeline.RAG.IsEnabled() {
		err := errors.New("RAG is disabled")
		_, _ = r.ragStore.MarkFailed(characterID, sourceID, err)
		return err
	}
	if r.orch == nil {
		err := errors.New("inference service is not configured")
		_, _ = r.ragStore.MarkFailed(characterID, sourceID, err)
		return err
	}
	chunkCount, err := r.orch.IndexKnowledgeSource(ctx, characterID, r.charStore.CharDir(characterID), source, path)
	if err != nil {
		_, _ = r.ragStore.MarkFailed(characterID, sourceID, err)
		return err
	}
	_, err = r.ragStore.MarkReady(characterID, sourceID, chunkCount)
	return err
}
