package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/cyberverse/server/internal/agenttask"
)

type internalTaskEventRequest struct {
	EventType string           `json:"event_type"`
	Status    agenttask.Status `json:"status"`
	Message   string           `json:"message"`
	Progress  int              `json:"progress"`
	Payload   json.RawMessage  `json:"payload"`
}

type internalTaskArtifactRequest struct {
	Type     string          `json:"type"`
	Title    string          `json:"title"`
	MimeType string          `json:"mime_type"`
	Content  string          `json:"content"`
	Metadata json.RawMessage `json:"metadata"`
}

func (r *Router) handleListSessionTasks(w http.ResponseWriter, req *http.Request) {
	if r.taskSvc == nil || r.taskSvc.Store() == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "agent task service is disabled"})
		return
	}
	sessionID := req.PathValue("id")
	if strings.TrimSpace(sessionID) == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "session id is required"})
		return
	}
	limit := parsePositiveInt(req.URL.Query().Get("limit"), 50, 200)
	tasks, err := r.taskSvc.Store().ListSessionTasks(req.Context(), sessionID, limit)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"tasks": tasks})
}

func (r *Router) handleGetTask(w http.ResponseWriter, req *http.Request) {
	if r.taskSvc == nil || r.taskSvc.Store() == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "agent task service is disabled"})
		return
	}
	task, err := r.taskSvc.Store().GetTask(req.Context(), req.PathValue("task_id"))
	if err != nil {
		writeTaskError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, task)
}

func (r *Router) handleListTaskEvents(w http.ResponseWriter, req *http.Request) {
	if r.taskSvc == nil || r.taskSvc.Store() == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "agent task service is disabled"})
		return
	}
	afterSeq, _ := strconv.ParseInt(req.URL.Query().Get("after_seq"), 10, 64)
	limit := parsePositiveInt(req.URL.Query().Get("limit"), 200, 500)
	events, err := r.taskSvc.Store().ListEventsAfter(req.Context(), req.PathValue("task_id"), afterSeq, limit)
	if err != nil {
		writeTaskError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"events": events})
}

func (r *Router) handleGetTaskArtifact(w http.ResponseWriter, req *http.Request) {
	if r.taskSvc == nil || r.taskSvc.Store() == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "agent task service is disabled"})
		return
	}
	artifact, content, err := r.taskSvc.Store().GetArtifact(req.Context(), req.PathValue("task_id"), req.PathValue("artifact_id"))
	if err != nil {
		writeTaskError(w, err)
		return
	}
	w.Header().Set("Content-Type", artifact.MimeType)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(content)
}

func (r *Router) handleInternalTaskEvent(w http.ResponseWriter, req *http.Request) {
	if !r.authorizeInternalTaskRequest(w, req) {
		return
	}
	var body internalTaskEventRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	event, task, err := r.taskSvc.AppendEvent(req.Context(), req.PathValue("task_id"), agenttask.AppendEventInput{
		EventType: body.EventType,
		Status:    body.Status,
		Message:   body.Message,
		Progress:  body.Progress,
		Payload:   body.Payload,
	})
	if err != nil {
		writeTaskError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"event": event, "task": task})
}

func (r *Router) handleInternalTaskArtifact(w http.ResponseWriter, req *http.Request) {
	if !r.authorizeInternalTaskRequest(w, req) {
		return
	}
	var body internalTaskArtifactRequest
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	artifact, err := r.taskSvc.CreateArtifact(req.Context(), req.PathValue("task_id"), agenttask.CreateArtifactInput{
		Type:     body.Type,
		Title:    body.Title,
		MimeType: body.MimeType,
		Content:  body.Content,
		Metadata: body.Metadata,
	})
	if err != nil {
		writeTaskError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, artifact)
}

func (r *Router) authorizeInternalTaskRequest(w http.ResponseWriter, req *http.Request) bool {
	if r.taskSvc == nil || r.taskSvc.Store() == nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "agent task service is disabled"})
		return false
	}
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

func parsePositiveInt(raw string, fallback, max int) int {
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return fallback
	}
	if max > 0 && v > max {
		return max
	}
	return v
}

func writeTaskError(w http.ResponseWriter, err error) {
	if errors.Is(err, agenttask.ErrNotFound) {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	if errors.Is(err, agenttask.ErrTerminal) {
		writeJSON(w, http.StatusConflict, ErrorResponse{Error: err.Error()})
		return
	}
	if errors.Is(err, context.Canceled) {
		writeJSON(w, http.StatusRequestTimeout, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
}
