package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/cyberverse/server/internal/agenttask"
	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/orchestrator"
)

func TestListSessionTasksAllowsClosedSession(t *testing.T) {
	root := t.TempDir()
	taskStore, err := agenttask.OpenStore(filepath.Join(root, "tasks.db"), filepath.Join(root, "artifacts"))
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer taskStore.Close()
	task, err := taskStore.CreateTask(context.Background(), agenttask.CreateTaskInput{
		ID:          "task-closed-session",
		SessionID:   "closed-session",
		UserRequest: "恢复历史任务",
	})
	if err != nil {
		t.Fatalf("CreateTask: %v", err)
	}

	charStore, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	router := NewRouter(
		orchestrator.NewSessionManager(1),
		nil,
		nil,
		nil,
		nil,
		charStore,
		"",
		"",
		agenttask.NewService(taskStore, nil),
	)
	req := httptest.NewRequest("GET", "/api/v1/sessions/closed-session/tasks", nil)
	w := httptest.NewRecorder()
	router.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Tasks []agenttask.Task `json:"tasks"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Tasks) != 1 || resp.Tasks[0].ID != task.ID {
		t.Fatalf("unexpected tasks response: %+v", resp.Tasks)
	}
}
