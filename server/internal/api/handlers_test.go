package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/orchestrator"
	pb "github.com/cyberverse/server/internal/pb"
	"github.com/cyberverse/server/internal/ws"
)

func newTestCharStore(t *testing.T) *character.Store {
	t.Helper()
	dir, err := os.MkdirTemp("", "chartest-*")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	store, err := character.NewStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func newTestRouter() *Router {
	return newTestRouterWithInference(&fakeInferenceService{
		avatarInfo: &pb.AvatarInfo{ModelName: "avatar.flash_head", OutputFps: 25, OutputWidth: 512, OutputHeight: 512},
	})
}

func newTestRouterWithMgr(mgr *orchestrator.SessionManager) *Router {
	return newTestRouterWithMgrAndInference(mgr, &fakeInferenceService{
		avatarInfo: &pb.AvatarInfo{ModelName: "avatar.flash_head", OutputFps: 25, OutputWidth: 512, OutputHeight: 512},
	})
}

func newTestRouterWithInference(inf *fakeInferenceService) *Router {
	return newTestRouterWithMgrAndInference(orchestrator.NewSessionManager(4), inf)
}

func newTestRouterWithMgrAndInference(mgr *orchestrator.SessionManager, inf *fakeInferenceService) *Router {
	hub := ws.NewHub()
	dir, _ := os.MkdirTemp("", "chartest-*")
	cs, _ := character.NewStore(dir)
	if inf.avatarInfo == nil {
		inf.avatarInfo = &pb.AvatarInfo{ModelName: "avatar.flash_head", OutputFps: 25, OutputWidth: 512, OutputHeight: 512}
	}
	orch := orchestrator.New(inf, hub, mgr, nil, cs)
	return NewRouter(mgr, orch, hub, nil, nil, cs, "", "")
}

func TestHealthEndpoint(t *testing.T) {
	r := newTestRouter()
	req := httptest.NewRequest("GET", "/api/v1/health", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["status"] != "ok" {
		t.Errorf("expected status ok, got %v", resp["status"])
	}
}

func TestCreateSession(t *testing.T) {
	r := newTestRouter()
	body := `{"mode": "voice_llm"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d", w.Code)
	}

	var resp CreateSessionResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.SessionID == "" {
		t.Error("expected non-empty session_id")
	}
}

func TestCreateSessionLoadsVoiceDialogContext(t *testing.T) {
	inf := &fakeInferenceService{
		avatarInfo:   &pb.AvatarInfo{ModelName: "avatar.flash_head", OutputFps: 25, OutputWidth: 512, OutputHeight: 512},
		voiceConfigs: make(chan inference.VoiceLLMSessionConfig, 1),
	}
	r := newTestRouterWithInference(inf)

	char, err := r.charStore.Create(&character.Character{
		Name:      "Memory",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}
	started := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	if err := r.charStore.SaveConversation(char.ID, "previous-session", started, started.Add(time.Minute), []map[string]any{
		{
			"role":      "user",
			"content":   "我叫小明",
			"timestamp": started.Format(time.RFC3339Nano),
		},
		{
			"role":      "assistant",
			"content":   "我记住了，你叫小明。",
			"timestamp": started.Add(time.Second).Format(time.RFC3339Nano),
		},
	}); err != nil {
		t.Fatal(err)
	}

	body := `{"mode":"voice_llm","character_id":"` + char.ID + `"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}

	select {
	case config := <-inf.voiceConfigs:
		if len(config.DialogContext) != 2 {
			t.Fatalf("expected 2 dialog context items, got %d", len(config.DialogContext))
		}
		if config.DialogContext[0].Role != "user" || config.DialogContext[0].Text != "我叫小明" {
			t.Fatalf("unexpected first dialog context item: %+v", config.DialogContext[0])
		}
		if config.DialogContext[1].Role != "assistant" || config.DialogContext[1].Text != "我记住了，你叫小明。" {
			t.Fatalf("unexpected second dialog context item: %+v", config.DialogContext[1])
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for voice LLM config")
	}
}

func TestCreateSessionInvalidJSON(t *testing.T) {
	r := newTestRouter()
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader("not json"))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestCreateSessionMaxConcurrent(t *testing.T) {
	mgr := orchestrator.NewSessionManager(1)
	r := newTestRouterWithMgr(mgr)

	body := `{"mode": "standard"}`
	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", w.Code)
	}

	req2 := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(body))
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)
	if w2.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", w2.Code)
	}
}

func TestDeleteSession(t *testing.T) {
	r := newTestRouter()

	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(`{"mode":"voice_llm"}`))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	var resp CreateSessionResponse
	json.NewDecoder(w.Body).Decode(&resp)

	req2 := httptest.NewRequest("DELETE", "/api/v1/sessions/"+resp.SessionID, nil)
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)

	if w2.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", w2.Code)
	}
}

func TestDeleteSessionNotFound(t *testing.T) {
	r := newTestRouter()
	req := httptest.NewRequest("DELETE", "/api/v1/sessions/nonexistent", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestSendMessage(t *testing.T) {
	r := newTestRouter()

	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(`{"mode":"voice_llm"}`))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	var resp CreateSessionResponse
	json.NewDecoder(w.Body).Decode(&resp)

	body := `{"text": "Hello"}`
	req2 := httptest.NewRequest("POST", "/api/v1/sessions/"+resp.SessionID+"/message", strings.NewReader(body))
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)

	if w2.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d", w2.Code)
	}
}

func TestSendMessageEmptyText(t *testing.T) {
	r := newTestRouter()

	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(`{"mode":"voice_llm"}`))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	var resp CreateSessionResponse
	json.NewDecoder(w.Body).Decode(&resp)

	req2 := httptest.NewRequest("POST", "/api/v1/sessions/"+resp.SessionID+"/message", strings.NewReader(`{"text": ""}`))
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)

	if w2.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w2.Code)
	}
}

func TestSendMessageInvalidJSON(t *testing.T) {
	r := newTestRouter()

	req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(`{"mode":"voice_llm"}`))
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	var resp CreateSessionResponse
	json.NewDecoder(w.Body).Decode(&resp)

	req2 := httptest.NewRequest("POST", "/api/v1/sessions/"+resp.SessionID+"/message", strings.NewReader("bad"))
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)

	if w2.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid JSON, got %d", w2.Code)
	}
}

func TestListSessions(t *testing.T) {
	r := newTestRouter()

	for i := 0; i < 2; i++ {
		req := httptest.NewRequest("POST", "/api/v1/sessions", strings.NewReader(`{"mode":"voice_llm"}`))
		w := httptest.NewRecorder()
		r.Handler().ServeHTTP(w, req)
	}

	req := httptest.NewRequest("GET", "/api/v1/sessions", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var sessions []map[string]string
	json.NewDecoder(w.Body).Decode(&sessions)
	if len(sessions) != 2 {
		t.Errorf("expected 2 sessions, got %d", len(sessions))
	}
}

func TestCORSHeaders(t *testing.T) {
	r := newTestRouter()
	req := httptest.NewRequest("OPTIONS", "/api/v1/health", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("expected CORS Allow-Origin header")
	}
}
