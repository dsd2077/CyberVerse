package orchestrator

import (
	"testing"
	"time"
)

func TestNewSession(t *testing.T) {
	s := NewSession("test-1", ModeVoiceLLM, "")
	if s.ID != "test-1" {
		t.Errorf("expected ID test-1, got %s", s.ID)
	}
	if s.GetState() != StateInit {
		t.Errorf("expected state Init, got %v", s.GetState())
	}
	if s.Mode != ModeVoiceLLM {
		t.Errorf("expected mode VoiceLLM, got %v", s.Mode)
	}
}

func TestSessionSetGetState(t *testing.T) {
	s := NewSession("test-1", ModeStandard, "")
	s.SetState(StateConnected)
	if s.GetState() != StateConnected {
		t.Errorf("expected Connected, got %v", s.GetState())
	}
}

func TestSessionAddMessage(t *testing.T) {
	s := NewSession("test-1", ModeStandard, "")
	s.AddMessage(ChatMessage{Role: "user", Content: "hello"})
	if len(s.History) != 1 {
		t.Errorf("expected 1 message, got %d", len(s.History))
	}
	if s.History[0].Content != "hello" {
		t.Errorf("expected 'hello', got '%s'", s.History[0].Content)
	}
}

func TestSessionTouch(t *testing.T) {
	s := NewSession("test-1", ModeStandard, "")
	before := s.LastActiveAt

	time.Sleep(10 * time.Millisecond)
	s.Touch()

	if !s.LastActiveAt.After(before) {
		t.Fatalf("expected LastActiveAt to advance, before=%v after=%v", before, s.LastActiveAt)
	}
}

func TestSessionStateString(t *testing.T) {
	tests := []struct {
		state    SessionState
		expected string
	}{
		{StateInit, "init"},
		{StateConnected, "connected"},
		{StateListening, "listening"},
		{StateProcessing, "processing"},
		{StateSpeaking, "speaking"},
		{StateClosed, "closed"},
		{SessionState(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.state.String(); got != tt.expected {
			t.Errorf("state %d: expected %s, got %s", tt.state, tt.expected, got)
		}
	}
}

func TestSessionManagerCreate(t *testing.T) {
	mgr := NewSessionManager(2)
	defer mgr.Stop()
	s1, err := mgr.Create("s1", ModeVoiceLLM, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s1.ID != "s1" {
		t.Errorf("expected s1, got %s", s1.ID)
	}
	if mgr.Count() != 1 {
		t.Errorf("expected count 1, got %d", mgr.Count())
	}
}

func TestSessionManagerMaxConcurrent(t *testing.T) {
	mgr := NewSessionManager(1)
	defer mgr.Stop()
	_, err := mgr.Create("s1", ModeVoiceLLM, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = mgr.Create("s2", ModeVoiceLLM, "")
	if err != ErrMaxSessions {
		t.Errorf("expected ErrMaxSessions, got %v", err)
	}
}

func TestSessionManagerDuplicate(t *testing.T) {
	mgr := NewSessionManager(10)
	defer mgr.Stop()
	mgr.Create("s1", ModeVoiceLLM, "")
	_, err := mgr.Create("s1", ModeVoiceLLM, "")
	if err != ErrSessionExists {
		t.Errorf("expected ErrSessionExists, got %v", err)
	}
}

func TestSessionManagerGetNotFound(t *testing.T) {
	mgr := NewSessionManager(10)
	defer mgr.Stop()
	_, err := mgr.Get("nonexistent")
	if err != ErrSessionNotFound {
		t.Errorf("expected ErrSessionNotFound, got %v", err)
	}
}

func TestSessionManagerDelete(t *testing.T) {
	mgr := NewSessionManager(10)
	defer mgr.Stop()
	mgr.Create("s1", ModeVoiceLLM, "")
	mgr.Delete("s1")
	if mgr.Count() != 0 {
		t.Errorf("expected count 0, got %d", mgr.Count())
	}
}

func TestSessionManagerList(t *testing.T) {
	mgr := NewSessionManager(10)
	defer mgr.Stop()
	mgr.Create("s1", ModeVoiceLLM, "")
	mgr.Create("s2", ModeStandard, "")
	list := mgr.List()
	if len(list) != 2 {
		t.Errorf("expected 2 sessions, got %d", len(list))
	}
}

func TestSessionManagerIdleEviction(t *testing.T) {
	mgr := NewSessionManagerWithTimeout(10, 50*time.Millisecond)
	defer mgr.Stop()
	mgr.Create("s1", ModeVoiceLLM, "")

	// Wait for idle timeout + cleanup interval
	time.Sleep(200 * time.Millisecond)
	mgr.evictIdle()

	if mgr.Count() != 0 {
		t.Errorf("expected 0 sessions after idle eviction, got %d", mgr.Count())
	}
}

func TestSessionManagerTouchKeepsSessionAlive(t *testing.T) {
	mgr := NewSessionManagerWithTimeout(10, 50*time.Millisecond)
	defer mgr.Stop()
	mgr.Create("s1", ModeVoiceLLM, "")

	time.Sleep(30 * time.Millisecond)
	if err := mgr.Touch("s1"); err != nil {
		t.Fatalf("touch failed: %v", err)
	}

	time.Sleep(30 * time.Millisecond)
	mgr.evictIdle()

	if mgr.Count() != 1 {
		t.Fatalf("expected touched session to stay alive, got %d sessions", mgr.Count())
	}
}

func TestSessionManagerDeleteSetsStateClosed(t *testing.T) {
	mgr := NewSessionManager(10)
	defer mgr.Stop()
	s, _ := mgr.Create("s1", ModeVoiceLLM, "")
	mgr.Delete("s1")
	if s.GetState() != StateClosed {
		t.Errorf("expected StateClosed after delete, got %v", s.GetState())
	}
}
