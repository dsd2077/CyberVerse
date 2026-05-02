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
	if s.History[0].Timestamp.IsZero() {
		t.Fatal("expected message timestamp to be set")
	}
}

func TestSessionAddMessageInsertsAssistantAfterMatchingTurn(t *testing.T) {
	s := NewSession("test-1", ModeVoiceLLM, "")
	s.AddMessage(ChatMessage{Role: "user", Content: "first", TurnSeq: 1})
	s.AddMessage(ChatMessage{Role: "user", Content: "second", TurnSeq: 2})
	s.AddMessage(ChatMessage{Role: "assistant", Content: "first answer", TurnSeq: 1})
	s.AddMessage(ChatMessage{Role: "assistant", Content: "second answer", TurnSeq: 2})

	got := s.HistorySnapshot()
	if len(got) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(got))
	}
	want := []string{"first", "first answer", "second", "second answer"}
	for i, text := range want {
		if got[i].Content != text {
			t.Fatalf("message %d: expected %q, got %+v", i, text, got)
		}
	}
}

func TestSessionDialogContextSnapshot(t *testing.T) {
	s := NewSession("test-1", ModeVoiceLLM, "")
	s.SetDialogContext([]DialogContextItem{
		{Role: "user", Text: "hello", Timestamp: 1000},
	})
	got := s.DialogContextSnapshot()
	if len(got) != 1 {
		t.Fatalf("expected 1 dialog context item, got %d", len(got))
	}
	got[0].Text = "mutated"
	if s.DialogContextSnapshot()[0].Text != "hello" {
		t.Fatal("expected dialog context snapshot to be isolated from caller mutation")
	}
}

func TestBuildDoubaoDialogContextKeepsRecentCompletePairs(t *testing.T) {
	now := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"session_id": "s1", "role": "assistant", "content": "orphan assistant", "timestamp": now.Add(-10 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s1", "role": "user", "content": "old question", "timestamp": now.Add(-9 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s1", "role": "assistant", "content": "old answer", "timestamp": now.Add(-8 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s1", "role": "system", "content": "ignore me", "timestamp": now.Add(-7 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "user", "content": "new question", "timestamp": now.Add(-6 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "assistant", "content": "new answer", "timestamp": now.Add(-5 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "user", "content": "incomplete", "timestamp": now.Add(-4 * time.Second).Format(time.RFC3339Nano)},
	}

	got := buildDoubaoDialogContext(messages, 1, now)
	if len(got) != 2 {
		t.Fatalf("expected 2 items for the most recent complete pair, got %d", len(got))
	}
	if got[0].Role != "user" || got[0].Text != "new question" {
		t.Fatalf("unexpected first item: %+v", got[0])
	}
	if got[1].Role != "assistant" || got[1].Text != "new answer" {
		t.Fatalf("unexpected second item: %+v", got[1])
	}
	if got[0].Timestamp >= got[1].Timestamp {
		t.Fatalf("expected strictly increasing timestamps: %+v", got)
	}
}

func TestBuildDoubaoDialogContextFallsBackForLegacyTimestamps(t *testing.T) {
	now := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"session_id": "legacy-session", "role": "user", "content": "legacy question", "timestamp": "2026-05-01T12:00:00Z"},
		{"session_id": "legacy-session", "role": "assistant", "content": "legacy answer", "timestamp": "2026-05-01T12:00:00Z"},
	}

	got := buildDoubaoDialogContext(messages, 20, now)
	if len(got) != 2 {
		t.Fatalf("expected 2 items, got %d", len(got))
	}
	if got[0].Timestamp >= got[1].Timestamp {
		t.Fatalf("expected same legacy timestamp to be made strictly increasing: %+v", got)
	}
	if got[1].Timestamp > now.UnixMilli() {
		t.Fatalf("expected timestamp not to exceed now: %+v", got)
	}
}

func TestBuildDoubaoDialogContextDropsMessagesWithoutSessionID(t *testing.T) {
	now := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"role": "user", "content": "unknown session user", "timestamp": now.Add(-2 * time.Second).Format(time.RFC3339Nano)},
		{"role": "assistant", "content": "unknown session assistant", "timestamp": now.Add(-time.Second).Format(time.RFC3339Nano)},
	}

	got := buildDoubaoDialogContext(messages, 20, now)
	if len(got) != 0 {
		t.Fatalf("expected messages without session_id to be dropped, got %+v", got)
	}
}

func TestBuildDoubaoDialogContextMergesConsecutiveUsersInSameSession(t *testing.T) {
	now := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"session_id": "s1", "role": "user", "content": "你好啊，大头包。", "timestamp": now.Add(-5 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s1", "role": "user", "content": "不跟你说了，拜拜。", "timestamp": now.Add(-4 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s1", "role": "assistant", "content": "拜拜，回头见。", "timestamp": now.Add(-3 * time.Second).Format(time.RFC3339Nano)},
	}

	got := buildDoubaoDialogContext(messages, 20, now)
	if len(got) != 2 {
		t.Fatalf("expected 2 items, got %d", len(got))
	}
	if got[0].Role != "user" || got[0].Text != "你好啊，大头包。\n不跟你说了，拜拜。" {
		t.Fatalf("unexpected merged user item: %+v", got[0])
	}
	if got[1].Role != "assistant" || got[1].Text != "拜拜，回头见。" {
		t.Fatalf("unexpected assistant item: %+v", got[1])
	}
}

func TestBuildDoubaoDialogContextDoesNotPairAcrossSessions(t *testing.T) {
	now := time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC)
	messages := []map[string]any{
		{"session_id": "s1", "role": "user", "content": "orphan user", "timestamp": now.Add(-5 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "assistant", "content": "orphan assistant", "timestamp": now.Add(-4 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "user", "content": "paired user", "timestamp": now.Add(-3 * time.Second).Format(time.RFC3339Nano)},
		{"session_id": "s2", "role": "assistant", "content": "paired assistant", "timestamp": now.Add(-2 * time.Second).Format(time.RFC3339Nano)},
	}

	got := buildDoubaoDialogContext(messages, 20, now)
	if len(got) != 2 {
		t.Fatalf("expected only the s2 pair, got %d items: %+v", len(got), got)
	}
	if got[0].Text != "paired user" || got[1].Text != "paired assistant" {
		t.Fatalf("unexpected cross-session pairing result: %+v", got)
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
