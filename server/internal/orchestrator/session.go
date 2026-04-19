package orchestrator

import (
	"context"
	"sync"
	"time"
)

type SessionState int

const (
	StateInit SessionState = iota
	StateConnected
	StateListening
	StateProcessing
	StateSpeaking
	StateClosed
)

func (s SessionState) String() string {
	switch s {
	case StateInit:
		return "init"
	case StateConnected:
		return "connected"
	case StateListening:
		return "listening"
	case StateProcessing:
		return "processing"
	case StateSpeaking:
		return "speaking"
	case StateClosed:
		return "closed"
	default:
		return "unknown"
	}
}

type PipelineMode int

const (
	ModeVoiceLLM PipelineMode = iota
	ModeStandard
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Session struct {
	ID             string `json:"id"`
	CharacterID    string `json:"character_id"`
	state          SessionState
	Mode           PipelineMode       `json:"mode"`
	History        []ChatMessage      `json:"history"`
	CreatedAt      time.Time          `json:"created_at"`
	LastActiveAt   time.Time          `json:"last_active_at"`
	PipelineCancel context.CancelFunc `json:"-"`
	// PipelineDone is closed when the pipeline goroutine finishes.
	// TeardownSession waits on this to ensure messages are saved before session deletion.
	PipelineDone chan struct{} `json:"-"`
	// RecordingDir is the absolute path where recordings for this session are saved.
	// Set by the orchestrator when the first recording turn begins.
	RecordingDir string `json:"-"`
	mu           sync.RWMutex
}

func NewSession(id string, mode PipelineMode, characterID string) *Session {
	now := time.Now()
	return &Session{
		ID:           id,
		CharacterID:  characterID,
		state:        StateInit,
		Mode:         mode,
		History:      make([]ChatMessage, 0),
		CreatedAt:    now,
		LastActiveAt: now,
	}
}

func (s *Session) SetState(state SessionState) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = state
	s.LastActiveAt = time.Now()
}

func (s *Session) Touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.LastActiveAt = time.Now()
}

func (s *Session) GetState() SessionState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state
}

// MarkPipelineRunning initializes PipelineDone. Call before launching a pipeline goroutine.
func (s *Session) MarkPipelineRunning() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.PipelineDone = make(chan struct{})
}

// MarkPipelineFinished signals that the pipeline goroutine has completed.
func (s *Session) MarkPipelineFinished() {
	s.mu.RLock()
	ch := s.PipelineDone
	s.mu.RUnlock()
	if ch != nil {
		select {
		case <-ch:
			// already closed
		default:
			close(ch)
		}
	}
}

// WaitPipelineDone blocks until the pipeline goroutine finishes (with timeout).
func (s *Session) WaitPipelineDone(timeout time.Duration) {
	s.mu.RLock()
	ch := s.PipelineDone
	s.mu.RUnlock()
	if ch == nil {
		return
	}
	select {
	case <-ch:
	case <-time.After(timeout):
	}
}

func (s *Session) AddMessage(msg ChatMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.History = append(s.History, msg)
	s.LastActiveAt = time.Now()
}

// SessionManager manages active sessions.
type SessionManager struct {
	sessions     map[string]*Session
	mu           sync.RWMutex
	maxConc      int
	idleTimeout  time.Duration
	stopCleanup  chan struct{}
	OnSessionEnd func(session *Session) // called before session is removed
}

func NewSessionManager(maxConcurrent int) *SessionManager {
	return NewSessionManagerWithTimeout(maxConcurrent, 5*time.Minute)
}

func NewSessionManagerWithTimeout(maxConcurrent int, idleTimeout time.Duration) *SessionManager {
	m := &SessionManager{
		sessions:    make(map[string]*Session),
		maxConc:     maxConcurrent,
		idleTimeout: idleTimeout,
		stopCleanup: make(chan struct{}),
	}
	go m.cleanupLoop()
	return m
}

func (m *SessionManager) cleanupLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			m.evictIdle()
		case <-m.stopCleanup:
			return
		}
	}
}

func (m *SessionManager) evictIdle() {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	for id, s := range m.sessions {
		s.mu.RLock()
		idle := now.Sub(s.LastActiveAt) > m.idleTimeout
		s.mu.RUnlock()
		if idle {
			if m.OnSessionEnd != nil {
				m.OnSessionEnd(s)
			}
			s.mu.Lock()
			s.state = StateClosed
			s.mu.Unlock()
			delete(m.sessions, id)
		}
	}
}

func (m *SessionManager) Stop() {
	close(m.stopCleanup)
}

func (m *SessionManager) Create(id string, mode PipelineMode, characterID string) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.sessions) >= m.maxConc {
		return nil, ErrMaxSessions
	}
	if _, exists := m.sessions[id]; exists {
		return nil, ErrSessionExists
	}

	session := NewSession(id, mode, characterID)
	m.sessions[id] = session
	return session, nil
}

func (m *SessionManager) Get(id string) (*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, exists := m.sessions[id]
	if !exists {
		return nil, ErrSessionNotFound
	}
	return session, nil
}

func (m *SessionManager) Touch(id string) error {
	session, err := m.Get(id)
	if err != nil {
		return err
	}
	session.Touch()
	return nil
}

func (m *SessionManager) Delete(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok {
		if m.OnSessionEnd != nil {
			m.OnSessionEnd(s)
		}
		s.mu.Lock()
		s.state = StateClosed
		s.mu.Unlock()
	}
	delete(m.sessions, id)
}

func (m *SessionManager) List() []*Session {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		result = append(result, s)
	}
	return result
}

func (m *SessionManager) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.sessions)
}
