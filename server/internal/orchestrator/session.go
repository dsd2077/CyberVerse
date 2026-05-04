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
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp,omitempty"`
	TurnSeq   uint64    `json:"turn_seq,omitempty"`
}

type VisualFrame struct {
	Data        []byte
	MimeType    string
	Width       int32
	Height      int32
	Source      string
	TimestampMS int64
	FrameSeq    int64
	ReceivedAt  time.Time
}

type VisualState struct {
	ActiveSource   string
	Frames         []VisualFrame
	LastAcceptedAt time.Time
}

type DialogContextItem struct {
	Role      string
	Text      string
	Timestamp int64
}

type Session struct {
	ID             string `json:"id"`
	CharacterID    string `json:"character_id"`
	state          SessionState
	Mode           PipelineMode        `json:"mode"`
	History        []ChatMessage       `json:"history"`
	DialogContext  []DialogContextItem `json:"-"`
	CreatedAt      time.Time           `json:"created_at"`
	LastActiveAt   time.Time           `json:"last_active_at"`
	PipelineCancel context.CancelFunc  `json:"-"`
	// PipelineDone is closed when the pipeline goroutine finishes.
	// TeardownSession waits on this to ensure messages are saved before session deletion.
	PipelineDone chan struct{} `json:"-"`
	// PipelineSeq increments each time a new pipeline starts.
	PipelineSeq uint64 `json:"-"`
	// TurnSeq increments each time a new conversational turn preempts playback.
	TurnSeq uint64 `json:"-"`
	// VoiceWelcomeSent prevents replaying the greeting when VoiceLLM pipelines restart.
	VoiceWelcomeSent bool `json:"-"`
	// RecordingDir is the absolute path where recordings for this session are saved.
	// Set by the orchestrator when the first recording turn begins.
	RecordingDir string      `json:"-"`
	Visual       VisualState `json:"-"`
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

// MarkPipelineRunning initializes PipelineDone and returns the new pipeline sequence.
// Call before launching a pipeline goroutine.
func (s *Session) MarkPipelineRunning() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.PipelineSeq++
	s.PipelineDone = make(chan struct{})
	return s.PipelineSeq
}

// MarkPipelineFinished signals that the pipeline goroutine has completed.
// Late completions from an older pipeline will not close the current pipeline's done channel.
func (s *Session) MarkPipelineFinished(seq uint64) {
	s.mu.RLock()
	ch := s.PipelineDone
	currentSeq := s.PipelineSeq
	s.mu.RUnlock()
	if seq != currentSeq {
		return
	}
	if ch != nil {
		select {
		case <-ch:
			// already closed
		default:
			close(ch)
		}
	}
}

func (s *Session) IsCurrentPipeline(seq uint64) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.PipelineSeq == seq
}

// MarkTurnStarted returns a monotonically increasing conversational turn sequence.
func (s *Session) MarkTurnStarted() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.TurnSeq++
	s.LastActiveAt = time.Now()
	return s.TurnSeq
}

func (s *Session) CurrentTurnSeq() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.TurnSeq
}

func (s *Session) IsCurrentTurn(seq uint64) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.TurnSeq == seq
}

func (s *Session) ConsumeVoiceWelcomeMessage(message string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.VoiceWelcomeSent {
		return ""
	}
	s.VoiceWelcomeSent = true
	return message
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
	now := time.Now()
	if msg.Timestamp.IsZero() {
		msg.Timestamp = now.UTC()
	}
	if msg.TurnSeq > 0 && msg.Role == "assistant" {
		for i := len(s.History) - 1; i >= 0; i-- {
			if s.History[i].TurnSeq == msg.TurnSeq {
				insertAt := i + 1
				s.History = append(s.History, ChatMessage{})
				copy(s.History[insertAt+1:], s.History[insertAt:])
				s.History[insertAt] = msg
				s.LastActiveAt = now
				return
			}
		}
	}
	s.History = append(s.History, msg)
	s.LastActiveAt = now
}

func (s *Session) HistorySnapshot() []ChatMessage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]ChatMessage(nil), s.History...)
}

// ConversationSnapshot returns session metadata and a copy of history for persistence.
func (s *Session) ConversationSnapshot() (sessionID, characterID string, createdAt, lastActiveAt time.Time, history []ChatMessage) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ID, s.CharacterID, s.CreatedAt, s.LastActiveAt, append([]ChatMessage(nil), s.History...)
}

func (s *Session) SetDialogContext(items []DialogContextItem) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.DialogContext = append([]DialogContextItem(nil), items...)
}

func (s *Session) DialogContextSnapshot() []DialogContextItem {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]DialogContextItem(nil), s.DialogContext...)
}

func copyVisualFrame(frame VisualFrame) VisualFrame {
	copied := frame
	if frame.Data != nil {
		copied.Data = append([]byte(nil), frame.Data...)
	}
	return copied
}

func (s *Session) StartVisualInput(source string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Visual.ActiveSource != source {
		s.Visual.Frames = nil
		s.Visual.LastAcceptedAt = time.Time{}
	}
	s.Visual.ActiveSource = source
	s.LastActiveAt = time.Now()
}

func (s *Session) StopVisualInput(source string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if source == "" || s.Visual.ActiveSource == "" || s.Visual.ActiveSource == source {
		s.Visual = VisualState{}
	}
	s.LastActiveAt = time.Now()
}

func (s *Session) StoreVisualFrame(frame VisualFrame, maxRecent int, minInterval time.Duration, now time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if maxRecent <= 0 {
		maxRecent = 1
	}
	if !s.Visual.LastAcceptedAt.IsZero() && minInterval > 0 && now.Sub(s.Visual.LastAcceptedAt) < minInterval {
		return false
	}
	if s.Visual.ActiveSource != frame.Source {
		s.Visual.ActiveSource = frame.Source
		s.Visual.Frames = nil
	}
	frame.ReceivedAt = now
	s.Visual.Frames = append(s.Visual.Frames, copyVisualFrame(frame))
	if len(s.Visual.Frames) > maxRecent {
		s.Visual.Frames = append([]VisualFrame(nil), s.Visual.Frames[len(s.Visual.Frames)-maxRecent:]...)
	}
	s.Visual.LastAcceptedAt = now
	s.LastActiveAt = now
	return true
}

func (s *Session) LatestVisualFrames(now time.Time, ttl time.Duration) []VisualFrame {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.Visual.Frames) == 0 || ttl <= 0 {
		return nil
	}
	frames := make([]VisualFrame, 0, len(s.Visual.Frames))
	for _, frame := range s.Visual.Frames {
		if now.Sub(frame.ReceivedAt) <= ttl {
			frames = append(frames, copyVisualFrame(frame))
		}
	}
	return frames
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
