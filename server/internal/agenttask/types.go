package agenttask

import (
	"encoding/json"
	"time"
)

type Status string

const (
	StatusQueued      Status = "queued"
	StatusRunning     Status = "running"
	StatusWaitingUser Status = "waiting_user"
	StatusCompleted   Status = "completed"
	StatusFailed      Status = "failed"
	StatusCancelled   Status = "cancelled"
)

func (s Status) IsTerminal() bool {
	switch s {
	case StatusCompleted, StatusFailed, StatusCancelled:
		return true
	default:
		return false
	}
}

type Task struct {
	ID            string     `json:"id"`
	SessionID     string     `json:"session_id"`
	CharacterID   string     `json:"character_id,omitempty"`
	Kind          string     `json:"kind"`
	Title         string     `json:"title"`
	UserRequest   string     `json:"user_request"`
	Status        Status     `json:"status"`
	Progress      int        `json:"progress"`
	ResultSummary string     `json:"result_summary,omitempty"`
	CreatedAt     time.Time  `json:"created_at"`
	UpdatedAt     time.Time  `json:"updated_at"`
	FinishedAt    *time.Time `json:"finished_at,omitempty"`
}

type Event struct {
	TaskID    string          `json:"task_id"`
	Seq       int64           `json:"seq"`
	EventType string          `json:"event_type"`
	Status    Status          `json:"status"`
	Message   string          `json:"message,omitempty"`
	Progress  int             `json:"progress"`
	Payload   json.RawMessage `json:"payload,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
}

type Artifact struct {
	ID          string          `json:"id"`
	TaskID      string          `json:"task_id"`
	Type        string          `json:"type"`
	Title       string          `json:"title"`
	MimeType    string          `json:"mime_type"`
	ContentPath string          `json:"-"`
	Metadata    json.RawMessage `json:"metadata,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
}

type CreateTaskInput struct {
	ID          string
	SessionID   string
	CharacterID string
	Kind        string
	Title       string
	UserRequest string
}

type AppendEventInput struct {
	EventType string          `json:"event_type"`
	Status    Status          `json:"status"`
	Message   string          `json:"message"`
	Progress  int             `json:"progress"`
	Payload   json.RawMessage `json:"payload"`
}

type CreateArtifactInput struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Title    string          `json:"title"`
	MimeType string          `json:"mime_type"`
	Content  string          `json:"content"`
	Metadata json.RawMessage `json:"metadata"`
}
