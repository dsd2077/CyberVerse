package agenttask

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/cyberverse/server/internal/ws"
)

type Service struct {
	store   *Store
	hub     *ws.Hub
	onEvent func(*Task, *Event)
}

func NewService(store *Store, hub *ws.Hub) *Service {
	return &Service{
		store: store,
		hub:   hub,
	}
}

func (s *Service) Enabled() bool {
	return s != nil && s.store != nil
}

func (s *Service) Store() *Store {
	if s == nil {
		return nil
	}
	return s.store
}

func (s *Service) SetEventHandler(handler func(*Task, *Event)) {
	if s == nil {
		return
	}
	s.onEvent = handler
}

func (s *Service) AppendEvent(ctx context.Context, taskID string, in AppendEventInput) (*Event, *Task, error) {
	if s == nil || s.store == nil {
		return nil, nil, errors.New("task service is not configured")
	}
	event, task, err := s.store.AppendEvent(ctx, taskID, in)
	if err != nil {
		return nil, nil, err
	}
	s.broadcastTaskEvent(task, event)
	if s.onEvent != nil {
		go s.onEvent(task, event)
	}
	return event, task, nil
}

func (s *Service) CreateArtifact(ctx context.Context, taskID string, in CreateArtifactInput) (*Artifact, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	artifact, err := s.store.CreateArtifact(ctx, taskID, in)
	if err != nil {
		return nil, err
	}
	payload, _ := json.Marshal(map[string]any{
		"artifact_id": artifact.ID,
		"title":       artifact.Title,
		"type":        artifact.Type,
		"mime_type":   artifact.MimeType,
	})
	_, _, _ = s.AppendEvent(ctx, taskID, AppendEventInput{
		EventType: "artifact.created",
		Status:    StatusRunning,
		Message:   "已生成一份资料：" + artifact.Title,
		Progress:  90,
		Payload:   payload,
	})
	return artifact, nil
}

func (s *Service) LatestActiveTask(ctx context.Context, sessionID string) (*Task, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	return s.store.LatestActiveTask(ctx, sessionID)
}

func (s *Service) RecentEventsSummary(ctx context.Context, taskID string, afterSeq int64, limit int) ([]Event, error) {
	if s == nil || s.store == nil {
		return nil, errors.New("task service is not configured")
	}
	return s.store.ListEventsAfter(ctx, taskID, afterSeq, limit)
}

func (s *Service) broadcastTaskEvent(task *Task, event *Event) {
	if s == nil || s.hub == nil || task == nil || event == nil {
		return
	}
	payload := map[string]any{
		"type":       "task_event",
		"task_id":    task.ID,
		"session_id": task.SessionID,
		"seq":        event.Seq,
		"event_type": event.EventType,
		"status":     event.Status,
		"message":    event.Message,
		"progress":   event.Progress,
		"created_at": event.CreatedAt,
		"task":       task,
	}
	if len(event.Payload) > 0 {
		payload["payload"] = event.Payload
	}
	s.hub.BroadcastJSON(task.SessionID, payload)
}
