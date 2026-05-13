package agenttask

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

var ErrNotFound = errors.New("task not found")
var ErrTerminal = errors.New("task is already terminal")

func validateStorageID(kind, id string) error {
	if id == "." || id == ".." || strings.ContainsAny(id, `/\`) {
		return fmt.Errorf("%s id must not contain path separators or traversal segments", kind)
	}
	return nil
}

type Store struct {
	db          *sql.DB
	artifactDir string
}

func OpenStore(dbPath, artifactDir string) (*Store, error) {
	if strings.TrimSpace(dbPath) == "" {
		return nil, errors.New("task database path is required")
	}
	if strings.TrimSpace(artifactDir) == "" {
		artifactDir = filepath.Join(filepath.Dir(dbPath), "artifacts")
	}
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("create task db dir: %w", err)
	}
	if err := os.MkdirAll(artifactDir, 0755); err != nil {
		return nil, fmt.Errorf("create artifact dir: %w", err)
	}
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	s := &Store{db: db, artifactDir: artifactDir}
	if err := s.migrate(context.Background()); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) ArtifactDir() string {
	if s == nil {
		return ""
	}
	return s.artifactDir
}

func (s *Store) migrate(ctx context.Context) error {
	if _, err := s.db.ExecContext(ctx, `PRAGMA journal_mode=WAL;`); err != nil {
		return err
	}
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS schema_migrations (
			version INTEGER PRIMARY KEY,
			applied_at TEXT NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS tasks (
			id TEXT PRIMARY KEY,
			session_id TEXT NOT NULL,
			character_id TEXT NOT NULL DEFAULT '',
			kind TEXT NOT NULL,
			title TEXT NOT NULL,
			user_request TEXT NOT NULL,
			status TEXT NOT NULL,
			progress INTEGER NOT NULL DEFAULT 0,
			result_summary TEXT NOT NULL DEFAULT '',
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			finished_at TEXT NOT NULL DEFAULT ''
		);`,
		`CREATE INDEX IF NOT EXISTS idx_tasks_session_updated ON tasks(session_id, updated_at DESC);`,
		`CREATE INDEX IF NOT EXISTS idx_tasks_session_status ON tasks(session_id, status, updated_at DESC);`,
		`CREATE TABLE IF NOT EXISTS task_events (
			task_id TEXT NOT NULL,
			seq INTEGER NOT NULL,
			event_type TEXT NOT NULL,
			status TEXT NOT NULL,
			message TEXT NOT NULL DEFAULT '',
			progress INTEGER NOT NULL DEFAULT 0,
			payload_json TEXT NOT NULL DEFAULT '',
			created_at TEXT NOT NULL,
			PRIMARY KEY(task_id, seq),
			FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
		);`,
		`CREATE TABLE IF NOT EXISTS artifacts (
			id TEXT PRIMARY KEY,
			task_id TEXT NOT NULL,
			type TEXT NOT NULL,
			title TEXT NOT NULL,
			mime_type TEXT NOT NULL,
			content_path TEXT NOT NULL,
			metadata_json TEXT NOT NULL DEFAULT '',
			created_at TEXT NOT NULL,
			FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
		);`,
		`CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id, created_at);`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.ExecContext(ctx, stmt); err != nil {
			return err
		}
	}
	return nil
}

func nowString() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func parseTimeValue(raw string) time.Time {
	if raw == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339Nano, raw)
	if err != nil {
		return time.Time{}
	}
	return t
}

func parseOptionalTime(raw string) *time.Time {
	t := parseTimeValue(raw)
	if t.IsZero() {
		return nil
	}
	return &t
}

func normalizeProgress(progress int) int {
	if progress < 0 {
		return 0
	}
	if progress > 100 {
		return 100
	}
	return progress
}

func normalizeKind(kind string) string {
	kind = strings.ToLower(strings.TrimSpace(kind))
	if kind == "" {
		return "research"
	}
	return kind
}

func defaultTitle(kind, userRequest string) string {
	title := strings.TrimSpace(userRequest)
	if title == "" {
		title = strings.TrimSpace(kind)
	}
	if len([]rune(title)) > 48 {
		rs := []rune(title)
		title = string(rs[:48])
	}
	return title
}

func (s *Store) CreateTask(ctx context.Context, in CreateTaskInput) (*Task, error) {
	if strings.TrimSpace(in.SessionID) == "" {
		return nil, errors.New("session id is required")
	}
	in.UserRequest = strings.TrimSpace(in.UserRequest)
	if in.UserRequest == "" {
		return nil, errors.New("user request is required")
	}
	kind := normalizeKind(in.Kind)
	title := strings.TrimSpace(in.Title)
	if title == "" {
		title = defaultTitle(kind, in.UserRequest)
	}
	id := strings.TrimSpace(in.ID)
	if id == "" {
		id = uuid.NewString()
	} else if err := validateStorageID("task", id); err != nil {
		return nil, err
	}
	now := nowString()
	task := &Task{
		ID:          id,
		SessionID:   strings.TrimSpace(in.SessionID),
		CharacterID: strings.TrimSpace(in.CharacterID),
		Kind:        kind,
		Title:       title,
		UserRequest: in.UserRequest,
		Status:      StatusQueued,
		CreatedAt:   parseTimeValue(now),
		UpdatedAt:   parseTimeValue(now),
	}
	_, err := s.db.ExecContext(ctx, `INSERT INTO tasks
		(id, session_id, character_id, kind, title, user_request, status, progress, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
		task.ID, task.SessionID, task.CharacterID, task.Kind, task.Title, task.UserRequest, task.Status, now, now)
	if err != nil {
		return nil, err
	}
	return task, nil
}

func (s *Store) GetTask(ctx context.Context, id string) (*Task, error) {
	row := s.db.QueryRowContext(ctx, `SELECT id, session_id, character_id, kind, title, user_request,
		status, progress, result_summary, created_at, updated_at, finished_at FROM tasks WHERE id = ?`, id)
	task, err := scanTask(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	return task, err
}

type taskScanner interface {
	Scan(dest ...any) error
}

func scanTask(row taskScanner) (*Task, error) {
	var task Task
	var status, createdAt, updatedAt, finishedAt string
	if err := row.Scan(&task.ID, &task.SessionID, &task.CharacterID, &task.Kind, &task.Title,
		&task.UserRequest, &status, &task.Progress, &task.ResultSummary, &createdAt, &updatedAt, &finishedAt); err != nil {
		return nil, err
	}
	task.Status = Status(status)
	task.CreatedAt = parseTimeValue(createdAt)
	task.UpdatedAt = parseTimeValue(updatedAt)
	task.FinishedAt = parseOptionalTime(finishedAt)
	return &task, nil
}

func (s *Store) ListSessionTasks(ctx context.Context, sessionID string, limit int) ([]Task, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, session_id, character_id, kind, title, user_request,
		status, progress, result_summary, created_at, updated_at, finished_at
		FROM tasks WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?`, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	tasks := make([]Task, 0)
	for rows.Next() {
		task, err := scanTask(rows)
		if err != nil {
			return nil, err
		}
		tasks = append(tasks, *task)
	}
	return tasks, rows.Err()
}

func (s *Store) LatestActiveTask(ctx context.Context, sessionID string) (*Task, error) {
	row := s.db.QueryRowContext(ctx, `SELECT id, session_id, character_id, kind, title, user_request,
		status, progress, result_summary, created_at, updated_at, finished_at
		FROM tasks
		WHERE session_id = ? AND status IN (?, ?, ?)
		ORDER BY updated_at DESC LIMIT 1`, sessionID, StatusQueued, StatusRunning, StatusWaitingUser)
	task, err := scanTask(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	return task, err
}

func (s *Store) ActiveTaskCount(ctx context.Context, sessionID string) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM tasks
		WHERE session_id = ? AND status IN (?, ?, ?)`, sessionID, StatusQueued, StatusRunning, StatusWaitingUser).Scan(&count)
	return count, err
}

func (s *Store) AppendEvent(ctx context.Context, taskID string, in AppendEventInput) (*Event, *Task, error) {
	if strings.TrimSpace(taskID) == "" {
		return nil, nil, errors.New("task id is required")
	}
	if strings.TrimSpace(in.EventType) == "" {
		in.EventType = "task.updated"
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, err
	}
	defer tx.Rollback()

	row := tx.QueryRowContext(ctx, `SELECT id, session_id, character_id, kind, title, user_request,
		status, progress, result_summary, created_at, updated_at, finished_at FROM tasks WHERE id = ?`, taskID)
	task, err := scanTask(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil, ErrNotFound
	}
	if err != nil {
		return nil, nil, err
	}
	if task.Status.IsTerminal() {
		return nil, nil, ErrTerminal
	}

	status := in.Status
	if status == "" {
		status = task.Status
	}
	progress := normalizeProgress(in.Progress)
	if progress == 0 && task.Progress > 0 && status != StatusQueued {
		progress = task.Progress
	}
	if status == StatusCompleted && progress < 100 {
		progress = 100
	}
	message := strings.TrimSpace(in.Message)
	payload := strings.TrimSpace(string(in.Payload))
	now := nowString()
	var seq int64
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(seq), 0) + 1 FROM task_events WHERE task_id = ?`, taskID).Scan(&seq); err != nil {
		return nil, nil, err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO task_events
		(task_id, seq, event_type, status, message, progress, payload_json, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		taskID, seq, strings.TrimSpace(in.EventType), status, message, progress, payload, now); err != nil {
		return nil, nil, err
	}

	finishedAt := ""
	if status.IsTerminal() {
		finishedAt = now
	}
	resultSummary := task.ResultSummary
	if status == StatusCompleted && message != "" {
		resultSummary = message
	}
	if _, err := tx.ExecContext(ctx, `UPDATE tasks SET status = ?, progress = ?, result_summary = ?,
		updated_at = ?, finished_at = CASE WHEN ? != '' THEN ? ELSE finished_at END WHERE id = ?`,
		status, progress, resultSummary, now, finishedAt, finishedAt, taskID); err != nil {
		return nil, nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, nil, err
	}

	event := &Event{
		TaskID:    taskID,
		Seq:       seq,
		EventType: strings.TrimSpace(in.EventType),
		Status:    status,
		Message:   message,
		Progress:  progress,
		Payload:   json.RawMessage(payload),
		CreatedAt: parseTimeValue(now),
	}
	updated, err := s.GetTask(ctx, taskID)
	if err != nil {
		return nil, nil, err
	}
	return event, updated, nil
}

func (s *Store) ListEventsAfter(ctx context.Context, taskID string, afterSeq int64, limit int) ([]Event, error) {
	if limit <= 0 || limit > 500 {
		limit = 200
	}
	rows, err := s.db.QueryContext(ctx, `SELECT task_id, seq, event_type, status, message, progress, payload_json, created_at
		FROM task_events WHERE task_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?`, taskID, afterSeq, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	events := make([]Event, 0)
	for rows.Next() {
		var ev Event
		var status, payload, createdAt string
		if err := rows.Scan(&ev.TaskID, &ev.Seq, &ev.EventType, &status, &ev.Message, &ev.Progress, &payload, &createdAt); err != nil {
			return nil, err
		}
		ev.Status = Status(status)
		ev.Payload = json.RawMessage(payload)
		ev.CreatedAt = parseTimeValue(createdAt)
		events = append(events, ev)
	}
	return events, rows.Err()
}

func (s *Store) CreateArtifact(ctx context.Context, taskID string, in CreateArtifactInput) (*Artifact, error) {
	if strings.TrimSpace(taskID) == "" {
		return nil, errors.New("task id is required")
	}
	if strings.TrimSpace(in.Content) == "" {
		return nil, errors.New("artifact content is required")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	row := tx.QueryRowContext(ctx, `SELECT id, session_id, character_id, kind, title, user_request,
		status, progress, result_summary, created_at, updated_at, finished_at FROM tasks WHERE id = ?`, taskID)
	task, err := scanTask(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if task.Status.IsTerminal() {
		return nil, ErrTerminal
	}

	typ := strings.TrimSpace(in.Type)
	if typ == "" {
		typ = "markdown"
	}
	mimeType := strings.TrimSpace(in.MimeType)
	if mimeType == "" {
		mimeType = "text/markdown; charset=utf-8"
	}
	title := strings.TrimSpace(in.Title)
	if title == "" {
		title = "任务资料"
	}
	id := strings.TrimSpace(in.ID)
	if id == "" {
		id = uuid.NewString()
	} else if err := validateStorageID("artifact", id); err != nil {
		return nil, err
	}
	taskDir := filepath.Join(s.artifactDir, taskID)
	if err := os.MkdirAll(taskDir, 0755); err != nil {
		return nil, err
	}
	ext := ".txt"
	if strings.Contains(mimeType, "markdown") || typ == "markdown" {
		ext = ".md"
	}
	if strings.Contains(mimeType, "html") || typ == "html" {
		ext = ".html"
	}
	contentPath := filepath.Join(taskDir, id+ext)
	if err := os.WriteFile(contentPath, []byte(in.Content), 0644); err != nil {
		return nil, err
	}
	metadata := strings.TrimSpace(string(in.Metadata))
	now := nowString()
	_, err = tx.ExecContext(ctx, `INSERT INTO artifacts
		(id, task_id, type, title, mime_type, content_path, metadata_json, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		id, taskID, typ, title, mimeType, contentPath, metadata, now)
	if err != nil {
		_ = os.Remove(contentPath)
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		_ = os.Remove(contentPath)
		return nil, err
	}
	return &Artifact{
		ID:          id,
		TaskID:      taskID,
		Type:        typ,
		Title:       title,
		MimeType:    mimeType,
		ContentPath: contentPath,
		Metadata:    json.RawMessage(metadata),
		CreatedAt:   parseTimeValue(now),
	}, nil
}

func (s *Store) GetArtifact(ctx context.Context, taskID, artifactID string) (*Artifact, []byte, error) {
	row := s.db.QueryRowContext(ctx, `SELECT id, task_id, type, title, mime_type, content_path, metadata_json, created_at
		FROM artifacts WHERE task_id = ? AND id = ?`, taskID, artifactID)
	var artifact Artifact
	var metadata, createdAt string
	if err := row.Scan(&artifact.ID, &artifact.TaskID, &artifact.Type, &artifact.Title, &artifact.MimeType,
		&artifact.ContentPath, &metadata, &createdAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil, ErrNotFound
		}
		return nil, nil, err
	}
	artifact.Metadata = json.RawMessage(metadata)
	artifact.CreatedAt = parseTimeValue(createdAt)
	content, err := os.ReadFile(artifact.ContentPath)
	if err != nil {
		return nil, nil, err
	}
	return &artifact, content, nil
}
