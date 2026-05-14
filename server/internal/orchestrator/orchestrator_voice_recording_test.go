package orchestrator

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/agenttask"
	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/inference"
	pb "github.com/cyberverse/server/internal/pb"
	"github.com/cyberverse/server/internal/recording"
	"github.com/cyberverse/server/internal/ws"
)

type voiceRecordingInferenceStub struct {
	outputs       chan *pb.VoiceLLMOutput
	errs          chan error
	started       chan struct{}
	avatarStarted chan struct{}
	avatarRelease chan struct{}
	avatarDone    chan struct{}
	ttsTexts      chan string
}

func newVoiceRecordingInferenceStub() *voiceRecordingInferenceStub {
	return &voiceRecordingInferenceStub{
		outputs:       make(chan *pb.VoiceLLMOutput, 16),
		errs:          make(chan error),
		started:       make(chan struct{}),
		avatarStarted: make(chan struct{}),
		avatarRelease: make(chan struct{}),
		avatarDone:    make(chan struct{}),
		ttsTexts:      make(chan string, 4),
	}
}

func (f *voiceRecordingInferenceStub) HealthCheck(context.Context) error { return nil }

func (f *voiceRecordingInferenceStub) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	return nil, nil
}

func (f *voiceRecordingInferenceStub) SetAvatar(context.Context, string, []byte, string) error {
	return nil
}

func (f *voiceRecordingInferenceStub) GenerateAvatarStream(ctx context.Context, _ <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	close(f.avatarStarted)
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	go func() {
		defer close(f.avatarDone)
		select {
		case <-f.avatarRelease:
		case <-ctx.Done():
		}
		close(videoCh)
		close(errCh)
	}()
	return videoCh, errCh
}

func (f *voiceRecordingInferenceStub) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}

func (f *voiceRecordingInferenceStub) GenerateLLMStream(context.Context, string, []inference.ChatMessage, inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	ch := make(chan *pb.LLMChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *voiceRecordingInferenceStub) SynthesizeSpeechStream(ctx context.Context, textCh <-chan string, _ inference.TTSConfig) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	go func() {
		defer close(ch)
		defer close(errCh)
		for text := range textCh {
			select {
			case f.ttsTexts <- text:
			case <-ctx.Done():
				return
			}
			ch <- &pb.AudioChunk{Data: []byte{1, 0, 2, 0}, SampleRate: 24000}
			ch <- &pb.AudioChunk{SampleRate: 24000, IsFinal: true}
		}
	}()
	return ch, errCh
}

func (f *voiceRecordingInferenceStub) TranscribeStream(context.Context, <-chan []byte, inference.ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *voiceRecordingInferenceStub) CheckVoice(context.Context, inference.VoiceLLMSessionConfig) (string, error) {
	return "", nil
}

func (f *voiceRecordingInferenceStub) ConverseStream(context.Context, <-chan inference.VoiceLLMInputEvent, inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	close(f.started)
	return f.outputs, f.errs
}

func (f *voiceRecordingInferenceStub) Interrupt(context.Context, string) error { return nil }
func (f *voiceRecordingInferenceStub) Close() error                            { return nil }

func newVoiceRecordingHarness(t *testing.T) (*Orchestrator, *Session, *character.Store, *voiceRecordingInferenceStub) {
	t.Helper()

	root := t.TempDir()
	charStore, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := charStore.Create(&character.Character{Name: "Recorder", VoiceType: "温柔文雅"})
	if err != nil {
		t.Fatal(err)
	}

	sessionMgr := NewSessionManager(4)
	t.Cleanup(sessionMgr.Stop)
	session, err := sessionMgr.Create("session-recording", ModeOmni, char.ID)
	if err != nil {
		t.Fatal(err)
	}

	inf := newVoiceRecordingInferenceStub()
	recorder := recording.NewVideoRecorder(config.RecordingConfig{
		Enabled:   true,
		OutputDir: filepath.Join(root, "recordings"),
		CRF:       23,
	})
	orch := New(inf, nil, sessionMgr, recorder, charStore)

	return orch, session, charStore, inf
}

func waitForFile(t *testing.T, path string) []byte {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(path)
		if err == nil {
			return data
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for file %s", path)
	return nil
}

func TestVoiceTurnSavesTranscriptAndRawAudioOnFinalBeforeAvatarDone(t *testing.T) {
	orch, session, charStore, inf := newVoiceRecordingHarness(t)

	if err := orch.HandleAudioStream(context.Background(), session.ID, make(chan []byte)); err != nil {
		t.Fatal(err)
	}
	<-inf.started

	inf.outputs <- &pb.VoiceLLMOutput{
		UserTranscript: "用户问题",
		QuestionId:     "q1",
		ReplyId:        "r1",
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		Transcript: "完整回答",
		QuestionId: "q1",
		ReplyId:    "r1",
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		Audio:      &pb.AudioChunk{Data: []byte{1, 0, 2, 0}, SampleRate: 24000},
		QuestionId: "q1",
		ReplyId:    "r1",
	}
	<-inf.avatarStarted
	inf.outputs <- &pb.VoiceLLMOutput{
		Audio:      &pb.AudioChunk{SampleRate: 24000, IsFinal: true},
		Transcript: "完整回答",
		IsFinal:    true,
		QuestionId: "q1",
		ReplyId:    "r1",
	}

	sessionDir := charStore.SessionRecordingDir(session.CharacterID, session.ID, session.CreatedAt)
	if got := string(waitForFile(t, filepath.Join(sessionDir, "turn1.txt"))); got != "完整回答" {
		t.Fatalf("unexpected transcript: %q", got)
	}
	sessionJSON := waitForFile(t, filepath.Join(sessionDir, "session.json"))
	var saved struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(sessionJSON, &saved); err != nil {
		t.Fatalf("unmarshal session.json: %v", err)
	}
	if len(saved.Messages) != 2 {
		t.Fatalf("expected user and assistant in session.json, got %+v", saved.Messages)
	}
	if saved.Messages[0]["role"] != "user" || saved.Messages[0]["content"] != "用户问题" {
		t.Fatalf("unexpected user message in session.json: %+v", saved.Messages)
	}
	if saved.Messages[1]["role"] != "assistant" || saved.Messages[1]["content"] != "完整回答" {
		t.Fatalf("unexpected assistant message in session.json: %+v", saved.Messages)
	}
	if got := waitForFile(t, filepath.Join(sessionDir, "turn1-raw.wav")); len(got) <= 44 {
		t.Fatalf("expected raw wav data, got %d bytes", len(got))
	}

	select {
	case <-inf.avatarDone:
		t.Fatal("avatar finished before the test released it")
	default:
	}

	history := session.HistorySnapshot()
	if len(history) != 2 || history[1].Role != "assistant" || history[1].Content != "完整回答" {
		t.Fatalf("assistant history was not saved immediately: %+v", history)
	}

	close(inf.avatarRelease)
	close(inf.outputs)
	close(inf.errs)
	session.WaitPipelineDone(2 * time.Second)
}

func TestVoicePipelineBroadcastsPersonaTaskEvent(t *testing.T) {
	orch, session, _, inf := newVoiceRecordingHarness(t)
	orch.wsHub = ws.NewHub()
	taskStore, err := agenttask.OpenStore(filepath.Join(t.TempDir(), "tasks.db"), filepath.Join(t.TempDir(), "artifacts"))
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer taskStore.Close()
	taskSvc := agenttask.NewService(taskStore, nil)
	taskHandlerCalled := make(chan struct{}, 1)
	taskSvc.SetEventHandler(func(*agenttask.Task, *agenttask.Event) {
		taskHandlerCalled <- struct{}{}
	})
	orch.SetTaskService(taskSvc)
	client := &ws.Client{SessionID: session.ID, Send: make(chan []byte, 16)}
	orch.wsHub.Register(client)

	if err := orch.HandleAudioStream(context.Background(), session.ID, make(chan []byte)); err != nil {
		t.Fatal(err)
	}
	<-inf.started

	inf.outputs <- &pb.VoiceLLMOutput{
		TaskEventJson: `{"task_id":"task-1","seq":1,"event_type":"task.queued","status":"queued","message":"任务已加入队列。","progress":0,"task":{"id":"task-1","session_id":"session-recording","title":"知乎热榜","user_request":"知乎热榜","status":"queued","progress":0}}`,
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		TaskEventJson: `{"task_id":"task-1","seq":2,"event_type":"task.started","status":"running","message":"后台任务已启动。","progress":5,"task":{"id":"task-1","session_id":"session-recording","title":"知乎热榜","user_request":"知乎热榜","status":"running","progress":5}}`,
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		TaskEventJson: `{"task_id":"task-1","seq":3,"event_type":"artifact.created","status":"running","message":"已生成一份资料：知乎热榜","progress":90,"task":{"id":"task-1","session_id":"session-recording","title":"知乎热榜","user_request":"知乎热榜","status":"running","progress":90},"payload":{"artifact_id":"artifact-1","title":"知乎热榜","type":"html","mime_type":"text/html; charset=utf-8","content":"<!doctype html><html><body>ok</body></html>"}}`,
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		TaskEventJson: `{"task_id":"task-1","seq":4,"event_type":"task.completed","status":"completed","message":"资料已生成。","progress":100,"task":{"id":"task-1","session_id":"session-recording","title":"知乎热榜","user_request":"知乎热榜","status":"completed","progress":100},"payload":{"artifact_id":"artifact-1"}}`,
	}
	close(inf.outputs)
	close(inf.errs)

	var taskEvents []map[string]any
	deadline := time.After(2 * time.Second)
	for len(taskEvents) < 4 {
		select {
		case data := <-client.Send:
			var msg map[string]any
			if err := json.Unmarshal(data, &msg); err != nil {
				t.Fatalf("unmarshal websocket message: %v", err)
			}
			if msg["type"] == "task_event" {
				taskEvents = append(taskEvents, msg)
			}
		case <-deadline:
			t.Fatal("timed out waiting for task_event websocket message")
		}
	}

	completed := taskEvents[len(taskEvents)-1]
	if completed["session_id"] != session.ID {
		t.Fatalf("unexpected session_id: %+v", completed)
	}
	if completed["task_id"] != "task-1" || completed["event_type"] != "task.completed" {
		t.Fatalf("unexpected task event payload: %+v", completed)
	}
	artifactPayload, ok := taskEvents[2]["payload"].(map[string]any)
	if !ok || artifactPayload["content"] != nil || artifactPayload["artifact_id"] != "artifact-1" {
		t.Fatalf("artifact event should broadcast persisted artifact metadata without content: %+v", taskEvents[2])
	}

	tasks, err := taskStore.ListSessionTasks(context.Background(), session.ID, 10)
	if err != nil {
		t.Fatalf("ListSessionTasks: %v", err)
	}
	if len(tasks) != 1 || tasks[0].ID != "task-1" || tasks[0].Status != agenttask.StatusCompleted {
		t.Fatalf("unexpected persisted task: %+v", tasks)
	}
	events, err := taskStore.ListEventsAfter(context.Background(), "task-1", 0, 10)
	if err != nil {
		t.Fatalf("ListEventsAfter: %v", err)
	}
	if len(events) != 4 || events[0].Seq != 1 || events[3].EventType != "task.completed" {
		t.Fatalf("unexpected persisted events: %+v", events)
	}
	artifact, content, err := taskStore.GetArtifact(context.Background(), "task-1", "artifact-1")
	if err != nil {
		t.Fatalf("GetArtifact: %v", err)
	}
	if artifact.MimeType != "text/html; charset=utf-8" || string(content) != "<!doctype html><html><body>ok</body></html>" {
		t.Fatalf("unexpected persisted artifact: artifact=%+v content=%q", artifact, content)
	}
	select {
	case <-taskHandlerCalled:
		t.Fatal("persona task projection should not invoke task service event handler")
	default:
	}

	session.WaitPipelineDone(2 * time.Second)
	orch.wsHub.Unregister(client)
}

func TestVoiceTurnDropsUnkeyedStaleFinalAfterBargeIn(t *testing.T) {
	orch, session, charStore, inf := newVoiceRecordingHarness(t)

	if err := orch.HandleAudioStream(context.Background(), session.ID, make(chan []byte)); err != nil {
		t.Fatal(err)
	}
	<-inf.started

	inf.outputs <- &pb.VoiceLLMOutput{UserTranscript: "第一问", QuestionId: "q1", ReplyId: "r1"}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "上一轮未完成", QuestionId: "q1", ReplyId: "r1"}
	inf.outputs <- &pb.VoiceLLMOutput{BargeIn: true, QuestionId: "q2", ReplyId: "r2"}
	inf.outputs <- &pb.VoiceLLMOutput{UserTranscript: "第二问", QuestionId: "q2", ReplyId: "r2"}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "上一轮迟到完成文本", IsFinal: true}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "第二轮回答", IsFinal: true, QuestionId: "q2", ReplyId: "r2"}
	close(inf.outputs)
	close(inf.errs)
	session.WaitPipelineDone(2 * time.Second)

	history := session.HistorySnapshot()
	if len(history) != 3 {
		t.Fatalf("expected two users and one assistant, got %+v", history)
	}
	if history[2].Role != "assistant" || history[2].Content != "第二轮回答" {
		t.Fatalf("stale final was mixed into the new turn: %+v", history)
	}

	sessionDir := charStore.SessionRecordingDir(session.CharacterID, session.ID, session.CreatedAt)
	if _, err := os.Stat(filepath.Join(sessionDir, "turn1.txt")); !os.IsNotExist(err) {
		t.Fatalf("expected aborted turn1 transcript to be absent, stat err=%v", err)
	}
	if got := string(waitForFile(t, filepath.Join(sessionDir, "turn2.txt"))); got != "第二轮回答" {
		t.Fatalf("unexpected turn2 transcript: %q", got)
	}
}

func TestVoiceTurnRecordsKeyedStaleFinalToOriginalTurnAfterBargeIn(t *testing.T) {
	orch, session, charStore, inf := newVoiceRecordingHarness(t)

	if err := orch.HandleAudioStream(context.Background(), session.ID, make(chan []byte)); err != nil {
		t.Fatal(err)
	}
	<-inf.started

	inf.outputs <- &pb.VoiceLLMOutput{UserTranscript: "第一问", QuestionId: "q1", ReplyId: "r1"}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "上一轮未完成", QuestionId: "q1", ReplyId: "r1"}
	inf.outputs <- &pb.VoiceLLMOutput{BargeIn: true, QuestionId: "q2", ReplyId: "r2"}
	inf.outputs <- &pb.VoiceLLMOutput{UserTranscript: "第二问", QuestionId: "q2", ReplyId: "r2"}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "第一轮完整回答", IsFinal: true, QuestionId: "q1", ReplyId: "r1"}
	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "第二轮回答", IsFinal: true, QuestionId: "q2", ReplyId: "r2"}
	close(inf.outputs)
	close(inf.errs)
	session.WaitPipelineDone(2 * time.Second)

	sessionDir := charStore.SessionRecordingDir(session.CharacterID, session.ID, session.CreatedAt)
	if got := string(waitForFile(t, filepath.Join(sessionDir, "turn1.txt"))); got != "第一轮完整回答" {
		t.Fatalf("unexpected turn1 transcript: %q", got)
	}
	if got := string(waitForFile(t, filepath.Join(sessionDir, "turn2.txt"))); got != "第二轮回答" {
		t.Fatalf("unexpected turn2 transcript: %q", got)
	}

	history := session.HistorySnapshot()
	if len(history) != 4 {
		t.Fatalf("expected two users and two assistants, got %+v", history)
	}
	if history[1].Content != "第一轮完整回答" || history[3].Content != "第二轮回答" {
		t.Fatalf("assistant turns were mixed: %+v", history)
	}
}

func TestVoiceTextOnlyFinalTriggersAssistantSpeech(t *testing.T) {
	orch, session, _, inf := newVoiceRecordingHarness(t)

	if err := orch.HandleTextInput(context.Background(), session.ID, "用文字问一个问题"); err != nil {
		t.Fatal(err)
	}
	<-inf.started

	inf.outputs <- &pb.VoiceLLMOutput{Transcript: "这是文字模式回答", IsFinal: true, QuestionId: "q1", ReplyId: "r1"}

	select {
	case got := <-inf.ttsTexts:
		if got != "这是文字模式回答" {
			t.Fatalf("unexpected TTS text: %q", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for text-only PersonaAgent response to reach TTS")
	}

	select {
	case <-inf.avatarStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for text-only PersonaAgent response to start avatar")
	}

	close(inf.avatarRelease)
	close(inf.outputs)
	close(inf.errs)
	session.WaitPipelineDone(2 * time.Second)
}
