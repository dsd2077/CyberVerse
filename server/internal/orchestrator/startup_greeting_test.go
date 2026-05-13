package orchestrator

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/inference"
	pb "github.com/cyberverse/server/internal/pb"
)

type startupGreetingInferenceStub struct {
	mu      sync.Mutex
	started chan struct{}
	calls   int
	configs []inference.VoiceLLMSessionConfig
	inputs  []inference.VoiceLLMInputEvent
}

func newStartupGreetingInferenceStub() *startupGreetingInferenceStub {
	return &startupGreetingInferenceStub{started: make(chan struct{})}
}

func (f *startupGreetingInferenceStub) HealthCheck(context.Context) error { return nil }

func (f *startupGreetingInferenceStub) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	return nil, nil
}

func (f *startupGreetingInferenceStub) SetAvatar(context.Context, string, []byte, string) error {
	return nil
}

func (f *startupGreetingInferenceStub) GenerateAvatarStream(context.Context, <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}

func (f *startupGreetingInferenceStub) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}

func (f *startupGreetingInferenceStub) GenerateLLMStream(context.Context, string, []inference.ChatMessage, inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	ch := make(chan *pb.LLMChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *startupGreetingInferenceStub) SynthesizeSpeechStream(context.Context, <-chan string, inference.TTSConfig) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *startupGreetingInferenceStub) TranscribeStream(context.Context, <-chan []byte, inference.ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *startupGreetingInferenceStub) CheckVoice(context.Context, inference.VoiceLLMSessionConfig) (string, error) {
	return "", nil
}

func (f *startupGreetingInferenceStub) ConverseStream(ctx context.Context, inputCh <-chan inference.VoiceLLMInputEvent, config inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	outputCh := make(chan *pb.VoiceLLMOutput, 2)
	errCh := make(chan error, 1)

	f.mu.Lock()
	f.calls++
	f.configs = append(f.configs, config)
	f.mu.Unlock()

	go func() {
		defer close(outputCh)
		defer close(errCh)
		select {
		case input, ok := <-inputCh:
			if ok {
				f.mu.Lock()
				f.inputs = append(f.inputs, input)
				f.mu.Unlock()
			}
		case <-ctx.Done():
			return
		}
		select {
		case <-f.started:
		default:
			close(f.started)
		}
		outputCh <- &pb.VoiceLLMOutput{UserTranscript: "内部启动提示", QuestionId: "q1", ReplyId: "r1"}
		outputCh <- &pb.VoiceLLMOutput{Transcript: "欢迎回来，我们继续聊。", IsFinal: true, QuestionId: "q1", ReplyId: "r1"}
	}()

	return outputCh, errCh
}

func (f *startupGreetingInferenceStub) Interrupt(context.Context, string) error { return nil }
func (f *startupGreetingInferenceStub) Close() error                            { return nil }

func (f *startupGreetingInferenceStub) snapshot() (int, []inference.VoiceLLMSessionConfig, []inference.VoiceLLMInputEvent) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls, append([]inference.VoiceLLMSessionConfig(nil), f.configs...), append([]inference.VoiceLLMInputEvent(nil), f.inputs...)
}

func TestHandleClientMediaReadyStartsOneOmniGreeting(t *testing.T) {
	root := t.TempDir()
	store, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := store.Create(&character.Character{
		Name:           "晴天",
		VoiceProvider:  "qwen_omni",
		WelcomeMessage: "固定欢迎语不应直接播放。",
	})
	if err != nil {
		t.Fatal(err)
	}
	sessionMgr := NewSessionManager(4)
	t.Cleanup(sessionMgr.Stop)
	session, err := sessionMgr.Create("session-greeting", ModeOmni, char.ID)
	if err != nil {
		t.Fatal(err)
	}
	session.SetDialogContext([]DialogContextItem{
		{Role: "user", Text: "上次我们聊了旅行计划。", Timestamp: 1},
		{Role: "assistant", Text: "我帮你列了行李清单。", Timestamp: 2},
	})
	inf := newStartupGreetingInferenceStub()
	orch := New(inf, nil, sessionMgr, nil, store)

	if err := orch.HandleClientMediaReady(context.Background(), session.ID); err != nil {
		t.Fatal(err)
	}
	if err := orch.HandleClientMediaReady(context.Background(), session.ID); err != nil {
		t.Fatal(err)
	}

	select {
	case <-inf.started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for startup greeting")
	}
	session.WaitPipelineDone(2 * time.Second)

	calls, configs, inputs := inf.snapshot()
	if calls != 1 {
		t.Fatalf("expected one greeting call, got %d", calls)
	}
	if len(configs) != 1 || configs[0].Provider != "qwen_omni" {
		t.Fatalf("expected underlying provider qwen_omni, got %+v", configs)
	}
	if configs[0].WelcomeMessage != "" {
		t.Fatalf("expected fixed welcome to be withheld, got %q", configs[0].WelcomeMessage)
	}
	if len(inputs) != 1 || !strings.Contains(inputs[0].Text, "上次我们聊了旅行计划") {
		t.Fatalf("expected internal prompt with history, got %+v", inputs)
	}

	history := session.HistorySnapshot()
	if len(history) != 1 {
		t.Fatalf("expected only assistant greeting in history, got %+v", history)
	}
	if history[0].Role != "assistant" || history[0].Content != "欢迎回来，我们继续聊。" {
		t.Fatalf("unexpected greeting history: %+v", history)
	}
}

func TestHandleClientMediaReadyIgnoresStandardSession(t *testing.T) {
	sessionMgr := NewSessionManager(4)
	t.Cleanup(sessionMgr.Stop)
	session, err := sessionMgr.Create("session-standard", ModeStandard, "")
	if err != nil {
		t.Fatal(err)
	}
	inf := newStartupGreetingInferenceStub()
	orch := New(inf, nil, sessionMgr, nil, nil)

	if err := orch.HandleClientMediaReady(context.Background(), session.ID); err != nil {
		t.Fatal(err)
	}

	calls, _, _ := inf.snapshot()
	if calls != 0 {
		t.Fatalf("expected no greeting call for standard mode, got %d", calls)
	}
	if session.VoiceStartupGreetingStarted {
		t.Fatal("standard session should not mark voice startup greeting as started")
	}
}
