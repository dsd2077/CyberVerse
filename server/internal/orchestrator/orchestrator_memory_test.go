package orchestrator

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/inference"
	pb "github.com/cyberverse/server/internal/pb"
	"github.com/cyberverse/server/internal/ws"
)

type standardMemoryInferenceStub struct {
	messages chan []inference.ChatMessage
}

func (f *standardMemoryInferenceStub) HealthCheck(context.Context) error { return nil }
func (f *standardMemoryInferenceStub) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	return nil, nil
}
func (f *standardMemoryInferenceStub) SetAvatar(context.Context, string, []byte, string) error {
	return nil
}
func (f *standardMemoryInferenceStub) GenerateAvatarStream(context.Context, <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *standardMemoryInferenceStub) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}
func (f *standardMemoryInferenceStub) GenerateLLMStream(_ context.Context, _ string, messages []inference.ChatMessage, _ inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	f.messages <- append([]inference.ChatMessage(nil), messages...)
	ch := make(chan *pb.LLMChunk, 1)
	errCh := make(chan error)
	ch <- &pb.LLMChunk{Token: "记住了", AccumulatedText: "记住了", IsFinal: true}
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *standardMemoryInferenceStub) SynthesizeSpeechStream(context.Context, <-chan string, inference.TTSConfig) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *standardMemoryInferenceStub) TranscribeStream(context.Context, <-chan []byte, inference.ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *standardMemoryInferenceStub) CheckVoice(context.Context, inference.VoiceLLMSessionConfig) (string, error) {
	return "", nil
}
func (f *standardMemoryInferenceStub) ConverseStream(context.Context, <-chan inference.VoiceLLMInputEvent, inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	ch := make(chan *pb.VoiceLLMOutput)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}
func (f *standardMemoryInferenceStub) Interrupt(context.Context, string) error { return nil }
func (f *standardMemoryInferenceStub) Close() error                            { return nil }

type fakeConversationMemory struct {
	recalls  []string
	retains  []string
	retainCh chan string
}

func (f *fakeConversationMemory) Recall(_ context.Context, query string) ([]string, error) {
	f.recalls = append(f.recalls, query)
	return []string{"用户喜欢用 Pixi 管理环境。"}, nil
}

func (f *fakeConversationMemory) Retain(_ context.Context, content string, _ string) error {
	f.retains = append(f.retains, content)
	if f.retainCh != nil {
		f.retainCh <- content
	}
	return nil
}

func TestStandardPipelineRecallsAndRetainsMemory(t *testing.T) {
	root := t.TempDir()
	charStore, err := character.NewStore(filepath.Join(root, "characters"))
	if err != nil {
		t.Fatal(err)
	}
	char, err := charStore.Create(&character.Character{Name: "Standard Memory", VoiceType: "Ethan"})
	if err != nil {
		t.Fatal(err)
	}
	sessionMgr := NewSessionManager(4)
	t.Cleanup(sessionMgr.Stop)
	session, err := sessionMgr.Create("session-standard-memory", ModeStandard, char.ID)
	if err != nil {
		t.Fatal(err)
	}

	inf := &standardMemoryInferenceStub{messages: make(chan []inference.ChatMessage, 1)}
	memory := &fakeConversationMemory{retainCh: make(chan string, 1)}
	orch := New(inf, ws.NewHub(), sessionMgr, nil, charStore)
	orch.memoryClient = memory

	if err := orch.HandleTextInput(context.Background(), session.ID, "我喜欢什么环境管理工具？"); err != nil {
		t.Fatal(err)
	}

	var messages []inference.ChatMessage
	select {
	case messages = <-inf.messages:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for standard LLM messages")
	}
	session.WaitPipelineDone(2 * time.Second)

	var systemText string
	for _, message := range messages {
		if message.Role == "system" {
			systemText += "\n" + message.Content
		}
	}
	if !strings.Contains(systemText, "相关长期记忆") || !strings.Contains(systemText, "用户喜欢用 Pixi 管理环境。") {
		t.Fatalf("standard prompt did not include recalled memory: %q", systemText)
	}
	if len(memory.recalls) != 1 || memory.recalls[0] != "我喜欢什么环境管理工具？" {
		t.Fatalf("unexpected memory recalls: %+v", memory.recalls)
	}

	var retained string
	select {
	case retained = <-memory.retainCh:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for retained memory, got %+v", memory.retains)
	}
	if !strings.Contains(retained, "用户: 我喜欢什么环境管理工具？") || !strings.Contains(retained, "助手: 记住了") {
		t.Fatalf("unexpected retained memory content: %q", retained)
	}
}
