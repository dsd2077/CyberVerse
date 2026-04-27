package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/inference"
	pb "github.com/cyberverse/server/internal/pb"
)

type idleVideoInferenceStub struct {
	avatarInfo     *pb.AvatarInfo
	setAvatarCalls int
	generateCalls  int
}

func (f *idleVideoInferenceStub) HealthCheck(context.Context) error { return nil }

func (f *idleVideoInferenceStub) AvatarInfo(context.Context) (*pb.AvatarInfo, error) {
	return f.avatarInfo, nil
}

func (f *idleVideoInferenceStub) SetAvatar(context.Context, string, []byte, string) error {
	f.setAvatarCalls++
	return nil
}

func (f *idleVideoInferenceStub) GenerateAvatarStream(context.Context, <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}

func (f *idleVideoInferenceStub) GenerateAvatar(context.Context, []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error) {
	f.generateCalls++
	videoCh := make(chan *pb.VideoChunk)
	errCh := make(chan error)
	close(videoCh)
	close(errCh)
	return videoCh, errCh
}

func (f *idleVideoInferenceStub) GenerateLLMStream(context.Context, string, []inference.ChatMessage, inference.LLMConfig) (<-chan *pb.LLMChunk, <-chan error) {
	ch := make(chan *pb.LLMChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *idleVideoInferenceStub) SynthesizeSpeechStream(context.Context, <-chan string) (<-chan *pb.AudioChunk, <-chan error) {
	ch := make(chan *pb.AudioChunk)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *idleVideoInferenceStub) TranscribeStream(context.Context, <-chan []byte) (<-chan *pb.TranscriptEvent, <-chan error) {
	ch := make(chan *pb.TranscriptEvent)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *idleVideoInferenceStub) CheckVoice(context.Context, inference.VoiceLLMSessionConfig) (string, error) {
	return "", nil
}

func (f *idleVideoInferenceStub) ConverseStream(context.Context, <-chan inference.VoiceLLMInputEvent, inference.VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error) {
	ch := make(chan *pb.VoiceLLMOutput)
	errCh := make(chan error)
	close(ch)
	close(errCh)
	return ch, errCh
}

func (f *idleVideoInferenceStub) Interrupt(context.Context, string) error { return nil }
func (f *idleVideoInferenceStub) Close() error                            { return nil }

func writeTinyPNG(t *testing.T, path string) {
	t.Helper()
	data := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
		0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x03, 0x01, 0x01, 0x00, 0xc9, 0xfe, 0x92,
		0xef, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
		0x44, 0xae, 0x42, 0x60, 0x82,
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func newIdleVideoCharacter(t *testing.T, store *character.Store) (*character.Character, string) {
	t.Helper()

	char, err := store.Create(&character.Character{
		Name:      "Idle",
		VoiceType: "温柔文雅",
	})
	if err != nil {
		t.Fatal(err)
	}

	filename := "img_001.png"
	if err := os.MkdirAll(store.ImagesDir(char.ID), 0755); err != nil {
		t.Fatal(err)
	}
	writeTinyPNG(t, filepath.Join(store.ImagesDir(char.ID), filename))
	if err := store.AddImage(char.ID, character.ImageInfo{
		Filename: filename,
		OrigName: "avatar.png",
		AddedAt:  time.Now().UTC().Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}
	return char, filename
}

func TestEnsureIdleVideoUsesExactResolutionCacheHit(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, filename := newIdleVideoCharacter(t, store)

	inf := &idleVideoInferenceStub{
		avatarInfo: &pb.AvatarInfo{
			ModelName:    "avatar.live_act",
			OutputFps:    24,
			OutputWidth:  320,
			OutputHeight: 480,
		},
	}

	sizeDir := store.IdleVideosForSizeDir(char.ID, filename, 320, 480)
	if err := os.MkdirAll(sizeDir, 0755); err != nil {
		t.Fatal(err)
	}
	wantPath := filepath.Join(sizeDir, "custom_idle.mp4")
	if err := os.WriteFile(wantPath, []byte("cached"), 0644); err != nil {
		t.Fatal(err)
	}

	orch := New(inf, nil, NewSessionManager(4), nil, store)
	gotPath, err := orch.EnsureIdleVideo(context.Background(), char.ID)
	if err != nil {
		t.Fatalf("expected cache hit, got error: %v", err)
	}
	if gotPath != wantPath {
		t.Fatalf("expected cached path %q, got %q", wantPath, gotPath)
	}
	if inf.generateCalls != 0 {
		t.Fatalf("expected no generation on exact-resolution cache hit, got %d calls", inf.generateCalls)
	}
}

func TestEnsureIdleVideoIgnoresDifferentResolutionCache(t *testing.T) {
	store, err := character.NewStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	char, filename := newIdleVideoCharacter(t, store)

	oldPath := store.IdleVideoPath(char.ID, filename, character.DefaultIdleVideoProfile, 512, 512)
	if err := os.MkdirAll(filepath.Dir(oldPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(oldPath, []byte("stale"), 0644); err != nil {
		t.Fatal(err)
	}

	inf := &idleVideoInferenceStub{
		avatarInfo: &pb.AvatarInfo{
			ModelName:    "avatar.live_act",
			OutputFps:    24,
			OutputWidth:  320,
			OutputHeight: 480,
		},
	}
	orch := New(inf, nil, NewSessionManager(4), nil, store)

	if _, err := orch.EnsureIdleVideo(context.Background(), char.ID); err == nil {
		t.Fatal("expected generation attempt because only wrong-resolution cache exists")
	}
	if inf.generateCalls != 1 {
		t.Fatalf("expected one generation attempt, got %d", inf.generateCalls)
	}
}
