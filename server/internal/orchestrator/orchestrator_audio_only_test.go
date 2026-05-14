package orchestrator

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/mediapeer"
	pb "github.com/cyberverse/server/internal/pb"
	"github.com/cyberverse/server/internal/ws"
)

type audioOnlyPeer struct {
	audioCh chan []byte
	avCh    chan *mediapeer.RawAVSegment
	mu      sync.Mutex
}

func newAudioOnlyPeer() *audioOnlyPeer {
	return &audioOnlyPeer{
		audioCh: make(chan []byte, 4),
		avCh:    make(chan *mediapeer.RawAVSegment, 4),
	}
}

func (p *audioOnlyPeer) Connect(context.Context) error   { return nil }
func (p *audioOnlyPeer) StartAVPipeline(context.Context) {}
func (p *audioOnlyPeer) AdvancePlaybackEpoch(uint64)     {}
func (p *audioOnlyPeer) WaitAVDrain(time.Duration)       {}
func (p *audioOnlyPeer) StopAVPipeline()                 {}
func (p *audioOnlyPeer) SubscribeUserAudio() <-chan []byte {
	ch := make(chan []byte)
	close(ch)
	return ch
}
func (p *audioOnlyPeer) Disconnect() error { return nil }

func (p *audioOnlyPeer) SendAVSegment(seg *mediapeer.RawAVSegment) error {
	p.avCh <- seg
	return nil
}

func (p *audioOnlyPeer) PublishAudioFrame(pcm []byte, _ int) error {
	copyPCM := append([]byte(nil), pcm...)
	p.audioCh <- copyPCM
	return nil
}

func TestVoiceLLMPipelinePublishesAudioOnlyWhenAvatarDisabled(t *testing.T) {
	root := t.TempDir()
	charStore, err := character.NewStore(root)
	if err != nil {
		t.Fatal(err)
	}
	disabled := false
	inf := newVoiceRecordingInferenceStub()
	mgr := NewSessionManager(4)
	orch := New(
		inf,
		ws.NewHub(),
		mgr,
		nil,
		charStore,
		config.PipelineConfig{AvatarEnabled: &disabled},
	)
	session, err := mgr.Create("session-audio-only", ModeOmni, "")
	if err != nil {
		t.Fatal(err)
	}
	peer := newAudioOnlyPeer()
	orch.mu.Lock()
	orch.peers[session.ID] = peer
	orch.mu.Unlock()

	inputCh := make(chan inference.VoiceLLMInputEvent)
	close(inputCh)
	pipelineSeq := session.MarkPipelineRunning()
	go orch.runVoiceLLMPipelineWithConfig(
		context.Background(),
		session,
		session.ID,
		inputCh,
		pipelineSeq,
		0,
		inference.VoiceLLMSessionConfig{SessionID: session.ID},
		false,
	)

	select {
	case <-inf.started:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for voice stream start")
	}

	pcm := make([]byte, 640)
	inf.outputs <- &pb.VoiceLLMOutput{
		UserTranscript: "hello",
		QuestionId:     "q1",
		ReplyId:        "r1",
	}
	inf.outputs <- &pb.VoiceLLMOutput{
		Audio:      &pb.AudioChunk{Data: pcm, SampleRate: 16000, Channels: 1, Format: "pcm_s16le"},
		Transcript: "hi",
		IsFinal:    true,
		QuestionId: "q1",
		ReplyId:    "r1",
	}
	close(inf.outputs)
	close(inf.errs)

	select {
	case got := <-peer.audioCh:
		if len(got) != len(pcm) {
			t.Fatalf("expected %d audio bytes, got %d", len(pcm), len(got))
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for audio-only publish")
	}
	select {
	case <-inf.avatarStarted:
		t.Fatal("GenerateAvatarStream should not be called when avatar is disabled")
	default:
	}
	select {
	case seg := <-peer.avCh:
		t.Fatalf("expected no AV segment when avatar is disabled, got %+v", seg)
	default:
	}

	session.WaitPipelineDone(time.Second)
	if got := session.GetState(); got != StateListening {
		t.Fatalf("expected session to return to listening, got %s", got)
	}
	history := session.HistorySnapshot()
	if len(history) != 2 || history[0].Role != "user" || history[1].Role != "assistant" {
		t.Fatalf("expected user and assistant messages to be saved, got %+v", history)
	}
}
