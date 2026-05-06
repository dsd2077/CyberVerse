package orchestrator

import (
	"encoding/base64"
	"errors"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/ws"
)

func newVisualInputTestOrchestratorWithConfig(t *testing.T, mode PipelineMode, provider string, visualCfg config.VisualInputConfig) (*Orchestrator, *Session) {
	t.Helper()
	mgr := NewSessionManager(4)
	var charStore *character.Store
	charID := ""
	if provider != "" {
		var err error
		charStore, err = character.NewStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		char, err := charStore.Create(&character.Character{
			Name:          "Visual",
			Mode:          "voice_llm",
			VoiceProvider: provider,
			VoiceType:     "Tina",
		})
		if err != nil {
			t.Fatal(err)
		}
		charID = char.ID
	}
	session, err := mgr.Create("session-visual", mode, charID)
	if err != nil {
		t.Fatal(err)
	}
	orch := New(
		&idleVideoInferenceStub{},
		ws.NewHub(),
		mgr,
		nil,
		charStore,
		config.PipelineConfig{
			VisualInput: visualCfg,
		},
	)
	return orch, session
}

func newVisualInputTestOrchestrator(t *testing.T, mode PipelineMode) (*Orchestrator, *Session) {
	t.Helper()
	enabled := true
	return newVisualInputTestOrchestratorWithConfig(t, mode, "", config.VisualInputConfig{
		Enabled:         &enabled,
		FrameIntervalMS: 1000,
		MaxWidth:        1280,
		MaxHeight:       720,
		MaxFrameBytes:   1024,
		MaxRecentFrames: 2,
		FrameTTLMS:      10000,
	})
}

func newVisualInputTestVoiceOrchestrator(t *testing.T, provider string) (*Orchestrator, *Session) {
	t.Helper()
	enabled := true
	return newVisualInputTestOrchestratorWithConfig(t, ModeVoiceLLM, provider, config.VisualInputConfig{
		Enabled:         &enabled,
		FrameIntervalMS: 1000,
		MaxWidth:        1280,
		MaxHeight:       720,
		MaxFrameBytes:   1024,
		MaxRecentFrames: 2,
		FrameTTLMS:      10000,
	})
}

func TestHandleVisualFrameStoresLatestForStandardSession(t *testing.T) {
	orch, session := newVisualInputTestOrchestrator(t, ModeStandard)

	if err := orch.HandleVisualInputStart(session.ID, "screen"); err != nil {
		t.Fatal(err)
	}
	err := orch.HandleVisualFrame(session.ID, ws.WSMessage{
		Source:      "screen",
		Mime:        "image/jpeg",
		Data:        base64.StdEncoding.EncodeToString([]byte{0xff, 0xd8, 0xff, 0x00}),
		Width:       640,
		Height:      360,
		TimestampMS: 123,
		FrameSeq:    1,
	})
	if err != nil {
		t.Fatal(err)
	}

	frames := session.LatestVisualFrames(time.Now(), time.Second)
	if len(frames) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(frames))
	}
	if frames[0].Source != "screen" || frames[0].MimeType != "image/jpeg" || frames[0].FrameSeq != 1 {
		t.Fatalf("unexpected frame: %+v", frames[0])
	}
}

func TestHandleVisualFrameStoresLatestForQwenOmniVoiceLLMSession(t *testing.T) {
	orch, session := newVisualInputTestVoiceOrchestrator(t, "qwen_omni")

	if err := orch.HandleVisualInputStart(session.ID, "camera"); err != nil {
		t.Fatal(err)
	}
	err := orch.HandleVisualFrame(session.ID, ws.WSMessage{
		Source:      "camera",
		Mime:        "image/jpeg",
		Data:        base64.StdEncoding.EncodeToString([]byte{0xff, 0xd8, 0xff, 0x00}),
		Width:       640,
		Height:      360,
		TimestampMS: 123,
		FrameSeq:    1,
	})
	if err != nil {
		t.Fatal(err)
	}

	frames := session.LatestVisualFrames(time.Now(), time.Second)
	if len(frames) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(frames))
	}
	if frames[0].Source != "camera" || frames[0].MimeType != "image/jpeg" || frames[0].FrameSeq != 1 {
		t.Fatalf("unexpected frame: %+v", frames[0])
	}
}

func TestHandleVisualFrameRejectsDoubaoVoiceLLMSession(t *testing.T) {
	orch, session := newVisualInputTestVoiceOrchestrator(t, "doubao")

	err := orch.HandleVisualFrame(session.ID, ws.WSMessage{
		Source: "camera",
		Mime:   "image/jpeg",
		Data:   base64.StdEncoding.EncodeToString([]byte{0xff, 0xd8, 0xff, 0x00}),
		Width:  640,
		Height: 360,
	})
	if !errors.Is(err, ErrVisualInputUnsupported) {
		t.Fatalf("expected ErrVisualInputUnsupported, got %v", err)
	}
}

func TestQwenOmniVisualInputConfigIsClamped(t *testing.T) {
	enabled := true
	orch, session := newVisualInputTestOrchestratorWithConfig(t, ModeVoiceLLM, "qwen_omni", config.VisualInputConfig{
		Enabled:         &enabled,
		FrameIntervalMS: 250,
		MaxWidth:        1280,
		MaxHeight:       720,
		MaxFrameBytes:   900 * 1024,
		MaxRecentFrames: 2,
		FrameTTLMS:      10000,
	})

	cfg, ok := orch.VisualInputConfigForSession(session)
	if !ok {
		t.Fatal("expected qwen_omni visual input support")
	}
	if cfg.FrameIntervalMS != 1000 {
		t.Fatalf("expected frame interval clamp to 1000ms, got %d", cfg.FrameIntervalMS)
	}
	if cfg.MaxFrameBytes != qwenOmniMaxVisualFrameBytes {
		t.Fatalf("expected max frame bytes clamp to %d, got %d", qwenOmniMaxVisualFrameBytes, cfg.MaxFrameBytes)
	}
}
