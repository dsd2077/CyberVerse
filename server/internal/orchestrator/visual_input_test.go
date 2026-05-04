package orchestrator

import (
	"encoding/base64"
	"errors"
	"testing"
	"time"

	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/ws"
)

func newVisualInputTestOrchestrator(t *testing.T, mode PipelineMode) (*Orchestrator, *Session) {
	t.Helper()
	mgr := NewSessionManager(4)
	session, err := mgr.Create("session-visual", mode, "")
	if err != nil {
		t.Fatal(err)
	}
	enabled := true
	orch := New(
		&idleVideoInferenceStub{},
		ws.NewHub(),
		mgr,
		nil,
		nil,
		config.PipelineConfig{
			VisualInput: config.VisualInputConfig{
				Enabled:         &enabled,
				FrameIntervalMS: 1000,
				MaxWidth:        1280,
				MaxHeight:       720,
				MaxFrameBytes:   1024,
				MaxRecentFrames: 2,
				FrameTTLMS:      10000,
			},
		},
	)
	return orch, session
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

func TestHandleVisualFrameRejectsVoiceLLMSession(t *testing.T) {
	orch, session := newVisualInputTestOrchestrator(t, ModeVoiceLLM)

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
