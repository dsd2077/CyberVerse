// Package mediapeer defines the common interface for media transport backends.
// Both LiveKit Bot and DirectPeer (pion/webrtc P2P) implement this interface,
// allowing the orchestrator to be transport-agnostic.
package mediapeer

import (
	"context"
	"time"
)

// MediaPeer is the common interface for media transport backends.
type MediaPeer interface {
	// Connect establishes the media connection.
	Connect(ctx context.Context) error

	// StartAVPipeline launches encode and publish goroutines.
	StartAVPipeline(ctx context.Context)

	// SendAVSegment enqueues a raw AV segment for encode+publish.
	SendAVSegment(seg *RawAVSegment) error

	// AdvancePlaybackEpoch invalidates any older queued or in-flight AV segments.
	AdvancePlaybackEpoch(epoch uint64)

	// WaitAVDrain blocks until all queued segments are published.
	WaitAVDrain(timeout time.Duration)

	// StopAVPipeline shuts down the AV pipeline goroutines.
	StopAVPipeline()

	// PublishAudioFrame publishes raw PCM audio (for TTS in standard pipeline).
	PublishAudioFrame(pcm []byte, sampleRate int) error

	// SubscribeUserAudio returns a channel receiving decoded PCM from the user's mic.
	SubscribeUserAudio() <-chan []byte

	// Disconnect tears down the connection and releases resources.
	Disconnect() error
}
