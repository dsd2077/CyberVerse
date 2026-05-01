package livekit

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cyberverse/server/internal/mediapeer"
	mediasdk "github.com/livekit/media-sdk"
	protoLogger "github.com/livekit/protocol/logger"
	lksdk "github.com/livekit/server-sdk-go/v2"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/rtcp"
	"github.com/pion/webrtc/v4"
)

// Compile-time check: Bot implements mediapeer.MediaPeer.
var _ mediapeer.MediaPeer = (*Bot)(nil)

// Bot is a server-side participant that joins a LiveKit room to
// publish avatar video/audio and subscribe to user audio.
type Bot struct {
	room       *lksdk.Room
	url        string
	apiKey     string
	apiSecret  string
	roomName   string
	identity   string
	userAudioC chan []byte
	cancel     context.CancelFunc
	mu         sync.Mutex

	publishAudioMu         sync.Mutex
	publishAudioTrack      *lkmedia.PCMLocalTrack
	publishAudioSampleRate int

	publishVideoMu    sync.Mutex
	publishVideoTrack *lksdk.LocalTrack
	publishVideoW     int
	publishVideoH     int

	pcmRemoteTracksMu sync.Mutex
	pcmRemoteTracks   map[*webrtc.TrackRemote]*lkmedia.PCMRemoteTrack

	lastPublishEnd time.Time // [AVGap] 上次发布结束的时间戳

	// AV pipeline
	encodeCh         chan *RawAVSegment
	publishCh        chan *AVSegment
	avPipelineCtx    context.Context
	avPipelineCancel context.CancelFunc
	avPipelineWg     sync.WaitGroup
	playbackEpoch    atomic.Uint64
}

// NewBot creates a new bot for the given room.
func NewBot(url, apiKey, apiSecret, roomName string) *Bot {
	b := &Bot{
		url:             url,
		apiKey:          apiKey,
		apiSecret:       apiSecret,
		roomName:        roomName,
		identity:        "cyberverse-bot",
		userAudioC:      make(chan []byte, 64),
		pcmRemoteTracks: make(map[*webrtc.TrackRemote]*lkmedia.PCMRemoteTrack),
	}
	return b
}

// Connect joins the LiveKit room as a bot participant.
func (b *Bot) Connect(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	ctx, cancel := context.WithCancel(ctx)
	b.cancel = cancel

	// Generate a bot token
	token, err := GenerateToken(b.apiKey, b.apiSecret, b.roomName, b.identity, true)
	if err != nil {
		cancel()
		return fmt.Errorf("failed to generate bot token: %w", err)
	}

	// Create room and set up callbacks
	roomCallback := &lksdk.RoomCallback{
		ParticipantCallback: lksdk.ParticipantCallback{
			OnTrackSubscribed: b.onTrackSubscribed,
			OnTrackUnsubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, _ *lksdk.RemoteParticipant) {
				b.pcmRemoteTracksMu.Lock()
				pcmTrack := b.pcmRemoteTracks[track]
				delete(b.pcmRemoteTracks, track)
				b.pcmRemoteTracksMu.Unlock()
				if pcmTrack != nil {
					pcmTrack.Close()
				}
			},
		},
	}

	room, err := lksdk.ConnectToRoom(b.url, lksdk.ConnectInfo{
		APIKey:              b.apiKey,
		APISecret:           b.apiSecret,
		RoomName:            b.roomName,
		ParticipantIdentity: b.identity,
	}, roomCallback)
	if err != nil {
		cancel()
		return fmt.Errorf("failed to connect bot to room %s: %w", b.roomName, err)
	}

	// Suppress noisy pion WebRTC INFO logs (ICE/TURN probing); keep SDK / connection errors.
	if lkZap, err := protoLogger.NewZapLogger(&protoLogger.Config{
		Level: "info",
		ComponentLevels: map[string]string{
			"pion.pc":   "error",
			"pion.ice":  "error",
			"pion.turn": "error",
		},
	}); err == nil {
		room.SetLogger(lkZap)
	} else {
		log.Printf("LiveKit: failed to init quieter logger (pion logs may be verbose): %v", err)
	}

	b.room = room
	_ = token // token was used for info, ConnectToRoom uses API key/secret directly

	log.Printf("Bot connected to room %s", b.roomName)
	return nil
}

// onTrackSubscribed is called when a remote participant publishes a track.
// We capture user audio tracks.
func (b *Bot) onTrackSubscribed(track *webrtc.TrackRemote, publication *lksdk.RemoteTrackPublication, participant *lksdk.RemoteParticipant) {
	if track.Kind() != webrtc.RTPCodecTypeAudio {
		return
	}

	log.Printf("Bot subscribed to audio track from %s", participant.Identity())

	// Decode opus -> PCM16 and resample to 16kHz mono so Doubao can directly consume it.
	// PCMRemoteTrack will run its own RTP processing loop in background.
	writer := &pcm16ToChannelWriter{ch: b.userAudioC}
	pcmTrack, err := lkmedia.NewPCMRemoteTrack(
		track,
		writer,
		lkmedia.WithTargetSampleRate(16000),
		lkmedia.WithTargetChannels(1),
	)
	if err != nil {
		log.Printf("Failed to create PCMRemoteTrack: %v", err)
		return
	}

	b.pcmRemoteTracksMu.Lock()
	b.pcmRemoteTracks[track] = pcmTrack
	b.pcmRemoteTracksMu.Unlock()
}

type pcm16ToChannelWriter struct {
	ch chan []byte
}

func (w *pcm16ToChannelWriter) WriteSample(sample mediasdk.PCM16Sample) error {
	// Convert PCM16 sample slice into little-endian byte stream.
	out := make([]byte, len(sample)*2)
	for i, s := range sample {
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(s))
	}

	select {
	case w.ch <- out:
	default:
		// Drop if channel full (backpressure)
	}
	return nil
}

func (w *pcm16ToChannelWriter) Close() error {
	return nil
}

// SubscribeUserAudio returns a channel that receives user audio data.
func (b *Bot) SubscribeUserAudio() <-chan []byte {
	return b.userAudioC
}

// PublishVideoFrame publishes one or more raw RGB24 frames (width*height*3 per frame).
// If len(frame) == width*height*3, a single frame at 25fps is encoded; otherwise
// use PublishAvatarAVChunk with explicit numFrames/fps.
func (b *Bot) PublishVideoFrame(frame []byte, width, height int) error {
	if width <= 0 || height <= 0 {
		return nil
	}
	per := width * height * 3
	if per <= 0 || len(frame) < per {
		return nil
	}
	n := len(frame) / per
	return b.PublishAvatarAVChunk(nil, 0, frame[:n*per], width, height, n, 25)
}

// PublishAvatarAVChunk encodes an RGB24 chunk to VP8 and publishes video samples,
// interleaving aligned slices of assistant PCM so audio/video advance together.
// Frames are paced at real-time (one per frameDur) so the WebRTC receiver's
// playout buffer stays full and does not stall between chunks.
func (b *Bot) PublishAvatarAVChunk(pcm []byte, sampleRate int, rgb []byte, width, height, numFrames, fps int) error {
	return b.PublishAvatarAVChunkWithTrace("", pcm, sampleRate, rgb, width, height, numFrames, fps)
}

// PublishAvatarAVChunkWithTrace is the legacy traced entrypoint; trace labels are ignored (logging removed).
func (b *Bot) PublishAvatarAVChunkWithTrace(_ string, pcm []byte, sampleRate int, rgb []byte, width, height, numFrames, fps int) error {
	if width <= 0 || height <= 0 || numFrames <= 0 {
		return nil
	}
	if fps <= 0 {
		fps = 25
	}

	vp8Samples, err := encodeRGBChunkToVP8Samples(rgb, width, height, numFrames, fps)
	if err != nil {
		log.Printf("VP8 encode failed: %v", err)
		return err
	}
	if len(vp8Samples) == 0 {
		return nil
	}

	b.mu.Lock()
	room := b.room
	b.mu.Unlock()
	if room == nil {
		return fmt.Errorf("bot is not connected to a room")
	}

	pcmSlices := audioPCMSlicesForVideoFrames(pcm, len(vp8Samples))
	hasPCM := len(pcm) > 0 && sampleRate > 0

	// Frame duration used for real-time pacing between WriteSample calls.
	frameDur := time.Second / time.Duration(fps)

	// Ensure video track exists before acquiring timed locks.
	b.publishVideoMu.Lock()
	if err := b.ensureVideoTrackLocked(room, width, height); err != nil {
		b.publishVideoMu.Unlock()
		return err
	}
	b.publishVideoMu.Unlock()

	// Pace one frame at a time: acquire locks, write audio+video, release,
	// then sleep for the remainder of frameDur. This avoids holding the audio
	// lock across the entire segment (which would block concurrent PublishAudioFrame
	// calls from the standard pipeline's TTS goroutine).
	for i := range vp8Samples {
		frameStart := time.Now()

		if hasPCM && i < len(pcmSlices) && len(pcmSlices[i]) > 0 {
			b.publishAudioMu.Lock()
			writeErr := b.writePCMLocked(room, pcmSlices[i], sampleRate)
			b.publishAudioMu.Unlock()
			if writeErr != nil {
				return writeErr
			}
		}

		b.publishVideoMu.Lock()
		writeErr := b.publishVideoTrack.WriteSample(vp8Samples[i], nil)
		b.publishVideoMu.Unlock()
		if writeErr != nil {
			return fmt.Errorf("WriteSample video: %w", writeErr)
		}

		// Sleep for the remainder of this frame's slot to pace output at real-time.
		if elapsed := time.Since(frameStart); elapsed < frameDur {
			time.Sleep(frameDur - elapsed)
		}
	}

	b.mu.Lock()
	b.lastPublishEnd = time.Now()
	b.mu.Unlock()

	return nil
}

func (b *Bot) ensureVideoTrackLocked(room *lksdk.Room, width, height int) error {
	if b.publishVideoTrack != nil && b.publishVideoW == width && b.publishVideoH == height {
		return nil
	}
	if b.publishVideoTrack != nil {
		_ = b.publishVideoTrack.Close()
		b.publishVideoTrack = nil
	}

	track, err := lksdk.NewLocalTrack(
		webrtc.RTPCodecCapability{
			MimeType:  webrtc.MimeTypeVP8,
			ClockRate: 90000,
		},
		// All-Intra encoding: PLI/FIR are no-ops (every frame is already a keyframe).
		lksdk.WithRTCPHandler(func(_ rtcp.Packet) {}),
	)
	if err != nil {
		return fmt.Errorf("NewLocalTrack VP8: %w", err)
	}

	bindStart := time.Now()
	track.OnBind(func() {})

	if _, err := room.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{
		Name:        "assistant-video",
		VideoWidth:  width,
		VideoHeight: height,
		Stream:      "assistant",
	}); err != nil {
		_ = track.Close()
		return fmt.Errorf("PublishTrack video: %w", err)
	}
	b.publishVideoTrack = track
	b.publishVideoW = width
	b.publishVideoH = height

	// LocalTrack.WriteSample is a no-op until WebRTC Bind() sets packetizer; without
	// waiting, all VP8 samples are silently dropped while audio (PCMLocalTrack) still works.
	deadline := time.Now().Add(10 * time.Second)
	for !track.IsBound() && time.Now().Before(deadline) {
		time.Sleep(30 * time.Millisecond)
	}
	elapsed := time.Since(bindStart)
	if !track.IsBound() {
		log.Printf("assistant-video: track not bound after 10s (elapsed=%.3fs); VP8 WriteSample may be dropped", elapsed.Seconds())
	}
	return nil
}

// writePCMLocked requires publishAudioMu held.
func (b *Bot) writePCMLocked(room *lksdk.Room, pcm []byte, sampleRate int) error {
	if len(pcm) == 0 {
		return nil
	}
	if b.publishAudioTrack == nil || b.publishAudioSampleRate != sampleRate {
		track, err := lkmedia.NewPCMLocalTrack(sampleRate, 1, protoLogger.GetLogger())
		if err != nil {
			return fmt.Errorf("create PCMLocalTrack failed: %w", err)
		}
		if _, err := room.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{
			Name:   "assistant-audio",
			Stream: "assistant",
		}); err != nil {
			_ = track.Close()
			return fmt.Errorf("PublishTrack audio: %w", err)
		}
		b.publishAudioTrack = track
		b.publishAudioSampleRate = sampleRate
	}
	sampleCount := len(pcm) / 2
	samples := make(mediasdk.PCM16Sample, sampleCount)
	for i := 0; i < sampleCount; i++ {
		u := binary.LittleEndian.Uint16(pcm[i*2 : i*2+2])
		samples[i] = int16(u)
	}
	return b.publishAudioTrack.WriteSample(samples)
}

// PublishAudioFrame publishes raw PCM audio data to the room.
func (b *Bot) PublishAudioFrame(pcm []byte, sampleRate int) error {
	if len(pcm) == 0 {
		return nil
	}
	if sampleRate <= 0 {
		return fmt.Errorf("invalid sampleRate=%d", sampleRate)
	}

	b.mu.Lock()
	room := b.room
	b.mu.Unlock()
	if room == nil {
		return fmt.Errorf("bot is not connected to a room")
	}

	b.publishAudioMu.Lock()
	defer b.publishAudioMu.Unlock()

	return b.writePCMLocked(room, pcm, sampleRate)
}

// ── AV Pipeline ────────────────────────────────────────────────────────────

// StartAVPipeline launches the encode and publish goroutines.
// Channel capacities are kept small (1+1) so the pipeline applies back-pressure
// to the producer, preventing segments from piling up and causing AV desync.
// This still allows overlapping encode(N+1) with publish(N).
func (b *Bot) StartAVPipeline(ctx context.Context) {
	b.avPipelineCtx, b.avPipelineCancel = context.WithCancel(ctx)
	b.encodeCh = make(chan *RawAVSegment, 1)
	b.publishCh = make(chan *AVSegment, 1)

	b.avPipelineWg.Add(2)
	go b.runEncoder()
	go b.runPublisher()
}

// StopAVPipeline gracefully shuts down the pipeline and waits for goroutines.
func (b *Bot) StopAVPipeline() {
	if b.avPipelineCancel != nil {
		b.avPipelineCancel()
	}
	if b.encodeCh != nil {
		close(b.encodeCh)
	}
	b.avPipelineWg.Wait()
}

// SendAVSegment enqueues a raw segment for encoding and publishing.
// With small channel buffers this will block when the pipeline is saturated,
// which is intentional — it prevents the producer from racing ahead and
// causing audio-video desync.
func (b *Bot) SendAVSegment(seg *RawAVSegment) error {
	seg.QueuedAt = time.Now()
	if b.isPlaybackStale(seg.Epoch) {
		return nil
	}
	select {
	case b.encodeCh <- seg:
		return nil
	case <-b.avPipelineCtx.Done():
		return b.avPipelineCtx.Err()
	}
}

func (b *Bot) AdvancePlaybackEpoch(epoch uint64) {
	if epoch == 0 {
		return
	}
	for {
		current := b.playbackEpoch.Load()
		if epoch <= current {
			return
		}
		if b.playbackEpoch.CompareAndSwap(current, epoch) {
			return
		}
	}
}

// WaitAVDrain sends a fence marker through the AV pipeline and blocks until
// all preceding segments have been fully published (including real-time pacing).
// This ensures the client receives every frame before the caller proceeds
// (e.g. switching avatar status from speaking to idle).
func (b *Bot) WaitAVDrain(timeout time.Duration) {
	if b.encodeCh == nil {
		return
	}
	fence := make(chan struct{})
	select {
	case b.encodeCh <- &RawAVSegment{Fence: fence}:
	case <-b.avPipelineCtx.Done():
		return
	case <-time.After(timeout):
		return
	}
	select {
	case <-fence:
	case <-b.avPipelineCtx.Done():
	case <-time.After(timeout):
	}
}

// runEncoder reads raw segments from encodeCh, encodes to VP8, and sends to publishCh.
func (b *Bot) runEncoder() {
	defer b.avPipelineWg.Done()
	defer close(b.publishCh)

	for raw := range b.encodeCh {
		if b.avPipelineCtx.Err() != nil {
			return
		}

		// Fence marker: pass through to publisher without encoding.
		if raw.Fence != nil && len(raw.RGB) == 0 {
			select {
			case b.publishCh <- &AVSegment{Fence: raw.Fence}:
			case <-b.avPipelineCtx.Done():
				return
			}
			continue
		}
		if b.isPlaybackStale(raw.Epoch) {
			continue
		}

		vp8Samples, err := encodeRGBChunkToVP8Samples(raw.RGB, raw.Width, raw.Height, raw.NumFrames, raw.FPS)
		if err != nil {
			log.Printf("[AVPipeline] encode failed: %v", err)
			continue
		}
		if len(vp8Samples) == 0 {
			continue
		}
		seg := &AVSegment{
			TraceLabel: raw.TraceLabel,
			Epoch:      raw.Epoch,
			VP8Samples: vp8Samples,
			PCM:        raw.PCM,
			SampleRate: raw.SampleRate,
			Width:      raw.Width,
			Height:     raw.Height,
			FPS:        raw.FPS,
			NumFrames:  raw.NumFrames,
			QueuedAt:   raw.QueuedAt,
		}
		select {
		case b.publishCh <- seg:
		case <-b.avPipelineCtx.Done():
			return
		}
	}
}

// runPublisher reads encoded segments from publishCh and publishes with real-time pacing.
func (b *Bot) runPublisher() {
	defer b.avPipelineWg.Done()

	for seg := range b.publishCh {
		if b.avPipelineCtx.Err() != nil {
			return
		}
		// Fence marker: signal drain completion without publishing.
		if seg.Fence != nil && len(seg.VP8Samples) == 0 {
			close(seg.Fence)
			continue
		}
		if b.isPlaybackStale(seg.Epoch) {
			continue
		}
		b.publishAVSegment(seg)
	}
}

// publishAVSegment publishes pre-encoded VP8 samples with real-time pacing.
// Extracted from PublishAvatarAVChunkWithTrace.
func (b *Bot) publishAVSegment(seg *AVSegment) {
	b.mu.Lock()
	room := b.room
	b.mu.Unlock()
	if room == nil {
		log.Printf("[AVPipeline] publish skipped: bot not connected")
		return
	}

	fps := seg.FPS
	if fps <= 0 {
		fps = 25
	}
	frameDur := time.Second / time.Duration(fps)

	// Ensure video track exists.
	b.publishVideoMu.Lock()
	if err := b.ensureVideoTrackLocked(room, seg.Width, seg.Height); err != nil {
		b.publishVideoMu.Unlock()
		log.Printf("[AVPipeline] ensureVideoTrack failed: %v", err)
		return
	}
	b.publishVideoMu.Unlock()

	pcmSlices := audioPCMSlicesForVideoFrames(seg.PCM, len(seg.VP8Samples))
	hasPCM := len(seg.PCM) > 0 && seg.SampleRate > 0

	for i := range seg.VP8Samples {
		if b.isPlaybackStale(seg.Epoch) {
			return
		}
		frameStart := time.Now()

		if hasPCM && i < len(pcmSlices) && len(pcmSlices[i]) > 0 {
			b.publishAudioMu.Lock()
			if err := b.writePCMLocked(room, pcmSlices[i], seg.SampleRate); err != nil {
				b.publishAudioMu.Unlock()
				log.Printf("[AVPipeline] audio write error: %v", err)
				return
			}
			b.publishAudioMu.Unlock()
		}

		b.publishVideoMu.Lock()
		if err := b.publishVideoTrack.WriteSample(seg.VP8Samples[i], nil); err != nil {
			b.publishVideoMu.Unlock()
			log.Printf("[AVPipeline] video write error: %v", err)
			return
		}
		b.publishVideoMu.Unlock()

		if elapsed := time.Since(frameStart); elapsed < frameDur {
			time.Sleep(frameDur - elapsed)
		}
	}

	b.mu.Lock()
	b.lastPublishEnd = time.Now()
	b.mu.Unlock()
}

func (b *Bot) isPlaybackStale(epoch uint64) bool {
	if epoch == 0 {
		return false
	}
	current := b.playbackEpoch.Load()
	return current > 0 && epoch < current
}

// Disconnect leaves the room and cleans up.
func (b *Bot) Disconnect() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.cancel != nil {
		b.cancel()
	}

	b.publishVideoMu.Lock()
	if b.publishVideoTrack != nil {
		_ = b.publishVideoTrack.Close()
		b.publishVideoTrack = nil
	}
	b.publishVideoMu.Unlock()

	if b.room != nil {
		b.room.Disconnect()
		b.room = nil
	}

	// Don't close userAudioC here -- readers may still be draining
	close(b.userAudioC)
	log.Printf("Bot disconnected from room %s", b.roomName)
	return nil
}

// WaitForParticipant blocks until a non-bot participant joins or ctx is cancelled.
func (b *Bot) WaitForParticipant(ctx context.Context) error {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if b.room == nil {
				continue
			}
			participants := b.room.GetRemoteParticipants()
			if len(participants) > 0 {
				return nil
			}
		}
	}
}
