// Package direct implements a P2P WebRTC media peer using pion/webrtc.
// It replaces LiveKit for 1:1 scenarios, eliminating the external SFU dependency.
// Signaling (offer/answer/ICE) flows through the existing WebSocket hub.
package direct

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cyberverse/server/internal/mediapeer"
	"github.com/pion/interceptor/pkg/cc"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	opus "gopkg.in/hraban/opus.v2"
)

// Compile-time check: DirectPeer implements mediapeer.MediaPeer.
var _ mediapeer.MediaPeer = (*DirectPeer)(nil)

// SignalingFunc sends a signaling message to the browser via the WebSocket hub.
type SignalingFunc func(sessionID string, msg map[string]any)

// DirectPeer is a P2P WebRTC media peer using pion/webrtc directly.
type DirectPeer struct {
	sessionID   string
	signalingFn SignalingFunc
	iceServers  []webrtc.ICEServer
	webrtcAPI   *webrtc.API
	estimatorCh <-chan cc.BandwidthEstimator

	pc         *webrtc.PeerConnection
	videoTrack *webrtc.TrackLocalStaticSample
	audioTrack *webrtc.TrackLocalStaticSample

	userAudioCh chan []byte

	// Opus encoder for outgoing PCM → Opus
	opusMu        sync.Mutex
	opusEncoder   *opus.Encoder
	opusEncoderSR int

	// Connection state
	connected chan struct{}
	mu        sync.Mutex

	// AV pipeline (same pattern as Bot)
	encodeCh         chan *mediapeer.RawAVSegment
	publishCh        chan *mediapeer.AVSegment
	avPipelineCtx    context.Context
	avPipelineCancel context.CancelFunc
	avPipelineWg     sync.WaitGroup

	// RTP timestamp gap correction: tracks the wall-clock time of the
	// last WriteSample call so the next segment's first sample can carry
	// a Duration that advances the RTP timestamp over the idle gap.
	lastVideoWriteTime time.Time
	lastAudioWriteTime time.Time
	playbackEpoch      atomic.Uint64
}

// NewDirectPeer creates a new P2P WebRTC peer for the given session.
// api should be created via NewWebRTCAPI (with interceptors); estimatorCh receives the GCC bandwidth estimator.
func NewDirectPeer(sessionID string, signalingFn SignalingFunc, iceServers []webrtc.ICEServer,
	api *webrtc.API, estimatorCh <-chan cc.BandwidthEstimator) *DirectPeer {
	return &DirectPeer{
		sessionID:   sessionID,
		signalingFn: signalingFn,
		iceServers:  iceServers,
		webrtcAPI:   api,
		estimatorCh: estimatorCh,
		userAudioCh: make(chan []byte, 64),
		connected:   make(chan struct{}),
	}
}

// Connect creates the PeerConnection and local tracks.
// It does NOT start negotiation — call StartNegotiation() after
// the browser signals readiness via "webrtc_ready".
func (p *DirectPeer) Connect(ctx context.Context) error {
	config := webrtc.Configuration{
		ICEServers: p.iceServers,
	}

	var pc *webrtc.PeerConnection
	var err error
	if p.webrtcAPI != nil {
		pc, err = p.webrtcAPI.NewPeerConnection(config)
	} else {
		pc, err = webrtc.NewPeerConnection(config)
	}
	if err != nil {
		return fmt.Errorf("create PeerConnection: %w", err)
	}

	// Create VP8 video track
	videoTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8},
		"assistant-video",
		"cyberverse",
	)
	if err != nil {
		pc.Close()
		return fmt.Errorf("create video track: %w", err)
	}
	videoSender, err := pc.AddTrack(videoTrack)
	if err != nil {
		pc.Close()
		return fmt.Errorf("add video track: %w", err)
	}

	// Create Opus audio track
	audioTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus},
		"assistant-audio",
		"cyberverse",
	)
	if err != nil {
		pc.Close()
		return fmt.Errorf("create audio track: %w", err)
	}
	audioSender, err := pc.AddTrack(audioTrack)
	if err != nil {
		pc.Close()
		return fmt.Errorf("add audio track: %w", err)
	}

	// RTCP reader goroutines — MUST read continuously so that
	// NACK/TWCC/GCC interceptors receive browser feedback and work correctly.
	go readRTCP(videoSender)
	go readRTCP(audioSender)

	// Add a recvonly audio transceiver so the SDP offer explicitly requests
	// the browser's microphone audio. Without this, pion's sendrecv transceiver
	// from AddTrack may not produce an OnTrack callback for the browser's mic.
	if _, err := pc.AddTransceiverFromKind(
		webrtc.RTPCodecTypeAudio,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly},
	); err != nil {
		pc.Close()
		return fmt.Errorf("add mic receive transceiver: %w", err)
	}

	// Handle incoming user audio track (mic)
	pc.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		log.Printf("[DirectPeer] session=%s subscribed to user audio track codec=%s", p.sessionID, track.Codec().MimeType)
		go p.readUserAudio(track)
	})

	// Forward ICE candidates to browser via signaling
	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		if c == nil {
			log.Printf("[DirectPeer] session=%s ICE gathering complete", p.sessionID)
			return
		}
		init := c.ToJSON()
		log.Printf("[DirectPeer] session=%s local ICE candidate: %s | sdp: %s", p.sessionID, c.String(), init.Candidate)
		msg := map[string]any{
			"type":      "ice_candidate",
			"candidate": init.Candidate,
		}
		if init.SDPMid != nil {
			msg["sdp_mid"] = *init.SDPMid
		}
		if init.SDPMLineIndex != nil {
			msg["sdp_mline_index"] = *init.SDPMLineIndex
		}
		p.signalingFn(p.sessionID, msg)
	})

	// Track ICE connection state (more granular than PeerConnection state)
	pc.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		log.Printf("[DirectPeer] session=%s ICE connection state: %s", p.sessionID, state.String())
	})

	// Track connection state
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("[DirectPeer] session=%s connection state: %s", p.sessionID, state.String())
		if state == webrtc.PeerConnectionStateConnected {
			select {
			case <-p.connected:
			default:
				close(p.connected)
			}
		}
	})

	p.mu.Lock()
	p.pc = pc
	p.videoTrack = videoTrack
	p.audioTrack = audioTrack
	p.mu.Unlock()

	// Monitor GCC bandwidth estimation
	if p.estimatorCh != nil {
		go func() {
			select {
			case estimator, ok := <-p.estimatorCh:
				if !ok {
					return
				}
				estimator.OnTargetBitrateChange(func(bitrate int) {
					log.Printf("[DirectPeer] session=%s GCC target bitrate: %d kbps", p.sessionID, bitrate/1000)
				})
			case <-p.connected:
			}
		}()
	}

	return nil
}

// readRTCP continuously reads RTCP packets from an RTPSender.
// This is required for NACK/TWCC/GCC interceptors to receive browser feedback.
func readRTCP(sender *webrtc.RTPSender) {
	rtcpBuf := make([]byte, 1500)
	for {
		if _, _, err := sender.Read(rtcpBuf); err != nil {
			return
		}
	}
}

// StartNegotiation creates an SDP offer and sends it to the browser.
// Call this after the browser sends "webrtc_ready" via WebSocket.
// It waits for ICE gathering to complete so all candidates are embedded
// in the SDP offer itself (no trickle needed).
func (p *DirectPeer) StartNegotiation() error {
	p.mu.Lock()
	pc := p.pc
	p.mu.Unlock()
	if pc == nil {
		return fmt.Errorf("PeerConnection not created")
	}

	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return fmt.Errorf("create offer: %w", err)
	}
	if err := pc.SetLocalDescription(offer); err != nil {
		return fmt.Errorf("set local description: %w", err)
	}

	// Wait for ICE gathering to complete so all candidates are in the SDP.
	// For ICE-TCP with a single TCPMux listener this is nearly instant.
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	select {
	case <-gatherComplete:
	case <-time.After(5 * time.Second):
		log.Printf("[DirectPeer] session=%s ICE gathering timeout, sending partial offer", p.sessionID)
	}

	// Send the complete SDP (with candidates embedded)
	completeOffer := pc.LocalDescription()
	p.signalingFn(p.sessionID, map[string]any{
		"type": "webrtc_offer",
		"sdp":  completeOffer.SDP,
	})
	log.Printf("[DirectPeer] session=%s SDP offer sent (with candidates)", p.sessionID)
	return nil
}

// HandleSignaling processes incoming signaling messages from the browser.
func (p *DirectPeer) HandleSignaling(msgType, sdp, candidate string, sdpMid *string, sdpMLineIndex *uint16) {
	p.mu.Lock()
	pc := p.pc
	p.mu.Unlock()
	if pc == nil {
		return
	}

	switch msgType {
	case "webrtc_answer":
		if err := pc.SetRemoteDescription(webrtc.SessionDescription{
			Type: webrtc.SDPTypeAnswer,
			SDP:  sdp,
		}); err != nil {
			log.Printf("[DirectPeer] session=%s set remote description failed: %v", p.sessionID, err)
		} else {
			log.Printf("[DirectPeer] session=%s SDP answer set", p.sessionID)
		}

	case "ice_candidate":
		if candidate == "" {
			return
		}
		log.Printf("[DirectPeer] session=%s remote ICE candidate: %s", p.sessionID, candidate)
		init := webrtc.ICECandidateInit{
			Candidate: candidate,
		}
		if sdpMid != nil {
			init.SDPMid = sdpMid
		}
		if sdpMLineIndex != nil {
			init.SDPMLineIndex = sdpMLineIndex
		}
		if err := pc.AddICECandidate(init); err != nil {
			log.Printf("[DirectPeer] session=%s add ICE candidate failed: %v", p.sessionID, err)
		}
	}
}

// readUserAudio reads Opus RTP packets from the user's mic track,
// decodes to 16kHz mono PCM, and writes to userAudioCh.
func (p *DirectPeer) readUserAudio(track *webrtc.TrackRemote) {
	decoder, err := opus.NewDecoder(16000, 1)
	if err != nil {
		log.Printf("[DirectPeer] session=%s opus decoder creation failed: %v", p.sessionID, err)
		return
	}

	pcmBuf := make([]int16, 16000) // up to 1 second buffer

	for {
		pkt, _, err := track.ReadRTP()
		if err != nil {
			log.Printf("[DirectPeer] session=%s user audio track closed: %v", p.sessionID, err)
			return
		}

		n, err := decoder.Decode(pkt.Payload, pcmBuf)
		if err != nil {
			continue
		}
		if n == 0 {
			continue
		}

		// Convert int16 → little-endian bytes (matching VoiceLLM expected format)
		out := make([]byte, n*2)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint16(out[i*2:], uint16(pcmBuf[i]))
		}

		select {
		case p.userAudioCh <- out:
		default:
			// Drop if consumer is too slow (same backpressure as Bot)
		}
	}
}

// SubscribeUserAudio returns the channel receiving decoded user mic PCM.
func (p *DirectPeer) SubscribeUserAudio() <-chan []byte {
	return p.userAudioCh
}

// --- AV Pipeline (same pattern as Bot) ---

// StartAVPipeline launches encode and publish goroutines.
func (p *DirectPeer) StartAVPipeline(ctx context.Context) {
	p.avPipelineCtx, p.avPipelineCancel = context.WithCancel(ctx)
	p.encodeCh = make(chan *mediapeer.RawAVSegment, 1)
	p.publishCh = make(chan *mediapeer.AVSegment, 1)

	p.avPipelineWg.Add(2)
	go p.runEncoder()
	go p.runPublisher()
}

// SendAVSegment enqueues a raw segment for encoding and publishing.
func (p *DirectPeer) SendAVSegment(seg *mediapeer.RawAVSegment) error {
	seg.QueuedAt = time.Now()
	if p.isPlaybackStale(seg.Epoch) {
		return nil
	}
	select {
	case p.encodeCh <- seg:
		return nil
	case <-p.avPipelineCtx.Done():
		return fmt.Errorf("av pipeline cancelled")
	}
}

func (p *DirectPeer) AdvancePlaybackEpoch(epoch uint64) {
	if epoch == 0 {
		return
	}
	for {
		current := p.playbackEpoch.Load()
		if epoch <= current {
			return
		}
		if p.playbackEpoch.CompareAndSwap(current, epoch) {
			return
		}
	}
}

// WaitAVDrain blocks until all queued segments are published.
func (p *DirectPeer) WaitAVDrain(timeout time.Duration) {
	if p.encodeCh == nil {
		return
	}
	fence := make(chan struct{})
	select {
	case p.encodeCh <- &mediapeer.RawAVSegment{Fence: fence}:
	case <-p.avPipelineCtx.Done():
		return
	case <-time.After(timeout):
		return
	}
	select {
	case <-fence:
	case <-p.avPipelineCtx.Done():
	case <-time.After(timeout):
	}
}

// StopAVPipeline shuts down the AV pipeline goroutines.
func (p *DirectPeer) StopAVPipeline() {
	if p.avPipelineCancel != nil {
		p.avPipelineCancel()
	}
	if p.encodeCh != nil {
		close(p.encodeCh)
	}
	p.avPipelineWg.Wait()
}

func (p *DirectPeer) runEncoder() {
	defer p.avPipelineWg.Done()
	defer close(p.publishCh)

	for raw := range p.encodeCh {
		if p.avPipelineCtx.Err() != nil {
			return
		}

		// Fence marker: pass through
		if raw.Fence != nil && len(raw.RGB) == 0 {
			select {
			case p.publishCh <- &mediapeer.AVSegment{Fence: raw.Fence}:
			case <-p.avPipelineCtx.Done():
				return
			}
			continue
		}
		if p.isPlaybackStale(raw.Epoch) {
			continue
		}

		vp8Samples, err := mediapeer.EncodeRGBChunkToVP8Samples(raw.RGB, raw.Width, raw.Height, raw.NumFrames, raw.FPS)
		if err != nil {
			log.Printf("[DirectPeer] encode failed: %v", err)
			continue
		}
		if len(vp8Samples) == 0 {
			continue
		}

		seg := &mediapeer.AVSegment{
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
		case p.publishCh <- seg:
		case <-p.avPipelineCtx.Done():
			return
		}
	}
}

func (p *DirectPeer) runPublisher() {
	defer p.avPipelineWg.Done()

	for seg := range p.publishCh {
		if p.avPipelineCtx.Err() != nil {
			return
		}
		// Fence marker
		if seg.Fence != nil && len(seg.VP8Samples) == 0 {
			close(seg.Fence)
			continue
		}
		if p.isPlaybackStale(seg.Epoch) {
			continue
		}
		p.publishAVSegment(seg)
	}
}

func (p *DirectPeer) waitConnected(timeout time.Duration) bool {
	select {
	case <-p.connected:
		return true
	case <-p.avPipelineCtx.Done():
		return false
	case <-time.After(timeout):
		return false
	}
}

func (p *DirectPeer) publishAVSegment(seg *mediapeer.AVSegment) {
	// Wait for connection to be established
	if !p.waitConnected(10 * time.Second) {
		if p.avPipelineCtx.Err() == nil {
			log.Printf("[DirectPeer] session=%s publish timeout waiting for connection", p.sessionID)
		}
		return
	}

	fps := seg.FPS
	if fps <= 0 {
		fps = 25
	}
	frameDur := time.Second / time.Duration(fps)

	// Encode entire PCM buffer into Opus frames up-front to avoid
	// the sample-loss caused by slicing PCM per video frame.
	var opusFrames []media.Sample
	if len(seg.PCM) > 0 && seg.SampleRate > 0 {
		var err error
		opusFrames, err = mediapeer.EncodePCMToOpusSamples(seg.PCM, seg.SampleRate)
		if err != nil {
			log.Printf("[DirectPeer] audio encode error: %v", err)
			return
		}
	}

	nVideo := len(seg.VP8Samples)
	nAudio := len(opusFrames)

	// --- RTP timestamp gap correction ---
	// Between segments, WriteSample is not called so pion's internal RTP
	// timestamp counter freezes. When the next segment arrives, the first
	// frame's timestamp is only frameDur ahead of the last frame, but the
	// actual wall-clock gap is seconds. The browser's jitter buffer sees
	// this as "frame arrived seconds late" and grows indefinitely.
	//
	// Fix: set the first sample's Duration to the actual wall-clock gap.
	// pion advances the RTP timestamp by Duration * clockRate before
	// writing the packet, so the timestamp jumps to match arrival time.
	now := time.Now()
	if !p.lastVideoWriteTime.IsZero() && len(seg.VP8Samples) > 0 {
		videoGap := now.Sub(p.lastVideoWriteTime)
		if videoGap > 2*frameDur {
			seg.VP8Samples[0].Duration = videoGap
			log.Printf("[DirectPeer] session=%s RTP timestamp gap correction: video=%v", p.sessionID, videoGap)
		}
	}
	if !p.lastAudioWriteTime.IsZero() && len(opusFrames) > 0 {
		audioGap := now.Sub(p.lastAudioWriteTime)
		if audioGap > 40*time.Millisecond {
			opusFrames[0].Duration = audioGap
		}
	}

	for i := range seg.VP8Samples {
		if p.isPlaybackStale(seg.Epoch) {
			return
		}
		frameStart := time.Now()

		// Distribute pre-encoded Opus frames evenly across video frames.
		// Video frame i sends opusFrames[lo:hi].
		if nAudio > 0 {
			lo := (nAudio * i) / nVideo
			hi := (nAudio * (i + 1)) / nVideo
			for j := lo; j < hi; j++ {
				if err := p.audioTrack.WriteSample(opusFrames[j]); err != nil {
					log.Printf("[DirectPeer] audio write error: %v", err)
					return
				}
			}
		}

		// Write VP8 video sample
		if err := p.videoTrack.WriteSample(seg.VP8Samples[i]); err != nil {
			log.Printf("[DirectPeer] video write error: %v", err)
			return
		}

		// Real-time pacing
		if elapsed := time.Since(frameStart); elapsed < frameDur {
			time.Sleep(frameDur - elapsed)
		}
	}

	// Record the last write time for gap correction on the next segment.
	p.lastVideoWriteTime = time.Now()
	p.lastAudioWriteTime = p.lastVideoWriteTime
}

func (p *DirectPeer) isPlaybackStale(epoch uint64) bool {
	if epoch == 0 {
		return false
	}
	current := p.playbackEpoch.Load()
	return current > 0 && epoch < current
}

// PublishAudioFrame publishes raw PCM audio (for TTS in standard pipeline).
func (p *DirectPeer) PublishAudioFrame(pcm []byte, sampleRate int) error {
	if len(pcm) == 0 || sampleRate <= 0 {
		return nil
	}
	if !p.waitConnected(10 * time.Second) {
		if p.avPipelineCtx.Err() != nil {
			return fmt.Errorf("audio publish cancelled")
		}
		return fmt.Errorf("audio publish timeout waiting for connection")
	}
	return p.writeOpus(pcm, sampleRate)
}

// writeOpus encodes PCM to Opus and writes to the audio track.
// Opus encodes in 20ms frames.
func (p *DirectPeer) writeOpus(pcm []byte, sampleRate int) error {
	p.opusMu.Lock()
	defer p.opusMu.Unlock()

	// Lazily create or recreate encoder if sample rate changed
	if p.opusEncoder == nil || p.opusEncoderSR != sampleRate {
		enc, err := opus.NewEncoder(sampleRate, 1, opus.AppVoIP)
		if err != nil {
			return fmt.Errorf("create opus encoder: %w", err)
		}
		p.opusEncoder = enc
		p.opusEncoderSR = sampleRate
	}

	// Process PCM in 20ms frames
	samplesPerFrame := sampleRate / 50 // 20ms = 1/50 second
	bytesPerFrame := samplesPerFrame * 2 // 16-bit mono
	opusBuf := make([]byte, 4000)       // max opus frame size

	for offset := 0; offset+bytesPerFrame <= len(pcm); offset += bytesPerFrame {
		// Convert bytes to int16 samples
		samples := make([]int16, samplesPerFrame)
		for i := 0; i < samplesPerFrame; i++ {
			samples[i] = int16(binary.LittleEndian.Uint16(pcm[offset+i*2:]))
		}

		n, err := p.opusEncoder.Encode(samples, opusBuf)
		if err != nil {
			return fmt.Errorf("opus encode: %w", err)
		}
		if n == 0 {
			continue
		}

		if err := p.audioTrack.WriteSample(media.Sample{
			Data:     opusBuf[:n],
			Duration: 20 * time.Millisecond,
		}); err != nil {
			return fmt.Errorf("write audio sample: %w", err)
		}
	}
	return nil
}

// Disconnect tears down the peer connection and releases resources.
func (p *DirectPeer) Disconnect() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.pc != nil {
		err := p.pc.Close()
		p.pc = nil
		close(p.userAudioCh)
		return err
	}
	return nil
}
