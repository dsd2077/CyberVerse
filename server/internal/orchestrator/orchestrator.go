package orchestrator

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cyberverse/server/internal/character"
	"github.com/cyberverse/server/internal/config"
	"github.com/cyberverse/server/internal/direct"
	"github.com/cyberverse/server/internal/inference"
	"github.com/cyberverse/server/internal/livekit"
	"github.com/cyberverse/server/internal/mediapeer"
	"github.com/cyberverse/server/internal/pb"
	"github.com/cyberverse/server/internal/recording"
	"github.com/cyberverse/server/internal/ws"
	"github.com/pion/interceptor/pkg/cc"
	"github.com/pion/webrtc/v4"
	"google.golang.org/protobuf/proto"
)

// stdChunksPerSegment is how many avatar video chunks to batch before publishing
// in the standard (text→LLM→TTS→Avatar) pipeline. Each chunk is ~1.12s of
// content but takes ~1.64s to generate (RTF≈1.46). Batching 3 chunks into one
// 3.36s segment converts constant 0.52s micro-stutters into one clear ~1.9s
// pause every ~5s, matching the Gradio streaming reference approach.
const stdChunksPerSegment = 3

// No hard cap on the assistant PCM buffer: long responses (>20s) were
// previously truncated, causing the first N seconds of audio to be dropped
// and all video segments to play with misaligned (or silent) audio.
// Set to 0 to disable the overflow guard entirely.
const voiceMaxPCMBufferSamples = 0

type voiceAVSyncBuffer struct {
	mu               sync.Mutex
	pcmBytes         []byte
	sampleRate       int
	totalAudioIn     int64
	totalAudioOut    int64
	maxBufferSamples int
	// Carries fractional samples from frames*sampleRate/fps to avoid
	// long-session drift caused by per-segment integer rounding.
	sampleCarryNumer int64
}

func newVoiceAVSyncBuffer(maxBufferSamples int) *voiceAVSyncBuffer {
	if maxBufferSamples <= 0 {
		maxBufferSamples = voiceMaxPCMBufferSamples
	}
	return &voiceAVSyncBuffer{maxBufferSamples: maxBufferSamples}
}

func (b *voiceAVSyncBuffer) appendPCM(pcm []byte, sampleRate int) (droppedBytes int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(pcm) == 0 || sampleRate <= 0 {
		return 0
	}
	if b.sampleRate == 0 {
		b.sampleRate = sampleRate
	}
	if b.sampleRate != sampleRate {
		b.sampleRate = sampleRate
		b.pcmBytes = nil
	}

	if len(pcm)%2 != 0 {
		pcm = pcm[:len(pcm)-1]
	}
	if len(pcm) == 0 {
		return 0
	}

	b.pcmBytes = append(b.pcmBytes, pcm...)
	b.totalAudioIn += int64(len(pcm) / 2)

	maxBytes := b.maxBufferSamples * 2
	if maxBytes > 0 && len(b.pcmBytes) > maxBytes {
		droppedBytes = len(b.pcmBytes) - maxBytes
		if droppedBytes%2 != 0 {
			droppedBytes++
		}
		b.pcmBytes = b.pcmBytes[droppedBytes:]
	}
	return droppedBytes
}

func desiredSamplesForVideo(frames, fps, sampleRate int) int {
	if frames <= 0 || fps <= 0 || sampleRate <= 0 {
		return 0
	}
	// Rounded target samples for segment duration = frames / fps seconds.
	return (frames*sampleRate + fps/2) / fps
}

func (b *voiceAVSyncBuffer) takeSegmentPCM(frames, fps int, isFinal bool) ([]byte, int, int, int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if frames <= 0 || fps <= 0 || b.sampleRate <= 0 {
		return nil, 0, 0, 0
	}
	// Exact target with carry:
	// want = floor((frames*sampleRate + carry)/fps), carry = modulo part.
	numer := int64(frames*b.sampleRate) + b.sampleCarryNumer
	wantSamples := int(numer / int64(fps))
	b.sampleCarryNumer = numer % int64(fps)
	if wantSamples <= 0 {
		wantSamples = desiredSamplesForVideo(frames, fps, b.sampleRate)
	}
	if wantSamples <= 0 {
		return nil, 0, 0, len(b.pcmBytes) / 2
	}
	wantBytes := wantSamples * 2

	availableBytes := len(b.pcmBytes)
	takeBytes := wantBytes
	if takeBytes > availableBytes {
		takeBytes = availableBytes
	}
	if takeBytes%2 != 0 {
		takeBytes--
	}

	out := make([]byte, wantBytes) // strict lip-sync: always return exact segment duration
	if takeBytes > 0 {
		copy(out, b.pcmBytes[:takeBytes])
		b.pcmBytes = b.pcmBytes[takeBytes:]
	}
	outSamples := takeBytes / 2
	b.totalAudioOut += int64(outSamples)
	if isFinal {
		// Final close-loop: strict mode prefers exact A/V alignment over
		// carrying remaining tail audio into post-video silence.
		b.pcmBytes = nil
	}
	return out, outSamples, wantSamples, len(b.pcmBytes) / 2
}

func (b *voiceAVSyncBuffer) snapshot() (bufferedSamples int, totalIn int64, totalOut int64, sampleRate int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.pcmBytes) / 2, b.totalAudioIn, b.totalAudioOut, b.sampleRate
}

// Orchestrator manages the inference pipeline for each session,
// coordinating between the gRPC inference client, media peers,
// and WebSocket hub for real-time updates.
type Orchestrator struct {
	inference     inference.InferenceService
	wsHub         *ws.Hub
	sessionMgr    *SessionManager
	charStore     *character.Store
	peers         map[string]mediapeer.MediaPeer // sessionID → media peer (Bot or DirectPeer)
	directPeers   map[string]*direct.DirectPeer  // sessionID → DirectPeer (for signaling dispatch)
	recorder      *recording.VideoRecorder
	streamingMode string
	pipelineCfg   config.PipelineConfig
	turnServer    *direct.TURNServer
	webrtcAPI     *webrtc.API
	estimatorCh   <-chan cc.BandwidthEstimator
	avatarMu      sync.Mutex
	mu            sync.RWMutex
}

// New creates a new Orchestrator.
func New(inferenceClient inference.InferenceService, hub *ws.Hub, sessionMgr *SessionManager, recorder *recording.VideoRecorder, charStore *character.Store, pipelineCfg ...config.PipelineConfig) *Orchestrator {
	o := &Orchestrator{
		inference:   inferenceClient,
		wsHub:       hub,
		sessionMgr:  sessionMgr,
		charStore:   charStore,
		peers:       make(map[string]mediapeer.MediaPeer),
		directPeers: make(map[string]*direct.DirectPeer),
		recorder:    recorder,
	}
	if len(pipelineCfg) > 0 {
		o.pipelineCfg = pipelineCfg[0]
		o.streamingMode = pipelineCfg[0].StreamingMode
	}
	if o.streamingMode == "" {
		o.streamingMode = "direct"
	}
	return o
}

// HandleSignaling dispatches WebRTC signaling messages to the DirectPeer.
func (o *Orchestrator) HandleSignaling(sessionID string, msg ws.WSMessage) {
	o.mu.RLock()
	dp := o.directPeers[sessionID]
	o.mu.RUnlock()
	if dp == nil {
		return
	}

	switch msg.Type {
	case "webrtc_ready":
		// Send TURN ICE server config before the SDP offer
		if o.turnServer != nil {
			host := o.pipelineCfg.ICEPublicIP
			if host == "" {
				host = "127.0.0.1"
			}
			o.broadcastJSON(sessionID, map[string]any{
				"type":        "webrtc_config",
				"ice_servers": []any{o.turnServer.ICEServerConfig(host)},
			})
		}
		if err := dp.StartNegotiation(); err != nil {
			log.Printf("[Orchestrator] session=%s StartNegotiation failed: %v", sessionID, err)
		}
	case "webrtc_answer", "ice_candidate":
		var sdpMid *string
		if msg.SDPMid != "" {
			sdpMid = &msg.SDPMid
		}
		dp.HandleSignaling(msg.Type, msg.SDP, msg.Candidate, sdpMid, msg.SDPMLine)
	}
}

// SetTURNServer sets the embedded TURN server for NAT traversal.
func (o *Orchestrator) SetTURNServer(ts *direct.TURNServer) {
	o.turnServer = ts
}

// SetWebRTCAPI sets the shared webrtc.API with interceptors (NACK, TWCC, GCC).
func (o *Orchestrator) SetWebRTCAPI(api *webrtc.API, estimatorCh <-chan cc.BandwidthEstimator) {
	o.webrtcAPI = api
	o.estimatorCh = estimatorCh
}

// StreamingMode returns the current streaming mode.
func (o *Orchestrator) StreamingMode() string {
	return o.streamingMode
}

func (o *Orchestrator) HealthCheck(ctx context.Context) error {
	if o == nil || o.inference == nil {
		return errors.New("inference service is not configured")
	}
	return o.inference.HealthCheck(ctx)
}

func (o *Orchestrator) CheckVoice(ctx context.Context, voiceType string) (string, error) {
	if o == nil || o.inference == nil {
		return "", errors.New("inference service is not configured")
	}
	return o.inference.CheckVoice(ctx, inference.VoiceLLMSessionConfig{
		Voice: voiceType,
	})
}

func (o *Orchestrator) AvatarInfo(ctx context.Context) (*pb.AvatarInfo, error) {
	if o == nil || o.inference == nil {
		return nil, errors.New("inference service is not configured")
	}
	return o.inference.AvatarInfo(ctx)
}

func (o *Orchestrator) idleVideoProfile() string {
	return character.DefaultIdleVideoProfile
}

func (o *Orchestrator) idleVideoOutputSize(ctx context.Context) (int, int, error) {
	info, err := o.AvatarInfo(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("get avatar info for idle video: %w", err)
	}

	width := int(info.GetOutputWidth())
	height := int(info.GetOutputHeight())
	if width <= 0 || height <= 0 {
		return 0, 0, fmt.Errorf("invalid idle video output size: %dx%d", width, height)
	}
	return width, height, nil
}

func (o *Orchestrator) activeCharacterImage(characterID string) (*character.Character, string, error) {
	if o == nil || o.charStore == nil {
		return nil, "", errors.New("character store is not configured")
	}
	char, err := o.charStore.Get(characterID)
	if err != nil {
		return nil, "", err
	}
	if char.ActiveImage == "" {
		return char, "", nil
	}
	return char, char.ActiveImage, nil
}

func normalizeImageFormat(imageFilename string) string {
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(imageFilename)), ".")
	if ext == "" {
		return "png"
	}
	if ext == "jpg" {
		return "jpeg"
	}
	return ext
}

func (o *Orchestrator) loadCharacterImage(characterID, imageFilename string) ([]byte, string, error) {
	if o == nil || o.charStore == nil {
		return nil, "", errors.New("character store is not configured")
	}
	if imageFilename == "" {
		return nil, "", errors.New("active image is empty")
	}
	imgDir := o.charStore.ImagesDir(characterID)
	if imgDir == "" {
		return nil, "", fmt.Errorf("character images dir not found: %s", characterID)
	}
	path := filepath.Join(imgDir, filepath.Base(imageFilename))
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, "", fmt.Errorf("read character image %s: %w", path, err)
	}
	return data, normalizeImageFormat(imageFilename), nil
}

// buildTrailingSilence creates a 1.5-second silent PCM chunk (s16le mono)
// appended after TTS audio so the avatar can close its mouth before the idle switch.
func buildTrailingSilence(sampleRate int) *pb.AudioChunk {
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	numSamples := sampleRate * 3 / 2 // 1.5 seconds
	return &pb.AudioChunk{
		Data:       make([]byte, numSamples*2), // s16le: 2 bytes per sample
		SampleRate: int32(sampleRate),
		Channels:   1,
		Format:     "pcm_s16le",
		IsFinal:    true,
	}
}

func buildIdleBreathingPCM(duration time.Duration, sampleRate int) []byte {
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	totalSamples := int(math.Round(duration.Seconds() * float64(sampleRate)))
	if totalSamples <= 0 {
		return nil
	}

	out := make([]byte, totalSamples*2)
	rng := rand.New(rand.NewSource(42))
	fadeSamples := int(0.25 * float64(sampleRate))

	for i := 0; i < totalSamples; i++ {
		t := float64(i) / float64(sampleRate)
		cyclePos := math.Mod(t, 3.8)

		var env float64
		switch {
		case cyclePos < 1.1:
			p := cyclePos / 1.1
			env = 0.010 + 0.020*math.Sin(p*math.Pi/2)
		case cyclePos < 1.5:
			env = 0.028
		case cyclePos < 3.0:
			p := (cyclePos - 1.5) / 1.5
			env = 0.030 + 0.020*math.Cos(p*math.Pi/2)
		default:
			env = 0.006
		}

		texture := 0.55*math.Sin(2*math.Pi*170*t) +
			0.25*math.Sin(2*math.Pi*310*t+0.7) +
			0.20*(rng.Float64()*2-1)
		motion := 0.92 + 0.08*math.Sin(2*math.Pi*0.21*t+0.4)
		sample := env * texture * motion

		if fadeSamples > 0 {
			if i < fadeSamples {
				sample *= float64(i) / float64(fadeSamples)
			} else if remain := totalSamples - i; remain < fadeSamples {
				sample *= float64(remain) / float64(fadeSamples)
			}
		}

		if sample > 0.95 {
			sample = 0.95
		}
		if sample < -0.95 {
			sample = -0.95
		}
		pcm := int16(sample * 32767)
		binary.LittleEndian.PutUint16(out[i*2:], uint16(pcm))
	}

	return out
}

func fitPCMToVideoDuration(pcm []byte, sampleRate, frames, fps int) []byte {
	if len(pcm) == 0 || sampleRate <= 0 || frames <= 0 || fps <= 0 {
		return pcm
	}
	wantSamples := desiredSamplesForVideo(frames, fps, sampleRate)
	if wantSamples <= 0 {
		return pcm
	}
	wantBytes := wantSamples * 2
	if len(pcm) == wantBytes {
		return pcm
	}
	if len(pcm) > wantBytes {
		return pcm[:wantBytes]
	}
	out := make([]byte, wantBytes)
	copy(out, pcm)
	return out
}

func (o *Orchestrator) setAvatarFromCharacterImage(ctx context.Context, sessionID, characterID, imageFilename string) error {
	if o == nil || o.inference == nil {
		return errors.New("inference service is not configured")
	}
	imageData, format, err := o.loadCharacterImage(characterID, imageFilename)
	if err != nil {
		return err
	}

	o.avatarMu.Lock()
	defer o.avatarMu.Unlock()

	return o.inference.SetAvatar(ctx, sessionID, imageData, format)
}

// EnsureIdleVideo generates and caches the idle MP4 for the active image if missing.
func (o *Orchestrator) EnsureIdleVideo(ctx context.Context, characterID string) (string, error) {
	if o == nil || o.charStore == nil {
		return "", errors.New("character store is not configured")
	}
	if o.inference == nil {
		return "", errors.New("inference service is not configured")
	}

	_, imageFilename, err := o.activeCharacterImage(characterID)
	if err != nil || imageFilename == "" {
		return "", err
	}

	profile := o.idleVideoProfile()
	targetWidth, targetHeight, err := o.idleVideoOutputSize(ctx)
	if err != nil {
		return "", err
	}

	sizeDir := o.charStore.IdleVideosForSizeDir(characterID, imageFilename, targetWidth, targetHeight)
	if sizeDir == "" {
		return "", fmt.Errorf("idle video dir unavailable for character %s", characterID)
	}
	files, err := o.charStore.ListIdleVideos(characterID, imageFilename, targetWidth, targetHeight)
	if err == nil && len(files) > 0 {
		return filepath.Join(sizeDir, files[0]), nil
	}

	outPath := o.charStore.IdleVideoPath(characterID, imageFilename, profile, targetWidth, targetHeight)
	if outPath == "" {
		return "", fmt.Errorf("idle video path unavailable for character %s", characterID)
	}

	imageData, format, err := o.loadCharacterImage(characterID, imageFilename)
	if err != nil {
		return "", err
	}

	const (
		idleDuration   = 10 * time.Second
		idleSampleRate = 16000
		idleCRF        = 23
	)
	pcm := buildIdleBreathingPCM(idleDuration, idleSampleRate)
	audioChunk := &pb.AudioChunk{
		Data:       pcm,
		SampleRate: idleSampleRate,
		Channels:   1,
		Format:     "pcm_s16le",
		IsFinal:    true,
	}

	// Hold the mutex for the entire generation cycle (SetAvatar + GenerateAvatar
	// + frame collection) so that a concurrent SetupSession or another
	// EnsureIdleVideo call cannot change the inference server's avatar state
	// while we are still collecting frames.
	o.avatarMu.Lock()
	defer o.avatarMu.Unlock()

	jobID := fmt.Sprintf("idle-%s-%d", characterID, time.Now().UnixNano())
	if err := o.inference.SetAvatar(ctx, jobID, imageData, format); err != nil {
		return "", fmt.Errorf("set avatar for idle video: %w", err)
	}
	videoCh, errCh := o.inference.GenerateAvatar(ctx, []*pb.AudioChunk{audioChunk})

	rgbChunks := make([][]byte, 0, 8)
	width, height, fps, totalFrames := 0, 0, 25, 0
loop:
	for {
		select {
		case chunk, ok := <-videoCh:
			if !ok {
				break loop
			}
			if chunk == nil || len(chunk.Data) == 0 {
				continue
			}
			if width == 0 {
				width = int(chunk.Width)
				height = int(chunk.Height)
				if int(chunk.Fps) > 0 {
					fps = int(chunk.Fps)
				}
			}
			totalFrames += int(chunk.NumFrames)
			rgbCopy := make([]byte, len(chunk.Data))
			copy(rgbCopy, chunk.Data)
			rgbChunks = append(rgbChunks, rgbCopy)
		case genErr := <-errCh:
			if genErr != nil {
				// Drain videoCh so the gRPC stream can close cleanly.
				for range videoCh {
				}
				return "", fmt.Errorf("generate idle avatar video: %w", genErr)
			}
		}
	}
	// Drain errCh after videoCh closes in case an error arrived concurrently.
	select {
	case genErr := <-errCh:
		if genErr != nil {
			return "", fmt.Errorf("generate idle avatar video: %w", genErr)
		}
	default:
	}
	if len(rgbChunks) == 0 || width <= 0 || height <= 0 || totalFrames <= 0 {
		return "", errors.New("idle avatar generation produced no video frames")
	}
	if width != targetWidth || height != targetHeight {
		sizeDir = o.charStore.IdleVideosForSizeDir(characterID, imageFilename, width, height)
		if sizeDir == "" {
			return "", fmt.Errorf("idle video dir unavailable for character %s", characterID)
		}
		outPath = o.charStore.IdleVideoPath(characterID, imageFilename, profile, width, height)
		if outPath == "" {
			return "", fmt.Errorf("idle video path unavailable for character %s", characterID)
		}
	}

	pcm = fitPCMToVideoDuration(pcm, idleSampleRate, totalFrames, fps)
	if err := recording.EncodeRGB24ToMP4(outPath, width, height, fps, rgbChunks, pcm, idleSampleRate, idleCRF); err != nil {
		return "", fmt.Errorf("encode idle avatar mp4: %w", err)
	}
	return outPath, nil
}

// SetupSession creates a media peer (DirectPeer or LiveKit Bot) and prepares for streaming.
// When roomMgr is nil (direct mode), a DirectPeer is created instead of a LiveKit Bot.
func (o *Orchestrator) SetupSession(ctx context.Context, session *Session, roomMgr *livekit.RoomManager) (mediapeer.MediaPeer, error) {
	// Best-effort: apply the character's active avatar image.
	if session != nil && session.CharacterID != "" {
		_, imageFilename, err := o.activeCharacterImage(session.CharacterID)
		if err != nil {
			log.Printf("SetupSession: could not resolve active image for character %s: %v", session.CharacterID, err)
		} else if imageFilename != "" {
			if err := o.setAvatarFromCharacterImage(ctx, session.ID, session.CharacterID, imageFilename); err != nil {
				log.Printf("SetupSession: SetAvatar failed for character %s (proceeding with default avatar): %v", session.CharacterID, err)
			}
		}
	}

	var peer mediapeer.MediaPeer

	if o.streamingMode == "livekit" {
		// LiveKit SFU mode
		roomName := livekit.RoomName(session.ID)
		if err := roomMgr.CreateRoom(ctx, roomName); err != nil {
			return nil, err
		}

		bot := livekit.NewBot(
			roomMgr.URL(),
			roomMgr.APIKey(),
			roomMgr.APISecret(),
			roomName,
		)
		if err := bot.Connect(ctx); err != nil {
			return nil, err
		}
		peer = bot
	} else {
		// Direct P2P WebRTC mode
		signalingFn := func(sessionID string, msg map[string]any) {
			o.broadcastJSON(sessionID, msg)
		}
		iceServers := make([]webrtc.ICEServer, 0, len(o.pipelineCfg.ICEServers))
		for _, s := range o.pipelineCfg.ICEServers {
			iceServers = append(iceServers, webrtc.ICEServer{
				URLs:       s.URLs,
				Username:   s.Username,
				Credential: s.Credential,
			})
		}
		dp := direct.NewDirectPeer(session.ID, signalingFn, iceServers, o.webrtcAPI, o.estimatorCh)
		if err := dp.Connect(ctx); err != nil {
			return nil, err
		}
		peer = dp

		o.mu.Lock()
		o.directPeers[session.ID] = dp
		o.mu.Unlock()
	}

	// Use a detached context for the AV pipeline so it outlives the HTTP
	// request / setup timeout that ctx may be derived from.
	peer.StartAVPipeline(context.Background())

	o.mu.Lock()
	o.peers[session.ID] = peer
	o.mu.Unlock()

	session.SetState(StateConnected)
	return peer, nil
}

func (o *Orchestrator) stopPipelineAndWait(session *Session, sessionID string, interruptVoice bool) {
	if interruptVoice && session.Mode == ModeVoiceLLM && o.inference != nil {
		if err := o.inference.Interrupt(context.Background(), sessionID); err != nil {
			log.Printf("Failed to interrupt VoiceLLM for session %s: %v", sessionID, err)
		}
	}
	o.cancelPipeline(session)
	session.WaitPipelineDone(3 * time.Second)
}

func (o *Orchestrator) buildVoiceLLMSessionConfig(session *Session, sessionID string) inference.VoiceLLMSessionConfig {
	voiceConfig := inference.VoiceLLMSessionConfig{SessionID: sessionID}
	if session.CharacterID != "" && o.charStore != nil {
		if char, err := o.charStore.Get(session.CharacterID); err == nil {
			voiceConfig.SystemPrompt = char.SystemPrompt
			voiceConfig.Voice = char.VoiceType
			voiceConfig.BotName = char.Name
			voiceConfig.SpeakingStyle = char.SpeakingStyle
			voiceConfig.WelcomeMessage = session.ConsumeVoiceWelcomeMessage(char.WelcomeMessage)
		} else {
			log.Printf("buildVoiceLLMSessionConfig: could not fetch character %s: %v", session.CharacterID, err)
		}
	}
	return voiceConfig
}

func wrapVoiceAudioInput(ctx context.Context, audioCh <-chan []byte) <-chan inference.VoiceLLMInputEvent {
	inputCh := make(chan inference.VoiceLLMInputEvent, 64)
	go func() {
		defer close(inputCh)
		for {
			select {
			case <-ctx.Done():
				return
			case data, ok := <-audioCh:
				if !ok {
					return
				}
				if len(data) == 0 {
					continue
				}
				select {
				case inputCh <- inference.VoiceLLMInputEvent{Audio: data}:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return inputCh
}

func singleVoiceTextInput(text string) <-chan inference.VoiceLLMInputEvent {
	inputCh := make(chan inference.VoiceLLMInputEvent, 1)
	inputCh <- inference.VoiceLLMInputEvent{Text: text}
	close(inputCh)
	return inputCh
}

func drainUserAudio(audioCh <-chan []byte, maxDrain int) {
	for i := 0; i < maxDrain; i++ {
		select {
		case <-audioCh:
		default:
			return
		}
	}
}

func (o *Orchestrator) resumeVoiceAudioStream(sessionID string) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}
	if session.Mode != ModeVoiceLLM {
		return nil
	}

	o.mu.RLock()
	peer := o.peers[sessionID]
	o.mu.RUnlock()
	if peer == nil {
		return errors.New("media peer not found")
	}

	audioCh := peer.SubscribeUserAudio()
	drainUserAudio(audioCh, 256)
	return o.HandleAudioStream(context.Background(), sessionID, audioCh)
}

func (o *Orchestrator) handleStandardTextInput(ctx context.Context, session *Session, sessionID string, text string) error {
	o.stopPipelineAndWait(session, sessionID, false)

	pipeCtx, cancel := context.WithCancel(ctx)
	session.mu.Lock()
	session.PipelineCancel = cancel
	session.mu.Unlock()

	session.AddMessage(ChatMessage{Role: "user", Content: text})
	pipelineSeq := session.MarkPipelineRunning()
	go o.runStandardPipeline(pipeCtx, session, sessionID, pipelineSeq)
	return nil
}

func (o *Orchestrator) handleVoiceLLMTextInput(ctx context.Context, session *Session, sessionID string, text string) error {
	o.stopPipelineAndWait(session, sessionID, true)

	pipeCtx, cancel := context.WithCancel(ctx)
	session.mu.Lock()
	session.PipelineCancel = cancel
	session.mu.Unlock()

	session.AddMessage(ChatMessage{Role: "user", Content: text})
	pipelineSeq := session.MarkPipelineRunning()
	inputCh := singleVoiceTextInput(text)

	go func(seq uint64) {
		o.runVoiceLLMPipeline(pipeCtx, session, sessionID, inputCh, seq)
		if pipeCtx.Err() != nil || !session.IsCurrentPipeline(seq) {
			return
		}
		if err := o.resumeVoiceAudioStream(sessionID); err != nil {
			log.Printf("Failed to resume VoiceLLM audio stream for session %s: %v", sessionID, err)
		}
	}(pipelineSeq)

	return nil
}

// HandleTextInput processes a text message through either the standard
// LLM→TTS→Avatar pipeline or the VoiceLLM text-query path.
func (o *Orchestrator) HandleTextInput(ctx context.Context, sessionID string, text string) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if session.Mode == ModeVoiceLLM {
		return o.handleVoiceLLMTextInput(ctx, session, sessionID, text)
	}
	return o.handleStandardTextInput(ctx, session, sessionID, text)
}

// runStandardPipeline executes: LLM → sentence detection → TTS → Avatar.
func (o *Orchestrator) runStandardPipeline(ctx context.Context, session *Session, sessionID string, pipelineSeq uint64) {
	var fullResponseCh chan string // set below; read in defer to store assistant message
	defer func() {
		// Store assistant message in session history
		if fullResponseCh != nil {
			if resp, ok := <-fullResponseCh; ok && resp != "" {
				session.AddMessage(ChatMessage{Role: "assistant", Content: resp})
			}
		}
		session.MarkPipelineFinished(pipelineSeq)
		session.SetState(StateListening)
		o.broadcastStatus(sessionID, "idle")
	}()

	var turnRec *recording.TurnRecording
	if o.recorder != nil {
		sessionDir := o.sessionRecordingDir(session)
		turnRec = o.recorder.BeginTurn(sessionDir, "turn1", 512, 512, 25)
	}

	session.SetState(StateProcessing)
	o.broadcastStatus(sessionID, "processing")

	pipelineStart := time.Now()

	// Prepare LLM messages
	messages := make([]inference.ChatMessage, len(session.History))
	for i, m := range session.History {
		messages[i] = inference.ChatMessage{Role: m.Role, Content: m.Content}
	}

	// 1. Start LLM stream
	llmCh, llmErrCh := o.inference.GenerateLLMStream(ctx, sessionID, messages, inference.LLMConfig{
		Temperature: 0.7,
	})

	// 2. Collect LLM tokens, detect sentence boundaries, feed to TTS
	textCh := make(chan string, 8)
	fullResponseCh = make(chan string, 1) // captures full LLM response for history
	go func() {
		defer close(textCh)
		var sentence strings.Builder
		var fullResponse string
		sentenceEnders := ".!?。！？；;\n"

		for {
			select {
			case <-ctx.Done():
				if fullResponse != "" {
					fullResponseCh <- fullResponse
				}
				close(fullResponseCh)
				return
			case chunk, ok := <-llmCh:
				if !ok {
					// Flush remaining sentence
					if sentence.Len() > 0 {
						select {
						case textCh <- sentence.String():
						case <-ctx.Done():
						}
					}
					if fullResponse != "" {
						fullResponseCh <- fullResponse
					}
					close(fullResponseCh)
					return
				}

				// Broadcast LLM token to WebSocket
				o.broadcastJSON(sessionID, map[string]any{
					"type":        "llm_token",
					"token":       chunk.Token,
					"accumulated": chunk.AccumulatedText,
					"is_final":    chunk.IsFinal,
				})

				sentence.WriteString(chunk.Token)
				// Track full accumulated text
				if chunk.IsFinal {
					fullResponse = chunk.AccumulatedText
				}

				// Check for sentence boundary
				if chunk.IsSentenceEnd || (len(chunk.Token) > 0 && strings.ContainsAny(chunk.Token[len(chunk.Token)-1:], sentenceEnders)) {
					text := strings.TrimSpace(sentence.String())
					if text != "" {
						select {
						case textCh <- text:
						case <-ctx.Done():
							return
						}
					}
					sentence.Reset()
				}
			case err := <-llmErrCh:
				if err != nil {
					log.Printf("LLM stream error for session %s: %v", sessionID, err)
					o.broadcastError(sessionID, "LLM generation failed")
				}
				if fullResponse != "" {
					fullResponseCh <- fullResponse
				}
				close(fullResponseCh)
				return
			}
		}
	}()

	// 3. Start TTS stream
	ttsAudioCh, ttsErrCh := o.inference.SynthesizeSpeechStream(ctx, textCh)

	// 4. Start Avatar stream
	lastTTSSampleRate := 16000
	avatarAudioCh := make(chan *pb.AudioChunk, 8)
	go func() {
		defer close(avatarAudioCh)
		for {
			select {
			case <-ctx.Done():
				return
			case chunk, ok := <-ttsAudioCh:
				if !ok {
					// TTS done — append trailing silence so the avatar has
					// time to close its mouth before switching to idle video.
					silence := buildTrailingSilence(lastTTSSampleRate)
					select {
					case avatarAudioCh <- silence:
					case <-ctx.Done():
					}
					return
				}
				lastTTSSampleRate = int(chunk.GetSampleRate())
				// Forward audio to avatar and publish to bot
				select {
				case avatarAudioCh <- chunk:
				case <-ctx.Done():
					return
				}

				// Also publish audio to media peer
				o.mu.RLock()
				peer := o.peers[sessionID]
				o.mu.RUnlock()
				if peer != nil {
					_ = peer.PublishAudioFrame(chunk.Data, int(chunk.SampleRate))
				}
			case err := <-ttsErrCh:
				if err != nil {
					log.Printf("TTS stream error for session %s: %v", sessionID, err)
					o.broadcastError(sessionID, "Speech synthesis failed")
				}
				return
			}
		}
	}()

	// Delay speaking status until first video frame arrives (avoids frozen-frame stall on frontend).
	speakingBroadcasted := false

	// Serialize with concurrent avatar operations (see runVoiceLLMPipeline comment).
	o.avatarMu.Lock()
	videoCh, videoErrCh := o.inference.GenerateAvatarStream(ctx, avatarAudioCh)

	// 5. Publish video frames to LiveKit.
	// Accumulate stdChunksPerSegment chunks before publishing to avoid constant
	// micro-stutters caused by RTF>1 (model generates slower than real-time).
	var (
		segVideo       []byte
		segFrames      int
		segWidth       int
		segHeight      int
		segFPS         int
		segCount       int
		firstFrameSent bool
	)
	flushStdSeg := func() {
		if segCount == 0 {
			return
		}
		o.mu.RLock()
		peer := o.peers[sessionID]
		o.mu.RUnlock()
		if peer != nil {
			raw := &mediapeer.RawAVSegment{
				RGB:       segVideo,
				Width:     segWidth,
				Height:    segHeight,
				FPS:       segFPS,
				NumFrames: segFrames,
			}
			if err := peer.SendAVSegment(raw); err != nil {
				log.Printf("std av SendAVSegment failed session=%s: %v", sessionID, err)
			}
		}
		segVideo = nil
		segFrames = 0
		segCount = 0
	}

	for {
		select {
		case <-ctx.Done():
			flushStdSeg()
			if turnRec != nil {
				_ = turnRec.Finish()
			}
			o.avatarMu.Unlock()
			return
		case chunk, ok := <-videoCh:
			if !ok {
				flushStdSeg()
				if turnRec != nil {
					_ = turnRec.Finish()
				}
				if err := <-videoErrCh; err != nil {
					log.Printf("Avatar stream error for session %s: %v", sessionID, err)
					o.broadcastError(sessionID, "Avatar generation failed")
				}
				o.avatarMu.Unlock()
				return
			}
			nf := int(chunk.GetNumFrames())
			if nf <= 0 && int(chunk.GetWidth())*int(chunk.GetHeight())*3 > 0 {
				nf = len(chunk.GetData()) / (int(chunk.GetWidth()) * int(chunk.GetHeight()) * 3)
			}
			fps := int(chunk.GetFps())
			if fps <= 0 {
				fps = 25
			}
			if !firstFrameSent {
				firstFrameSent = true
				log.Printf("TTFF std pipeline session=%s first_video_chunk=%.3fs", sessionID, time.Since(pipelineStart).Seconds())
				if !speakingBroadcasted {
					speakingBroadcasted = true
					session.SetState(StateSpeaking)
					o.broadcastStatus(sessionID, "speaking")
				}
			}
			if turnRec != nil {
				turnRec.WriteVideoChunk(chunk.GetData())
			}
			segVideo = append(segVideo, chunk.GetData()...)
			segFrames += nf
			segWidth = int(chunk.GetWidth())
			segHeight = int(chunk.GetHeight())
			segFPS = fps
			segCount++
			if segCount >= stdChunksPerSegment {
				flushStdSeg()
			}
		}
	}
}

// HandleAudioStream processes incoming user audio through the VoiceLLM pipeline.
func (o *Orchestrator) HandleAudioStream(ctx context.Context, sessionID string, audioCh <-chan []byte) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}

	pipeCtx, cancel := context.WithCancel(ctx)
	session.mu.Lock()
	session.PipelineCancel = cancel
	session.mu.Unlock()

	pipelineSeq := session.MarkPipelineRunning()
	go o.runVoiceLLMPipeline(pipeCtx, session, sessionID, wrapVoiceAudioInput(pipeCtx, audioCh), pipelineSeq)
	return nil
}

// runVoiceLLMPipeline executes a VoiceLLM turn source -> VoiceLLM -> Avatar (video).
//
// Serial flow: collect all audio for one turn, merge into a single chunk,
// generate avatar video, and publish each video chunk immediately.
func (o *Orchestrator) runVoiceLLMPipeline(ctx context.Context, session *Session, sessionID string, inputCh <-chan inference.VoiceLLMInputEvent, pipelineSeq uint64) {
	// Function-level message accumulators: track pending user/assistant text
	// so we can store them even when ctx is cancelled mid-turn.
	var pendingUserText string
	var pendingAssistantText string

	defer func() {
		// Store any pending messages before cleanup
		log.Printf("voiceLLM defer: session=%s pendingUser=%q pendingAssistant=%q historyLen=%d",
			sessionID, pendingUserText, pendingAssistantText, len(session.History))
		if pendingUserText != "" {
			session.AddMessage(ChatMessage{Role: "user", Content: pendingUserText})
		}
		if pendingAssistantText != "" {
			session.AddMessage(ChatMessage{Role: "assistant", Content: pendingAssistantText})
		}
		log.Printf("voiceLLM defer done: session=%s finalHistoryLen=%d", sessionID, len(session.History))
		session.MarkPipelineFinished(pipelineSeq)
		session.SetState(StateListening)
		o.broadcastStatus(sessionID, "idle")
	}()

	sessionDir := ""
	if o.recorder != nil {
		sessionDir = o.sessionRecordingDir(session)
	}
	var recTurnNum int

	session.SetState(StateProcessing)
	o.broadcastStatus(sessionID, "processing")

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	voiceConfig := o.buildVoiceLLMSessionConfig(session, sessionID)

	// outputCh stays open for the entire session; a single turn ends with
	// audio.IsFinal=true. The channel closes when the bot disconnects.
	outputCh, errCh := o.inference.ConverseStream(ctx, inputCh, voiceConfig)

	// Outer loop: one iteration = one complete assistant turn.
	for {
		if ctx.Err() != nil {
			return
		}

		// ── Phase 1: collect this turn's audio ──────────────────────────────
		var (
			audioChunks []*pb.AudioChunk
			recAudioBuf []byte
			recAudioSR  int
			recTurnID   string
			turnDone    bool
			sessionDone bool
		)
		// Reset pending text for this turn
		pendingUserText = ""
		pendingAssistantText = ""
	collectLoop:
		for {
			select {
			case <-ctx.Done():
				return
			case output, ok := <-outputCh:
				if !ok {
					sessionDone = true
					break collectLoop
				}
				payload := map[string]any{
					"type":     "transcript",
					"text":     output.GetTranscript(),
					"is_final": output.GetIsFinal(),
					"speaker":  "assistant",
				}
				if output.GetTranscript() != "" {
					o.broadcastJSON(sessionID, payload)
					// Track assistant transcript for history storage (function-level)
					pendingAssistantText = output.GetTranscript()
				}
				if output.GetUserTranscript() != "" {
					o.broadcastJSON(sessionID, map[string]any{
						"type":     "transcript",
						"text":     output.GetUserTranscript(),
						"is_final": true,
						"speaker":  "user",
					})
					// Track user transcript for history storage (function-level)
					pendingUserText = output.GetUserTranscript()
				}

				audio := output.GetAudio()
				hasA := audio != nil && (len(audio.GetData()) > 0 || audio.GetIsFinal())
				if o.recorder != nil && recTurnID == "" && (output.GetTranscript() != "" || hasA) {
					recTurnNum++
					recTurnID = fmt.Sprintf("turn%d", recTurnNum)
				}
				if !hasA {
					if output.GetIsFinal() {
						turnDone = true
						break collectLoop
					}
					continue
				}

				if recTurnID != "" && len(audio.GetData()) > 0 {
					recAudioBuf = append(recAudioBuf, audio.GetData()...)
					if recAudioSR == 0 {
						recAudioSR = int(audio.GetSampleRate())
					}
				}

				audioChunks = append(audioChunks, proto.Clone(audio).(*pb.AudioChunk))

				if audio.GetIsFinal() {
					turnDone = true
					if o.recorder != nil && recTurnID != "" {
						log.Printf("recording: SaveRawAudio session=%s turn=%s pcmLen=%d sampleRate=%d dir=%s",
							sessionID, recTurnID, len(recAudioBuf), recAudioSR, sessionDir)
						saveTurnID, savePCM, saveSR := recTurnID, recAudioBuf, recAudioSR
						saveDir := sessionDir
						go func() {
							if err := o.recorder.SaveRawAudio(saveDir, saveTurnID, savePCM, saveSR); err != nil {
								log.Printf("recording: SaveRawAudio error session=%s turn=%s: %v", sessionID, saveTurnID, err)
							}
						}()
					}
					break collectLoop
				}
			}
		}

		if turnDone && o.recorder != nil && recTurnID != "" && pendingAssistantText != "" {
			log.Printf("recording: SaveTranscript session=%s turn=%s chars=%d dir=%s",
				sessionID, recTurnID, len(pendingAssistantText), sessionDir)
			if err := o.recorder.SaveTranscript(sessionDir, recTurnID, pendingAssistantText); err != nil {
				log.Printf("recording: SaveTranscript error session=%s turn=%s: %v", sessionID, recTurnID, err)
			}
		}

		turnStart := time.Now()

		// Commit this turn's messages to session history and clear pending
		log.Printf("voiceLLM turn commit: session=%s pendingUser=%q pendingAssistant=%q",
			sessionID, pendingUserText, pendingAssistantText)
		if pendingUserText != "" {
			session.AddMessage(ChatMessage{Role: "user", Content: pendingUserText})
			pendingUserText = ""
		}
		if pendingAssistantText != "" {
			session.AddMessage(ChatMessage{Role: "assistant", Content: pendingAssistantText})
			pendingAssistantText = ""
		}

		if sessionDone {
			if err := <-errCh; err != nil {
				log.Printf("VoiceLLM stream error for session %s: %v", sessionID, err)
				o.broadcastError(sessionID, "Voice conversation failed")
			}
			return
		}
		if len(audioChunks) == 0 {
			continue
		}

		// ── Phase 2: merge audio + prepare syncBuf + generate video ─────────
		var mergedData []byte
		var mergedSR int32
		var mergedChannels int32
		var mergedFormat string
		for _, c := range audioChunks {
			mergedData = append(mergedData, c.GetData()...)
			if mergedSR == 0 {
				mergedSR = c.GetSampleRate()
				mergedChannels = c.GetChannels()
				mergedFormat = c.GetFormat()
			}
		}
		// Append trailing silence so the avatar closes its mouth naturally.
		sr := int(mergedSR)
		if sr <= 0 {
			sr = 16000
		}
		silenceBytes := make([]byte, sr*2*3/2) // 1.5 seconds of silence (s16le mono)
		mergedData = append(mergedData, silenceBytes...)

		mergedChunk := &pb.AudioChunk{
			Data:       mergedData,
			SampleRate: mergedSR,
			Channels:   mergedChannels,
			Format:     mergedFormat,
			IsFinal:    true,
		}

		syncBuf := newVoiceAVSyncBuffer(voiceMaxPCMBufferSamples)
		if dropped := syncBuf.appendPCM(mergedChunk.GetData(), int(mergedChunk.GetSampleRate())); dropped > 0 {
			bufferedSamples, _, _, _ := syncBuf.snapshot()
			log.Printf("voice sync buffer overflow for session %s: dropped=%d bytes, buffered_samples=%d", sessionID, dropped, bufferedSamples)
		}

		// Serialize with concurrent avatar operations (e.g. background EnsureIdleVideo)
		// to prevent deadlock on the Python-side FlashHead threading.Lock.
		// The FlashHead plugin runs inference on the asyncio event-loop thread when
		// world_size > 1; a second GenerateStream gRPC call arriving while the first
		// holds the plugin lock causes a permanent deadlock.
		o.avatarMu.Lock()
		videoCh, avatarErrCh := o.inference.GenerateAvatar(ctx, []*pb.AudioChunk{mergedChunk})

		// ── Phase 3: stream video chunks ────────────────────────────────────
		// Delay speaking status until first video frame arrives (avoids frozen-frame stall on frontend).
		speakingBroadcasted := false

		var turnRec *recording.TurnRecording

		var (
			segVideo            []byte
			segFrames           int
			segWidth            int
			segHeight           int
			segFPS              int
			segCount            int
			segSeq              int64
			cumulativeDriftMs   int64
			lastKnownSampleRate int
			firstFrameSent      bool
		)
		flushVoiceSeg := func(isFinalSeg bool) {
			if segCount == 0 {
				return
			}
			segSeq++
			o.mu.RLock()
			peer := o.peers[sessionID]
			o.mu.RUnlock()
			if peer != nil {
				traceLabel := "voice-trace session=" + sessionID + " seg=" + strconv.FormatInt(segSeq, 10)
				segPCM, outSamples, wantSamples, bufferedSamplesAfterTake := syncBuf.takeSegmentPCM(segFrames, segFPS, isFinalSeg)
				bufferedSamples, _, _, sampleRate := syncBuf.snapshot()
				if sampleRate > 0 {
					lastKnownSampleRate = sampleRate
				} else {
					sampleRate = lastKnownSampleRate
				}
				audioSpanMs := 0
				videoSpanMs := 0
				if sampleRate > 0 {
					audioSpanMs = outSamples * 1000 / sampleRate
				}
				if segFPS > 0 {
					videoSpanMs = segFrames * 1000 / segFPS
				}
				deltaMs := audioSpanMs - videoSpanMs
				cumulativeDriftMs += int64(deltaMs)
				if outSamples < wantSamples {
					log.Printf("voice av drift for session %s: out_samples=%d want_samples=%d frames=%d fps=%d buffered_samples=%d",
						sessionID, outSamples, wantSamples, segFrames, segFPS, bufferedSamples)
					if sampleRate <= 0 {
						sampleRate = 16000
					}
					log.Printf("voice av strict pad session=%s: %d→%d samples (%.3fs silence added)",
						sessionID, outSamples, wantSamples,
						float64(wantSamples-outSamples)/float64(sampleRate))
				}
				if isFinalSeg && bufferedSamplesAfterTake > 0 {
					log.Printf("voice av strict trim tail session=%s: trimmed_samples=%d", sessionID, bufferedSamplesAfterTake)
				}
				// Send to AV pipeline (non-blocking encode+publish).
				raw := &mediapeer.RawAVSegment{
					TraceLabel: traceLabel,
					RGB:        segVideo,
					PCM:        segPCM,
					SampleRate: sampleRate,
					Width:      segWidth,
					Height:     segHeight,
					FPS:        segFPS,
					NumFrames:  segFrames,
				}
				if err := peer.SendAVSegment(raw); err != nil {
					log.Printf("voice av SendAVSegment failed session=%s seg=%d: %v", sessionID, segSeq, err)
				}
				if turnRec != nil {
					turnRec.WriteVideoChunk(segVideo)
					turnRec.WriteAudioChunk(segPCM, sampleRate)
				}
			}
			segVideo = nil
			segFrames = 0
			segCount = 0
		}

	videoLoop:
		for {
			select {
			case <-ctx.Done():
				flushVoiceSeg(false)
				if turnRec != nil {
					_ = turnRec.Finish()
				}
				o.avatarMu.Unlock()
				return
			case chunk, ok := <-videoCh:
				if !ok {
					flushVoiceSeg(false)
					if turnRec != nil {
						_ = turnRec.Finish()
						turnRec = nil
					}
					if remain, totalIn, totalOut, _ := syncBuf.snapshot(); remain > 0 {
						log.Printf("voice sync tail flush for session %s: dropping_unaligned_samples=%d total_in=%d total_out=%d", sessionID, remain, totalIn, totalOut)
					}
					if err := <-avatarErrCh; err != nil {
						log.Printf("Avatar stream error for session %s (voice_llm): %v", sessionID, err)
						o.broadcastError(sessionID, "Avatar generation failed")
					}
					break videoLoop
				}
				nf := int(chunk.GetNumFrames())
				if nf <= 0 && int(chunk.GetWidth())*int(chunk.GetHeight())*3 > 0 {
					nf = len(chunk.GetData()) / (int(chunk.GetWidth()) * int(chunk.GetHeight()) * 3)
				}
				fps := int(chunk.GetFps())
				if fps <= 0 {
					fps = 20 // FlashHead 固定输出 20fps
				}
				if !firstFrameSent {
					firstFrameSent = true
					log.Printf("TTFF voice pipeline session=%s first_video_chunk=%.3fs", sessionID, time.Since(turnStart).Seconds())
					if !speakingBroadcasted {
						speakingBroadcasted = true
						session.SetState(StateSpeaking)
						o.broadcastStatus(sessionID, "speaking")
					}
				}
				// 懒初始化录制：使用第一个 chunk 的实际 FPS 和尺寸
				if turnRec == nil && o.recorder != nil && nf > 0 {
					log.Printf("recording: BeginTurn session=%s turn=turn%d width=%d height=%d fps=%d dir=%s",
						sessionID, recTurnNum, chunk.GetWidth(), chunk.GetHeight(), fps, sessionDir)
					turnRec = o.recorder.BeginTurn(sessionDir, fmt.Sprintf("turn%d", recTurnNum), int(chunk.GetWidth()), int(chunk.GetHeight()), fps)
				}
				segVideo = append(segVideo, chunk.GetData()...)
				segFrames += nf
				segWidth = int(chunk.GetWidth())
				segHeight = int(chunk.GetHeight())
				segFPS = fps
				segCount++
				// Serial path: flush every chunk immediately.
				flushVoiceSeg(chunk.GetIsFinal())
				if chunk.GetIsFinal() {
					if turnRec != nil {
						_ = turnRec.Finish()
						turnRec = nil
					}
				}
			}
		}

		o.avatarMu.Unlock()

		// Wait for the peer's AV pipeline to finish publishing all queued
		// segments (encoding + real-time paced output) before changing status.
		// Without this, the frontend switches to idle video while the last
		// few seconds of speech are still being delivered via WebRTC.
		o.mu.RLock()
		peer := o.peers[sessionID]
		o.mu.RUnlock()
		if peer != nil {
			peer.WaitAVDrain(10 * time.Second)
		}

		// Turn done, go back to processing for the next turn.
		session.SetState(StateProcessing)
		o.broadcastStatus(sessionID, "processing")
	}
}

// Interrupt cancels the current pipeline for a session.
func (o *Orchestrator) Interrupt(sessionID string) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}

	o.cancelPipeline(session)

	// Also interrupt VoiceLLM on the inference side
	if session.Mode == ModeVoiceLLM {
		_ = o.inference.Interrupt(context.Background(), sessionID)
	}

	session.SetState(StateListening)
	o.broadcastStatus(sessionID, "idle")
	return nil
}

// TeardownSession cleans up all resources for a session.
func (o *Orchestrator) TeardownSession(sessionID string) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}

	o.cancelPipeline(session)

	// Wait for pipeline goroutine to finish storing messages (up to 3s)
	session.WaitPipelineDone(3 * time.Second)

	// Disconnect media peer
	o.mu.Lock()
	peer, ok := o.peers[sessionID]
	if ok {
		delete(o.peers, sessionID)
	}
	delete(o.directPeers, sessionID)
	o.mu.Unlock()

	if peer != nil {
		peer.StopAVPipeline()
		_ = peer.Disconnect()
	}

	// Close WebSocket connections
	o.wsHub.CloseSession(sessionID)

	session.SetState(StateClosed)
	return nil
}

// TeardownAll cleans up all sessions. Called during server shutdown.
func (o *Orchestrator) TeardownAll() {
	o.mu.Lock()
	peers := make(map[string]mediapeer.MediaPeer, len(o.peers))
	for k, v := range o.peers {
		peers[k] = v
	}
	o.peers = make(map[string]mediapeer.MediaPeer)
	o.directPeers = make(map[string]*direct.DirectPeer)
	o.mu.Unlock()

	for _, peer := range peers {
		peer.StopAVPipeline()
		_ = peer.Disconnect()
	}
}

// cancelPipeline cancels the active pipeline for a session if one exists.
func (o *Orchestrator) cancelPipeline(session *Session) {
	session.mu.Lock()
	cancel := session.PipelineCancel
	session.PipelineCancel = nil
	session.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}

// broadcastStatus sends an avatar_status message to all WebSocket clients.
func (o *Orchestrator) broadcastStatus(sessionID, status string) {
	o.broadcastJSON(sessionID, map[string]string{
		"type":   "avatar_status",
		"status": status,
	})
}

// broadcastError sends an error message to all WebSocket clients.
func (o *Orchestrator) broadcastError(sessionID, message string) {
	o.broadcastJSON(sessionID, map[string]string{
		"type":    "error",
		"message": message,
	})
}

// sessionRecordingDir returns the directory for recording output.
// If the session has a character, records go into the character's sessions/ dir.
// Otherwise falls back to a timestamp-based dir (used by the recorder's OutputDir).
func (o *Orchestrator) sessionRecordingDir(session *Session) string {
	if session.CharacterID != "" && o.charStore != nil {
		dir := o.charStore.SessionRecordingDir(session.CharacterID, session.ID, session.CreatedAt)
		if dir != "" {
			session.mu.Lock()
			session.RecordingDir = dir
			session.mu.Unlock()
			return dir
		}
	}
	// Fallback: legacy timestamp-based dir
	return time.Now().Format("20060102-150405")
}

// broadcastJSON marshals and broadcasts a JSON message.
func (o *Orchestrator) broadcastJSON(sessionID string, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		log.Printf("Failed to marshal broadcast: %v", err)
		return
	}
	o.wsHub.Broadcast(sessionID, data)
}
