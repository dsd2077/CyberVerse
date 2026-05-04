package orchestrator

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// stdChunksPerSegment is how many avatar video chunks to batch before publishing
// in the standard (ASR/text→LLM→TTS→Avatar) pipeline. Qwen TTS produces audio
// before avatar video, so each avatar chunk is sent as its own paced AV segment
// with matching PCM instead of publishing audio ahead of video.
const stdChunksPerSegment = 1

// No hard cap on the assistant PCM buffer: long responses (>20s) were
// previously truncated, causing the first N seconds of audio to be dropped
// and all video segments to play with misaligned (or silent) audio.
// Set to 0 to disable the overflow guard entirely.
const voiceMaxPCMBufferSamples = 0

const avatarImageMaxUploadHint = "角色头像图片超过当前 10MB 上传限制，已使用默认头像；待机视频也不会生成。请压缩或缩放角色图片到 10MB 以内后重试。"

const (
	doubaoDialogContextMaxPairs  = 20
	doubaoDialogContextLoadLimit = doubaoDialogContextMaxPairs * 4
)

var (
	ErrVisualInputUnsupported = errors.New("visual input is only supported in standard sessions")
	ErrVisualInputDisabled    = errors.New("visual input is disabled")
)

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

type voicePipelineTurn struct {
	seq                 uint64
	key                 string
	questionID          string
	replyID             string
	assistantText       string
	recTurnID           string
	recAudioBuf         []byte
	recAudioSR          int
	historySaved        bool
	conversationSaved   bool
	transcriptSaved     bool
	rawAudioSaved       bool
	sessionDir          string
	turnStart           time.Time
	userFinalAt         time.Time
	firstAudioAt        time.Time
	audioFinalAt        time.Time
	avatarWorkerAt      time.Time
	firstAvatarAudioAt  time.Time
	avatarInputClosedAt time.Time
	firstVideoAt        time.Time
	syncBuf             *voiceAVSyncBuffer
	avatarStarted       bool
	avatarInputClosed   bool
	avatarAudioCh       chan *pb.AudioChunk
	avatarCtx           context.Context
	avatarCancel        context.CancelFunc
	doneCh              chan voicePipelineTurnResult
	aborted             bool
}

type voicePipelineTurnResult struct {
	turn *voicePipelineTurn
	err  error
}

func voiceOutputTurnKey(output *pb.VoiceLLMOutput) string {
	if output == nil {
		return ""
	}
	if replyID := strings.TrimSpace(output.GetReplyId()); replyID != "" {
		return "reply:" + replyID
	}
	if questionID := strings.TrimSpace(output.GetQuestionId()); questionID != "" {
		return "question:" + questionID
	}
	return ""
}

func voiceOutputHasAssistantContent(output *pb.VoiceLLMOutput) bool {
	if output == nil {
		return false
	}
	if output.GetTranscript() != "" {
		return true
	}
	audio := output.GetAudio()
	return audio != nil && len(audio.GetData()) > 0
}

func voiceOutputIsFinal(output *pb.VoiceLLMOutput) bool {
	if output == nil {
		return false
	}
	if output.GetIsFinal() {
		return true
	}
	audio := output.GetAudio()
	return audio != nil && audio.GetIsFinal()
}

type dialogContextMessage struct {
	sessionID string
	role      string
	text      string
	timestamp time.Time
}

func stringValue(v any) string {
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}

func unixTimeFromNumber(n int64) time.Time {
	if n <= 0 {
		return time.Time{}
	}
	if n > 1_000_000_000_000 {
		return time.UnixMilli(n).UTC()
	}
	return time.Unix(n, 0).UTC()
}

func parseConversationTimestamp(v any) time.Time {
	switch t := v.(type) {
	case string:
		t = strings.TrimSpace(t)
		if t == "" {
			return time.Time{}
		}
		if parsed, err := time.Parse(time.RFC3339Nano, t); err == nil {
			return parsed.UTC()
		}
		if parsed, err := time.Parse(time.RFC3339, t); err == nil {
			return parsed.UTC()
		}
		if n, err := strconv.ParseInt(t, 10, 64); err == nil {
			return unixTimeFromNumber(n)
		}
	case float64:
		return unixTimeFromNumber(int64(t))
	case int:
		return unixTimeFromNumber(int64(t))
	case int64:
		return unixTimeFromNumber(t)
	case json.Number:
		if n, err := t.Int64(); err == nil {
			return unixTimeFromNumber(n)
		}
	}
	return time.Time{}
}

func buildDoubaoDialogContext(messages []map[string]any, maxPairs int, now time.Time) []DialogContextItem {
	if maxPairs <= 0 {
		maxPairs = doubaoDialogContextMaxPairs
	}
	if now.IsZero() {
		now = time.Now().UTC()
	} else {
		now = now.UTC()
	}

	filtered := make([]dialogContextMessage, 0, len(messages))
	for _, msg := range messages {
		role := strings.ToLower(stringValue(msg["role"]))
		if role != "user" && role != "assistant" {
			continue
		}
		text := stringValue(msg["content"])
		if text == "" {
			text = stringValue(msg["text"])
		}
		if text == "" {
			continue
		}
		sessionID := stringValue(msg["session_id"])
		if sessionID == "" {
			continue
		}
		filtered = append(filtered, dialogContextMessage{
			sessionID: sessionID,
			role:      role,
			text:      text,
			timestamp: parseConversationTimestamp(msg["timestamp"]),
		})
	}

	paired := make([]dialogContextMessage, 0, len(filtered))
	var pendingUsers []dialogContextMessage
	pendingSessionID := ""
	for _, msg := range filtered {
		if len(pendingUsers) > 0 && msg.sessionID != pendingSessionID {
			pendingUsers = nil
			pendingSessionID = ""
		}
		if msg.role == "user" {
			if len(pendingUsers) == 0 {
				pendingSessionID = msg.sessionID
			}
			pendingUsers = append(pendingUsers, msg)
			continue
		}
		if len(pendingUsers) == 0 || msg.sessionID != pendingSessionID {
			continue
		}

		var merged strings.Builder
		for i, userMsg := range pendingUsers {
			if i > 0 {
				merged.WriteString("\n")
			}
			merged.WriteString(userMsg.text)
		}
		paired = append(paired, dialogContextMessage{
			sessionID: pendingSessionID,
			role:      "user",
			text:      merged.String(),
			timestamp: pendingUsers[0].timestamp,
		}, msg)
		pendingUsers = nil
		pendingSessionID = ""
	}

	maxItems := maxPairs * 2
	if len(paired) > maxItems {
		paired = paired[len(paired)-maxItems:]
	}
	if len(paired) == 0 {
		return nil
	}

	items := make([]DialogContextItem, len(paired))
	fallbackStart := now.Add(-time.Duration(len(paired)) * time.Millisecond)
	for i, msg := range paired {
		ts := msg.timestamp
		if ts.IsZero() || ts.After(now) {
			ts = fallbackStart.Add(time.Duration(i) * time.Millisecond)
		}
		items[i] = DialogContextItem{
			Role:      msg.role,
			Text:      msg.text,
			Timestamp: ts.UnixMilli(),
		}
	}

	var last int64
	for i := range items {
		if items[i].Timestamp <= last {
			items[i].Timestamp = last + 1
		}
		last = items[i].Timestamp
	}
	if nowMS := now.UnixMilli(); last > nowMS {
		delta := last - nowMS
		for i := range items {
			items[i].Timestamp -= delta
		}
	}
	return items
}

func safeTraceValue(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "-"
	}
	replacer := strings.NewReplacer(" ", "_", "\t", "_", "\n", "_", "\r", "_")
	return replacer.Replace(value)
}

func voiceTraceLabel(sessionID string, turnSeq uint64, replyID, questionID string, segSeq int64) string {
	parts := []string{
		"sid=" + safeTraceValue(sessionID),
		"turn=" + strconv.FormatUint(turnSeq, 10),
		"reply=" + safeTraceValue(replyID),
	}
	if questionID != "" {
		parts = append(parts, "qid="+safeTraceValue(questionID))
	}
	if segSeq > 0 {
		parts = append(parts, "seg="+strconv.FormatInt(segSeq, 10))
	}
	return strings.Join(parts, " ")
}

func characterSystemPrompt(char *character.Character, includeName bool, includeSpeakingStyle bool) string {
	if char == nil {
		return ""
	}
	var parts []string
	appendField := func(label, value string) {
		value = strings.TrimSpace(value)
		if value != "" {
			parts = append(parts, label+"："+value)
		}
	}

	appendField("系统提示", char.SystemPrompt)
	if includeName {
		appendField("角色名称", char.Name)
	}
	appendField("角色描述", char.Description)
	appendField("角色性格", char.Personality)
	if includeSpeakingStyle {
		appendField("说话风格", char.SpeakingStyle)
	}
	return strings.Join(parts, "\n")
}

func formatSinceUserFinal(start time.Time) string {
	if start.IsZero() {
		return "-"
	}
	return strconv.FormatInt(time.Since(start).Milliseconds(), 10)
}

func logVoiceTrace(event, sessionID string, turnSeq uint64, replyID, questionID string, since time.Time, fields ...string) {
	parts := []string{
		fmt.Sprintf("voice_trace event=%-30s", event),
		"sid=" + safeTraceValue(sessionID),
		"turn=" + strconv.FormatUint(turnSeq, 10),
		"reply=" + safeTraceValue(replyID),
	}
	parts = append(parts, "qid="+safeTraceValue(questionID))
	parts = append(parts, "since_user_final_ms="+formatSinceUserFinal(since))
	parts = append(parts, fields...)
	log.Print(strings.Join(parts, " "))
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

func normalizedVisualInputConfig(cfg config.VisualInputConfig) config.VisualInputConfig {
	if cfg.FrameIntervalMS == 0 {
		cfg.FrameIntervalMS = 1000
	}
	if cfg.MaxWidth == 0 {
		cfg.MaxWidth = 1280
	}
	if cfg.MaxHeight == 0 {
		cfg.MaxHeight = 720
	}
	if cfg.MaxFrameBytes == 0 {
		cfg.MaxFrameBytes = 512 * 1024
	}
	if cfg.MaxRecentFrames == 0 {
		cfg.MaxRecentFrames = 2
	}
	if cfg.FrameTTLMS == 0 {
		cfg.FrameTTLMS = 10000
	}
	return cfg
}

func (o *Orchestrator) visualInputConfig() config.VisualInputConfig {
	if o == nil {
		return normalizedVisualInputConfig(config.VisualInputConfig{})
	}
	return normalizedVisualInputConfig(o.pipelineCfg.VisualInput)
}

func validateVisualSource(source string) error {
	switch source {
	case "camera", "screen":
		return nil
	default:
		return fmt.Errorf("invalid visual source")
	}
}

func (o *Orchestrator) visualSession(sessionID string) (*Session, config.VisualInputConfig, error) {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return nil, config.VisualInputConfig{}, err
	}
	if session.Mode != ModeStandard {
		return nil, config.VisualInputConfig{}, ErrVisualInputUnsupported
	}
	cfg := o.visualInputConfig()
	if !cfg.IsEnabled() {
		return nil, cfg, ErrVisualInputDisabled
	}
	return session, cfg, nil
}

func (o *Orchestrator) HandleVisualInputStart(sessionID string, source string) error {
	if err := validateVisualSource(source); err != nil {
		return err
	}
	session, _, err := o.visualSession(sessionID)
	if err != nil {
		return err
	}
	session.StartVisualInput(source)
	return nil
}

func (o *Orchestrator) HandleVisualInputStop(sessionID string, source string) error {
	if source != "" {
		if err := validateVisualSource(source); err != nil {
			return err
		}
	}
	session, _, err := o.visualSession(sessionID)
	if err != nil {
		return err
	}
	session.StopVisualInput(source)
	return nil
}

func (o *Orchestrator) HandleVisualFrame(sessionID string, msg ws.WSMessage) error {
	if err := validateVisualSource(msg.Source); err != nil {
		return err
	}
	session, cfg, err := o.visualSession(sessionID)
	if err != nil {
		return err
	}
	if msg.Mime != "image/jpeg" {
		return fmt.Errorf("invalid visual frame mime")
	}
	if msg.Width <= 0 || msg.Height <= 0 || int(msg.Width) > cfg.MaxWidth || int(msg.Height) > cfg.MaxHeight {
		return fmt.Errorf("invalid visual frame dimensions")
	}
	encoded := strings.TrimSpace(msg.Data)
	if encoded == "" {
		return fmt.Errorf("visual frame data is required")
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return fmt.Errorf("invalid visual frame data")
	}
	if len(decoded) == 0 || len(decoded) > cfg.MaxFrameBytes {
		return fmt.Errorf("visual frame exceeds size limit")
	}
	if len(decoded) < 3 || decoded[0] != 0xff || decoded[1] != 0xd8 || decoded[2] != 0xff {
		return fmt.Errorf("invalid visual frame jpeg data")
	}

	now := time.Now()
	frame := VisualFrame{
		Data:        decoded,
		MimeType:    msg.Mime,
		Width:       msg.Width,
		Height:      msg.Height,
		Source:      msg.Source,
		TimestampMS: msg.TimestampMS,
		FrameSeq:    msg.FrameSeq,
	}
	minInterval := time.Duration(cfg.FrameIntervalMS) * time.Millisecond
	session.StoreVisualFrame(frame, cfg.MaxRecentFrames, minInterval, now)
	return nil
}

func (o *Orchestrator) CheckVoice(ctx context.Context, provider string, voiceType string) (string, error) {
	if o == nil || o.inference == nil {
		return "", errors.New("inference service is not configured")
	}
	return o.inference.CheckVoice(ctx, inference.VoiceLLMSessionConfig{
		Provider: voiceLLMProviderOrDefault(provider),
		Voice:    voiceType,
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

func buildDefaultAvatarPNG(width, height int) ([]byte, error) {
	if width <= 0 {
		width = 512
	}
	if height <= 0 {
		height = 512
	}
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{128, 128, 128, 255}}, image.Point{}, draw.Src)

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// AvatarSetupWarning converts avatar setup failures into a concise message
// suitable for browser console diagnostics.
func AvatarSetupWarning(err error) string {
	if err == nil {
		return ""
	}
	if warning, ok := AvatarImageTooLargeWarning(err); ok {
		return warning
	}
	return fmt.Sprintf("角色头像设置失败，已使用默认头像：%v", err)
}

// AvatarImageTooLargeWarning reports whether a gRPC message-size failure was
// caused by an oversized avatar image.
func AvatarImageTooLargeWarning(err error) (string, bool) {
	if err == nil {
		return "", false
	}
	msg := strings.ToLower(err.Error())
	if status.Code(err) == codes.ResourceExhausted && strings.Contains(msg, "message larger than max") {
		return avatarImageMaxUploadHint, true
	}
	if strings.Contains(msg, "trying to send message larger than max") {
		return avatarImageMaxUploadHint, true
	}
	return "", false
}

func (o *Orchestrator) setDefaultAvatarLocked(ctx context.Context, sessionID string) error {
	if o == nil || o.inference == nil {
		return errors.New("inference service is not configured")
	}
	width, height := 512, 512
	if info, err := o.inference.AvatarInfo(ctx); err == nil && info != nil {
		if int(info.GetOutputWidth()) > 0 {
			width = int(info.GetOutputWidth())
		}
		if int(info.GetOutputHeight()) > 0 {
			height = int(info.GetOutputHeight())
		}
	}
	imageData, err := buildDefaultAvatarPNG(width, height)
	if err != nil {
		return fmt.Errorf("build default avatar image: %w", err)
	}
	if err := o.inference.SetAvatar(ctx, sessionID, imageData, "png"); err != nil {
		return fmt.Errorf("set default avatar: %w", err)
	}
	return nil
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

func audioChunkToPCM16(chunk *pb.AudioChunk) ([]byte, int) {
	if chunk == nil || len(chunk.GetData()) == 0 {
		return nil, 0
	}
	sampleRate := int(chunk.GetSampleRate())
	format := strings.ToLower(strings.TrimSpace(chunk.GetFormat()))
	data := chunk.GetData()

	switch format {
	case "float32", "f32", "pcm_f32le":
		n := len(data) / 4
		if n <= 0 {
			return nil, sampleRate
		}
		out := make([]byte, n*2)
		for i := 0; i < n; i++ {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				v = 0
			}
			if v > 1 {
				v = 1
			} else if v < -1 {
				v = -1
			}
			var sample int16
			if v >= 0 {
				sample = int16(v * 32767)
			} else {
				sample = int16(v * 32768)
			}
			binary.LittleEndian.PutUint16(out[i*2:], uint16(sample))
		}
		return out, sampleRate
	default:
		if len(data)%2 != 0 {
			data = data[:len(data)-1]
		}
		if len(data) == 0 {
			return nil, sampleRate
		}
		out := make([]byte, len(data))
		copy(out, data)
		return out, sampleRate
	}
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

	if err := o.inference.SetAvatar(ctx, sessionID, imageData, format); err != nil {
		if resetErr := o.setDefaultAvatarLocked(ctx, sessionID); resetErr != nil {
			return fmt.Errorf("set avatar from image %q (%d bytes): %w; default avatar reset failed: %v", imageFilename, len(imageData), err, resetErr)
		}
		log.Printf("SetAvatar failed for image %q (%d bytes); reset inference avatar to default placeholder", imageFilename, len(imageData))
		return fmt.Errorf("set avatar from image %q (%d bytes): %w", imageFilename, len(imageData), err)
	}
	return nil
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
		if resetErr := o.setDefaultAvatarLocked(ctx, jobID); resetErr != nil {
			log.Printf("EnsureIdleVideo: failed to reset default avatar after SetAvatar failure for character %s image=%s: %v", characterID, imageFilename, resetErr)
		}
		return "", fmt.Errorf("set avatar for idle video from image %q (%d bytes): %w", imageFilename, len(imageData), err)
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
func (o *Orchestrator) SetupSession(ctx context.Context, session *Session, roomMgr *livekit.RoomManager) (mediapeer.MediaPeer, []string, error) {
	warnings := []string{}

	// Best-effort: apply the character's active avatar image.
	if session != nil && session.CharacterID != "" {
		_, imageFilename, err := o.activeCharacterImage(session.CharacterID)
		if err != nil {
			log.Printf("SetupSession: could not resolve active image for character %s: %v", session.CharacterID, err)
		} else if imageFilename != "" {
			if err := o.setAvatarFromCharacterImage(ctx, session.ID, session.CharacterID, imageFilename); err != nil {
				warning := AvatarSetupWarning(err)
				warnings = append(warnings, warning)
				log.Printf("SetupSession: %s character=%s image=%s details=%v", warning, session.CharacterID, imageFilename, err)
			}
		}
	}

	var peer mediapeer.MediaPeer

	if o.streamingMode == "livekit" {
		// LiveKit SFU mode
		roomName := livekit.RoomName(session.ID)
		if err := roomMgr.CreateRoom(ctx, roomName); err != nil {
			return nil, warnings, err
		}

		bot := livekit.NewBot(
			roomMgr.URL(),
			roomMgr.APIKey(),
			roomMgr.APISecret(),
			roomName,
		)
		if err := bot.Connect(ctx); err != nil {
			return nil, warnings, err
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
			return nil, warnings, err
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
	return peer, warnings, nil
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

func (o *Orchestrator) HydrateVoiceDialogContext(session *Session) error {
	if o == nil || o.charStore == nil || session == nil {
		return nil
	}
	if session.Mode != ModeVoiceLLM || session.CharacterID == "" {
		return nil
	}
	messages, _, _, err := o.charStore.LoadRecentMessages(session.CharacterID, "", doubaoDialogContextLoadLimit)
	if err != nil {
		return err
	}
	session.SetDialogContext(buildDoubaoDialogContext(messages, doubaoDialogContextMaxPairs, time.Now().UTC()))
	return nil
}

func (o *Orchestrator) buildVoiceLLMSessionConfig(session *Session, sessionID string) inference.VoiceLLMSessionConfig {
	voiceConfig := inference.VoiceLLMSessionConfig{SessionID: sessionID}
	if session.CharacterID != "" && o.charStore != nil {
		if char, err := o.charStore.Get(session.CharacterID); err == nil {
			voiceConfig.Provider = voiceLLMProviderOrDefault(char.VoiceProvider)
			voiceConfig.SystemPrompt = char.SystemPrompt
			voiceConfig.Voice = char.VoiceType
			voiceConfig.BotName = char.Name
			voiceConfig.SpeakingStyle = char.SpeakingStyle
			voiceConfig.WelcomeMessage = session.ConsumeVoiceWelcomeMessage(char.WelcomeMessage)
		} else {
			log.Printf("buildVoiceLLMSessionConfig: could not fetch character %s: %v", session.CharacterID, err)
		}
	}
	for _, item := range session.DialogContextSnapshot() {
		voiceConfig.DialogContext = append(voiceConfig.DialogContext, inference.VoiceLLMDialogContextItem{
			Role:      item.Role,
			Text:      item.Text,
			Timestamp: item.Timestamp,
		})
	}
	return voiceConfig
}

func voiceLLMProviderOrDefault(provider string) string {
	provider = strings.ToLower(strings.TrimSpace(provider))
	switch provider {
	case "qwen_omni":
		return "qwen_omni"
	default:
		return "doubao"
	}
}

func (o *Orchestrator) standardComponentDefaults() character.Components {
	defaults := character.Components{LLM: "qwen", ASR: "qwen", TTS: "qwen"}
	if o.pipelineCfg.DefaultLLM != "" {
		defaults.LLM = o.pipelineCfg.DefaultLLM
	}
	if o.pipelineCfg.DefaultASR != "" {
		defaults.ASR = o.pipelineCfg.DefaultASR
	}
	if o.pipelineCfg.DefaultTTS != "" {
		defaults.TTS = o.pipelineCfg.DefaultTTS
	}
	return defaults
}

func (o *Orchestrator) standardCharacterConfig(session *Session) (character.Components, string, string, string) {
	components := character.NormalizeComponents(character.Components{}, o.standardComponentDefaults())
	voice := ""
	speakingStyle := ""
	language := ""

	if session.CharacterID != "" && o.charStore != nil {
		if char, err := o.charStore.Get(session.CharacterID); err == nil {
			components = character.NormalizeComponents(char.Components, components)
			voice = strings.TrimSpace(char.VoiceType)
			speakingStyle = strings.TrimSpace(char.SpeakingStyle)
		} else {
			log.Printf("standardCharacterConfig: could not fetch character %s: %v", session.CharacterID, err)
		}
	}

	if voice == "" && components.TTS == "qwen" {
		voice = "Momo"
	}
	return components, voice, speakingStyle, language
}

func (o *Orchestrator) standardSystemPrompt(session *Session) string {
	if session.CharacterID == "" || o.charStore == nil {
		return ""
	}
	char, err := o.charStore.Get(session.CharacterID)
	if err != nil {
		log.Printf("standardSystemPrompt: could not fetch character %s: %v", session.CharacterID, err)
		return ""
	}
	return characterSystemPrompt(char, true, true)
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

	turnSeq := session.MarkTurnStarted()
	o.advancePlaybackEpoch(sessionID, turnSeq)
	session.AddMessage(ChatMessage{Role: "user", Content: text, TurnSeq: turnSeq})
	pipelineSeq := session.MarkPipelineRunning()
	go o.runStandardPipeline(pipeCtx, session, sessionID, pipelineSeq, turnSeq)
	return nil
}

func (o *Orchestrator) handleVoiceLLMTextInput(ctx context.Context, session *Session, sessionID string, text string) error {
	turnSeq := session.MarkTurnStarted()
	o.advancePlaybackEpoch(sessionID, turnSeq)
	if o.inference != nil {
		if err := o.inference.Interrupt(context.Background(), sessionID); err != nil {
			log.Printf("Failed to interrupt VoiceLLM for session %s: %v", sessionID, err)
		}
	}
	o.cancelPipeline(session)

	pipeCtx, cancel := context.WithCancel(ctx)
	session.mu.Lock()
	session.PipelineCancel = cancel
	session.mu.Unlock()

	session.AddMessage(ChatMessage{Role: "user", Content: text, TurnSeq: turnSeq})
	session.SetState(StateProcessing)
	o.broadcastStatusTurn(sessionID, "processing", turnSeq)
	pipelineSeq := session.MarkPipelineRunning()
	inputCh := singleVoiceTextInput(text)

	go func(seq uint64) {
		o.runVoiceLLMPipeline(pipeCtx, session, sessionID, inputCh, seq, turnSeq)
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
func (o *Orchestrator) runStandardPipeline(ctx context.Context, session *Session, sessionID string, pipelineSeq uint64, turnSeq uint64) {
	var fullResponseCh chan string // set below; read in defer to store assistant message
	recSessionDir := ""
	recTurnID := "turn" + strconv.FormatUint(turnSeq, 10)
	var recMu sync.Mutex
	var recAudioBuf []byte
	var recAudioSR int
	var turnRec *recording.TurnRecording
	var recFinished bool
	if o.recorder != nil {
		recSessionDir = o.sessionRecordingDir(session)
	}

	appendRecAudio := func(pcm []byte, sampleRate int) {
		if o.recorder == nil || len(pcm) == 0 || sampleRate <= 0 {
			return
		}
		pcmCopy := append([]byte(nil), pcm...)
		recMu.Lock()
		if recAudioSR == 0 {
			recAudioSR = sampleRate
		}
		recAudioBuf = append(recAudioBuf, pcmCopy...)
		activeRec := turnRec
		if activeRec != nil && !recFinished {
			activeRec.WriteAudioChunk(pcmCopy, sampleRate)
		}
		recMu.Unlock()
	}

	beginRec := func(width, height, fps int) {
		if o.recorder == nil || width <= 0 || height <= 0 || fps <= 0 {
			return
		}
		recMu.Lock()
		if turnRec != nil || recFinished {
			recMu.Unlock()
			return
		}
		activeRec := o.recorder.BeginTurn(recSessionDir, recTurnID, width, height, fps)
		turnRec = activeRec
		audioCopy := append([]byte(nil), recAudioBuf...)
		audioSR := recAudioSR
		if activeRec != nil && len(audioCopy) > 0 && audioSR > 0 {
			activeRec.WriteAudioChunk(audioCopy, audioSR)
		}
		recMu.Unlock()
	}

	writeRecVideo := func(rgb []byte) {
		if o.recorder == nil || len(rgb) == 0 {
			return
		}
		recMu.Lock()
		activeRec := turnRec
		if activeRec != nil && !recFinished {
			activeRec.WriteVideoChunk(rgb)
		}
		recMu.Unlock()
	}

	finishRec := func() {
		if o.recorder == nil {
			return
		}
		recMu.Lock()
		if recFinished {
			recMu.Unlock()
			return
		}
		activeRec := turnRec
		turnRec = nil
		recFinished = true
		recMu.Unlock()
		if activeRec != nil {
			_ = activeRec.Finish()
		}
	}

	defer func() {
		// Store assistant message in session history
		assistantResp := ""
		if fullResponseCh != nil {
			if resp, ok := <-fullResponseCh; ok && resp != "" {
				assistantResp = resp
				session.AddMessage(ChatMessage{Role: "assistant", Content: resp, TurnSeq: turnSeq})
				if _, err := o.persistSessionConversation(session); err != nil {
					log.Printf("conversation: SaveConversation error session=%s: %v", sessionID, err)
				}
			}
		}
		if o.recorder != nil {
			recMu.Lock()
			audioCopy := append([]byte(nil), recAudioBuf...)
			audioSR := recAudioSR
			recMu.Unlock()
			if len(audioCopy) > 0 && audioSR > 0 {
				if err := o.recorder.SaveRawAudio(recSessionDir, recTurnID, audioCopy, audioSR); err != nil {
					log.Printf("recording: SaveRawAudio error session=%s turn=%s: %v", sessionID, recTurnID, err)
				}
			}
			if strings.TrimSpace(assistantResp) != "" {
				if err := o.recorder.SaveTranscript(recSessionDir, recTurnID, assistantResp); err != nil {
					log.Printf("recording: SaveTranscript error session=%s turn=%s: %v", sessionID, recTurnID, err)
				}
			}
		}
		session.MarkPipelineFinished(pipelineSeq)
		session.SetState(StateListening)
		o.broadcastStatus(sessionID, "idle")
	}()

	session.SetState(StateProcessing)
	o.broadcastStatus(sessionID, "processing")

	pipelineStart := time.Now()

	components, voice, speakingStyle, language := o.standardCharacterConfig(session)

	// Prepare LLM messages
	history := session.HistorySnapshot()
	messages := make([]inference.ChatMessage, 0, len(history)+1)
	if systemPrompt := o.standardSystemPrompt(session); systemPrompt != "" {
		messages = append(messages, inference.ChatMessage{Role: "system", Content: systemPrompt})
	}
	for _, m := range history {
		messages = append(messages, inference.ChatMessage{Role: m.Role, Content: m.Content})
	}
	visualCfg := o.visualInputConfig()
	visualFrames := session.LatestVisualFrames(time.Now(), time.Duration(visualCfg.FrameTTLMS)*time.Millisecond)
	if len(visualFrames) > 0 {
		images := make([]inference.ImageFrame, 0, len(visualFrames))
		for _, frame := range visualFrames {
			images = append(images, inference.ImageFrame{
				Data:        frame.Data,
				MimeType:    frame.MimeType,
				Width:       frame.Width,
				Height:      frame.Height,
				Source:      frame.Source,
				TimestampMS: frame.TimestampMS,
				FrameSeq:    frame.FrameSeq,
			})
		}
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				messages[i].Images = images
				break
			}
		}
	}

	// 1. Start LLM stream
	llmCh, llmErrCh := o.inference.GenerateLLMStream(ctx, sessionID, messages, inference.LLMConfig{
		Temperature: 0.7,
		Provider:    components.LLM,
	})

	// 2. Collect the full LLM response, then feed it to TTS once.
	textCh := make(chan string, 1)
	fullResponseCh = make(chan string, 1) // captures full LLM response for history
	go func() {
		defer close(textCh)
		var tokenBuffer strings.Builder
		var fullResponse string

		finalText := func() string {
			if text := strings.TrimSpace(fullResponse); text != "" {
				return text
			}
			return strings.TrimSpace(tokenBuffer.String())
		}

		finish := func(sendTTS bool) {
			text := finalText()
			if sendTTS && text != "" {
				select {
				case textCh <- text:
				case <-ctx.Done():
				}
			}
			if text != "" {
				fullResponseCh <- text
			}
			close(fullResponseCh)
		}

		for {
			select {
			case <-ctx.Done():
				finish(false)
				return
			case chunk, ok := <-llmCh:
				if !ok {
					finish(true)
					return
				}

				// Broadcast LLM token to WebSocket
				o.broadcastJSON(sessionID, map[string]any{
					"type":        "llm_token",
					"token":       chunk.Token,
					"accumulated": chunk.AccumulatedText,
					"is_final":    chunk.IsFinal,
					"turn_seq":    turnSeq,
				})

				tokenBuffer.WriteString(chunk.Token)
				// Track the latest accumulated text. A normally closed errCh
				// can race with the final chunk, so keep this current even
				// before the explicit final marker arrives.
				if chunk.AccumulatedText != "" {
					fullResponse = chunk.AccumulatedText
				}
			case err, ok := <-llmErrCh:
				if !ok {
					llmErrCh = nil
					continue
				}
				if err != nil {
					log.Printf("LLM stream error for session %s: %v", sessionID, err)
					o.broadcastError(sessionID, "LLM generation failed")
				}
				if err != nil {
					finish(true)
					return
				}
			}
		}
	}()

	// 3. Start TTS stream
	ttsAudioCh, ttsErrCh := o.inference.SynthesizeSpeechStream(ctx, textCh, inference.TTSConfig{
		Provider:      components.TTS,
		Voice:         voice,
		SpeakingStyle: speakingStyle,
		Language:      language,
		SessionID:     sessionID,
	})

	// 4. Start Avatar stream
	stdSyncBuf := newVoiceAVSyncBuffer(voiceMaxPCMBufferSamples)
	avatarAudioCh := make(chan *pb.AudioChunk, 8)
	go func() {
		defer close(avatarAudioCh)
		for {
			select {
			case <-ctx.Done():
				return
			case chunk, ok := <-ttsAudioCh:
				if !ok {
					return
				}
				// Avatar decodes AudioChunk.Format itself; browser audio and
				// recordings need signed 16-bit PCM.
				pcm, pcmSampleRate := audioChunkToPCM16(chunk)
				if len(pcm) > 0 {
					appendRecAudio(pcm, pcmSampleRate)
					if dropped := stdSyncBuf.appendPCM(pcm, pcmSampleRate); dropped > 0 {
						bufferedSamples, _, _, _ := stdSyncBuf.snapshot()
						log.Printf("std sync buffer overflow for session %s: dropped=%d bytes, buffered_samples=%d", sessionID, dropped, bufferedSamples)
					}
				}
				// Forward original audio to avatar. Browser audio is published
				// later as part of the same paced AV segment as the video.
				select {
				case avatarAudioCh <- chunk:
				case <-ctx.Done():
					return
				}
			case err, ok := <-ttsErrCh:
				if !ok {
					ttsErrCh = nil
					continue
				}
				if err == nil {
					continue
				}
				log.Printf("TTS stream error for session %s: %v", sessionID, err)
				o.broadcastError(sessionID, "Speech synthesis failed")
				return
			}
		}
	}()

	// Delay speaking status until first video frame arrives (avoids frozen-frame stall on frontend).
	speakingBroadcasted := false

	// Serialize with concurrent avatar operations (see runVoiceLLMPipeline comment).
	o.avatarMu.Lock()
	videoCh, videoErrCh := o.inference.GenerateAvatarStream(ctx, avatarAudioCh)

	// 5. Publish paced AV segments. The standard/Qwen chain receives audio
	// before video, so PCM is buffered and sliced to match each video segment.
	var (
		segVideo       []byte
		segFrames      int
		segWidth       int
		segHeight      int
		segFPS         int
		segCount       int
		segSeq         int64
		firstFrameSent bool
	)
	lookupStdPeer := func() mediapeer.MediaPeer {
		o.mu.RLock()
		defer o.mu.RUnlock()
		return o.peers[sessionID]
	}
	flushStdSeg := func(isFinalSeg bool) {
		if segCount == 0 {
			return
		}
		segSeq++
		if peer := lookupStdPeer(); peer != nil {
			segPCM, outSamples, wantSamples, bufferedSamplesAfterTake := stdSyncBuf.takeSegmentPCM(segFrames, segFPS, isFinalSeg)
			_, _, _, sampleRate := stdSyncBuf.snapshot()
			if sampleRate <= 0 {
				sampleRate = 16000
			}
			if segSeq == 1 && outSamples < wantSamples {
				bufferedSamples, _, _, _ := stdSyncBuf.snapshot()
				log.Printf("std av drift for session %s: out_samples=%d want_samples=%d frames=%d fps=%d buffered_samples=%d",
					sessionID, outSamples, wantSamples, segFrames, segFPS, bufferedSamples)
			}
			if isFinalSeg && bufferedSamplesAfterTake > 0 {
				log.Printf("std av strict trim tail session=%s: trimmed_samples=%d", sessionID, bufferedSamplesAfterTake)
			}
			raw := &mediapeer.RawAVSegment{
				TraceLabel: voiceTraceLabel(sessionID, turnSeq, "standard", "", segSeq),
				Epoch:      turnSeq,
				RGB:        segVideo,
				PCM:        segPCM,
				SampleRate: sampleRate,
				Width:      segWidth,
				Height:     segHeight,
				FPS:        segFPS,
				NumFrames:  segFrames,
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
			flushStdSeg(false)
			finishRec()
			o.avatarMu.Unlock()
			return
		case chunk, ok := <-videoCh:
			if !ok {
				flushStdSeg(true)
				finishRec()
				if err := <-videoErrCh; err != nil {
					log.Printf("Avatar stream error for session %s: %v", sessionID, err)
					o.broadcastError(sessionID, "Avatar generation failed")
				}
				if ctx.Err() == nil {
					if peer := lookupStdPeer(); peer != nil {
						peer.WaitAVDrain(10 * time.Second)
					}
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
			beginRec(int(chunk.GetWidth()), int(chunk.GetHeight()), fps)
			writeRecVideo(chunk.GetData())
			segVideo = append(segVideo, chunk.GetData()...)
			segFrames += nf
			segWidth = int(chunk.GetWidth())
			segHeight = int(chunk.GetHeight())
			segFPS = fps
			segCount++
			if segCount >= stdChunksPerSegment || chunk.GetIsFinal() {
				flushStdSeg(chunk.GetIsFinal())
			}
		}
	}
}

// HandleAudioStream processes incoming user audio through the session's pipeline.
func (o *Orchestrator) HandleAudioStream(ctx context.Context, sessionID string, audioCh <-chan []byte) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}

	if session.Mode == ModeStandard {
		go o.runStandardASRLoop(ctx, session, sessionID, audioCh)
		return nil
	}

	pipeCtx, cancel := context.WithCancel(ctx)
	session.mu.Lock()
	session.PipelineCancel = cancel
	session.mu.Unlock()

	pipelineSeq := session.MarkPipelineRunning()
	go o.runVoiceLLMPipeline(pipeCtx, session, sessionID, wrapVoiceAudioInput(pipeCtx, audioCh), pipelineSeq, 0)
	return nil
}

func (o *Orchestrator) runStandardASRLoop(ctx context.Context, session *Session, sessionID string, audioCh <-chan []byte) {
	components, _, _, language := o.standardCharacterConfig(session)
	transcriptCh, errCh := o.inference.TranscribeStream(ctx, audioCh, inference.ASRConfig{
		Provider:  components.ASR,
		Language:  language,
		SessionID: sessionID,
	})

	session.SetState(StateListening)
	o.broadcastStatus(sessionID, "idle")

	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-transcriptCh:
			if !ok {
				select {
				case err, ok := <-errCh:
					if ok && err != nil && ctx.Err() == nil {
						log.Printf("ASR stream error for session %s: %v", sessionID, err)
						o.broadcastError(sessionID, "Speech recognition failed")
					}
				default:
				}
				return
			}
			text := strings.TrimSpace(event.GetText())
			if text == "" {
				continue
			}

			if !event.GetIsFinal() {
				o.broadcastJSON(sessionID, map[string]any{
					"type":     "transcript",
					"text":     text,
					"is_final": false,
					"speaker":  "user",
				})
				continue
			}

			o.stopPipelineAndWait(session, sessionID, false)
			turnSeq := session.MarkTurnStarted()
			o.advancePlaybackEpoch(sessionID, turnSeq)
			session.AddMessage(ChatMessage{Role: "user", Content: text, TurnSeq: turnSeq})
			o.broadcastJSON(sessionID, map[string]any{
				"type":     "transcript",
				"text":     text,
				"is_final": true,
				"speaker":  "user",
				"turn_seq": turnSeq,
			})

			pipeCtx, cancel := context.WithCancel(context.Background())
			session.mu.Lock()
			session.PipelineCancel = cancel
			session.mu.Unlock()
			pipelineSeq := session.MarkPipelineRunning()
			go o.runStandardPipeline(pipeCtx, session, sessionID, pipelineSeq, turnSeq)
		case err, ok := <-errCh:
			if !ok {
				errCh = nil
				continue
			}
			if err == nil {
				continue
			}
			if ctx.Err() == nil {
				log.Printf("ASR stream error for session %s: %v", sessionID, err)
				o.broadcastError(sessionID, "Speech recognition failed")
			}
			return
		}
	}
}

// runVoiceLLMPipeline executes a VoiceLLM turn source -> VoiceLLM -> Avatar (video).
func (o *Orchestrator) runVoiceLLMPipeline(ctx context.Context, session *Session, sessionID string, inputCh <-chan inference.VoiceLLMInputEvent, pipelineSeq uint64, initialTurnSeq uint64) {
	sessionDir := ""
	if o.recorder != nil {
		sessionDir = o.sessionRecordingDir(session)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	voiceConfig := o.buildVoiceLLMSessionConfig(session, sessionID)
	outputCh, errCh := o.inference.ConverseStream(ctx, inputCh, voiceConfig)

	pendingTurnSeq := initialTurnSeq
	pendingTurnAssistantReady := initialTurnSeq > 0
	ignoredTurns := make(map[string]*voicePipelineTurn)

	var currentTurn *voicePipelineTurn
	var currentTurnDone <-chan voicePipelineTurnResult
	var streamErr error
	var pendingQuestionID string
	var pendingReplyID string
	var pendingUserFinalAt time.Time
	var pendingAssistantMustMatchKey bool

	lookupPeer := func() mediapeer.MediaPeer {
		o.mu.RLock()
		defer o.mu.RUnlock()
		return o.peers[sessionID]
	}

	reservePendingTurn := func() uint64 {
		if pendingTurnSeq == 0 {
			pendingTurnSeq = session.MarkTurnStarted()
			o.advancePlaybackEpoch(sessionID, pendingTurnSeq)
		}
		return pendingTurnSeq
	}

	broadcastProcessing := func(turnSeq uint64) {
		if turnSeq == 0 || !session.IsCurrentPipeline(pipelineSeq) {
			return
		}
		session.SetState(StateProcessing)
		o.broadcastStatusTurn(sessionID, "processing", turnSeq)
	}

	abortTurn := func(turn *voicePipelineTurn, ignoreKey bool) {
		if turn == nil {
			return
		}
		turn.aborted = true
		if ignoreKey && turn.key != "" {
			ignoredTurns[turn.key] = turn
		}
		if turn.avatarCancel != nil {
			turn.avatarCancel()
		}
	}

	saveAssistantMessage := func(turn *voicePipelineTurn) {
		if turn == nil || turn.historySaved || strings.TrimSpace(turn.assistantText) == "" {
			return
		}
		session.AddMessage(ChatMessage{Role: "assistant", Content: turn.assistantText, TurnSeq: turn.seq})
		turn.historySaved = true
	}

	saveTurnTranscript := func(turn *voicePipelineTurn) {
		if turn == nil || turn.transcriptSaved || strings.TrimSpace(turn.assistantText) == "" {
			return
		}
		if o.recorder == nil || turn.recTurnID == "" {
			return
		}
		if err := o.recorder.SaveTranscript(turn.sessionDir, turn.recTurnID, turn.assistantText); err != nil {
			log.Printf("recording: SaveTranscript error session=%s turn=%s: %v", sessionID, turn.recTurnID, err)
			return
		}
		turn.transcriptSaved = true
	}

	saveTurnRawAudio := func(turn *voicePipelineTurn) {
		if turn == nil || turn.rawAudioSaved || len(turn.recAudioBuf) == 0 {
			return
		}
		if o.recorder == nil || turn.recTurnID == "" {
			return
		}
		if err := o.recorder.SaveRawAudio(turn.sessionDir, turn.recTurnID, turn.recAudioBuf, turn.recAudioSR); err != nil {
			log.Printf("recording: SaveRawAudio error session=%s turn=%s: %v", sessionID, turn.recTurnID, err)
			return
		}
		turn.rawAudioSaved = true
	}

	saveTurnConversation := func(turn *voicePipelineTurn) {
		if turn == nil || turn.conversationSaved || strings.TrimSpace(turn.assistantText) == "" {
			return
		}
		saved, err := o.persistSessionConversation(session)
		if err != nil {
			log.Printf("conversation: SaveConversation error session=%s turn=%d: %v", sessionID, turn.seq, err)
			return
		}
		if saved {
			turn.conversationSaved = true
		}
	}

	saveCompletedTurn := func(turn *voicePipelineTurn) {
		if turn == nil || turn.aborted {
			return
		}
		saveAssistantMessage(turn)
		saveTurnRawAudio(turn)
		saveTurnTranscript(turn)
		saveTurnConversation(turn)
	}

	recordIgnoredTurnOutput := func(turn *voicePipelineTurn, output *pb.VoiceLLMOutput) bool {
		if turn == nil || output == nil {
			return false
		}
		isFinal := voiceOutputIsFinal(output)
		if transcript := output.GetTranscript(); transcript != "" {
			if isFinal {
				turn.assistantText = transcript
			} else {
				turn.assistantText += transcript
			}
		}
		if audio := output.GetAudio(); audio != nil && len(audio.GetData()) > 0 {
			turn.recAudioBuf = append(turn.recAudioBuf, audio.GetData()...)
			if int(audio.GetSampleRate()) > 0 {
				turn.recAudioSR = int(audio.GetSampleRate())
			}
		}
		if isFinal {
			shouldBroadcast := !turn.historySaved && strings.TrimSpace(turn.assistantText) != ""
			saveAssistantMessage(turn)
			saveTurnRawAudio(turn)
			saveTurnTranscript(turn)
			saveTurnConversation(turn)
			if shouldBroadcast {
				o.broadcastJSON(sessionID, map[string]any{
					"type":     "transcript",
					"text":     turn.assistantText,
					"is_final": true,
					"speaker":  "assistant",
					"turn_seq": turn.seq,
				})
			}
		}
		return isFinal
	}

	setIdleIfCurrent := func(turnSeq uint64) {
		if turnSeq == 0 || !session.IsCurrentPipeline(pipelineSeq) || !session.IsCurrentTurn(turnSeq) {
			return
		}
		session.SetState(StateListening)
		o.broadcastStatusTurn(sessionID, "idle", turnSeq)
	}

	startTurn := func(key string) *voicePipelineTurn {
		seq := pendingTurnSeq
		if seq == 0 {
			seq = session.MarkTurnStarted()
			o.advancePlaybackEpoch(sessionID, seq)
		}
		if key != "" {
			delete(ignoredTurns, key)
		}
		turn := &voicePipelineTurn{
			seq:         seq,
			key:         key,
			questionID:  pendingQuestionID,
			replyID:     pendingReplyID,
			recTurnID:   fmt.Sprintf("turn%d", seq),
			recAudioSR:  16000,
			sessionDir:  sessionDir,
			turnStart:   time.Now(),
			userFinalAt: pendingUserFinalAt,
			syncBuf:     newVoiceAVSyncBuffer(voiceMaxPCMBufferSamples),
		}
		pendingTurnSeq = 0
		pendingTurnAssistantReady = false
		pendingQuestionID = ""
		pendingReplyID = ""
		pendingUserFinalAt = time.Time{}
		pendingAssistantMustMatchKey = false
		logVoiceTrace("go_turn_started", sessionID, turn.seq, turn.replyID, turn.questionID, turn.userFinalAt)
		broadcastProcessing(seq)
		return turn
	}

	pendingTurnKey := func() string {
		if pendingReplyID != "" {
			return "reply:" + pendingReplyID
		}
		if pendingQuestionID != "" {
			return "question:" + pendingQuestionID
		}
		return ""
	}

	startAvatarWorker := func(turn *voicePipelineTurn) {
		if turn == nil || turn.aborted || turn.avatarStarted {
			return
		}
		turn.avatarStarted = true
		turn.doneCh = make(chan voicePipelineTurnResult, 1)
		turn.avatarAudioCh = make(chan *pb.AudioChunk, 64)
		turn.avatarCtx, turn.avatarCancel = context.WithCancel(ctx)
		turn.avatarWorkerAt = time.Now()
		logVoiceTrace("go_avatar_worker_started", sessionID, turn.seq, turn.replyID, turn.questionID, turn.userFinalAt)
		currentTurnDone = turn.doneCh

		go func(turn *voicePipelineTurn) {
			result := voicePipelineTurnResult{turn: turn}
			defer func() {
				turn.doneCh <- result
			}()

			o.avatarMu.Lock()
			defer o.avatarMu.Unlock()

			if turn.avatarCtx.Err() != nil {
				return
			}

			avatarCtx := inference.WithTraceContext(turn.avatarCtx, inference.TraceContext{
				SessionID:   sessionID,
				QuestionID:  turn.questionID,
				ReplyID:     turn.replyID,
				TurnSeq:     turn.seq,
				UserFinalAt: turn.userFinalAt,
			})
			videoCh, avatarErrCh := o.inference.GenerateAvatarStream(avatarCtx, turn.avatarAudioCh)
			logVoiceTrace("go_avatar_grpc_stream_opened", sessionID, turn.seq, turn.replyID, turn.questionID, turn.userFinalAt)

			var turnRec *recording.TurnRecording
			var (
				segVideo            []byte
				segFrames           int
				segWidth            int
				segHeight           int
				segFPS              int
				segCount            int
				segSeq              int64
				lastKnownSampleRate int
				firstFrameSent      bool
				speakingBroadcasted bool
			)

			flushVoiceSeg := func(isFinalSeg bool) {
				if segCount == 0 {
					return
				}
				segSeq++
				peer := lookupPeer()
				if peer != nil {
					traceLabel := ""
					if segSeq == 1 {
						traceLabel = voiceTraceLabel(sessionID, turn.seq, turn.replyID, turn.questionID, segSeq)
					}
					segPCM, outSamples, wantSamples, bufferedSamplesAfterTake := turn.syncBuf.takeSegmentPCM(segFrames, segFPS, isFinalSeg)
					bufferedSamples, _, _, sampleRate := turn.syncBuf.snapshot()
					if sampleRate > 0 {
						lastKnownSampleRate = sampleRate
					} else {
						sampleRate = lastKnownSampleRate
					}
					if segSeq == 1 && outSamples < wantSamples {
						if sampleRate <= 0 {
							sampleRate = 16000
						}
						log.Printf("voice av drift for session %s: out_samples=%d want_samples=%d frames=%d fps=%d buffered_samples=%d",
							sessionID, outSamples, wantSamples, segFrames, segFPS, bufferedSamples)
					}
					if segSeq == 1 && isFinalSeg && bufferedSamplesAfterTake > 0 {
						log.Printf("voice av strict trim tail session=%s: trimmed_samples=%d", sessionID, bufferedSamplesAfterTake)
					}
					raw := &mediapeer.RawAVSegment{
						TraceLabel:  traceLabel,
						Epoch:       turn.seq,
						RGB:         segVideo,
						PCM:         segPCM,
						UserFinalAt: turn.userFinalAt,
						SampleRate:  sampleRate,
						Width:       segWidth,
						Height:      segHeight,
						FPS:         segFPS,
						NumFrames:   segFrames,
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

			for {
				select {
				case <-turn.avatarCtx.Done():
					flushVoiceSeg(false)
					if turnRec != nil {
						_ = turnRec.Finish()
					}
					return
				case chunk, ok := <-videoCh:
					if !ok {
						flushVoiceSeg(false)
						if turnRec != nil {
							_ = turnRec.Finish()
							turnRec = nil
						}
						if remain, totalIn, totalOut, _ := turn.syncBuf.snapshot(); remain > 0 {
							log.Printf("voice sync tail flush for session %s: dropping_unaligned_samples=%d total_in=%d total_out=%d", sessionID, remain, totalIn, totalOut)
						}
						if err := <-avatarErrCh; err != nil && turn.avatarCtx.Err() == nil && !errors.Is(err, context.Canceled) {
							result.err = err
						}
						if turn.avatarCtx.Err() == nil {
							if peer := lookupPeer(); peer != nil {
								peer.WaitAVDrain(10 * time.Second)
							}
						}
						return
					}

					nf := int(chunk.GetNumFrames())
					if nf <= 0 && int(chunk.GetWidth())*int(chunk.GetHeight())*3 > 0 {
						nf = len(chunk.GetData()) / (int(chunk.GetWidth()) * int(chunk.GetHeight()) * 3)
					}
					fps := int(chunk.GetFps())
					if fps <= 0 {
						fps = 20
					}
					if !firstFrameSent {
						firstFrameSent = true
						turn.firstVideoAt = time.Now()
						logVoiceTrace(
							"go_avatar_first_video_received",
							sessionID,
							turn.seq,
							turn.replyID,
							turn.questionID,
							turn.userFinalAt,
							"avatar_ms="+strconv.FormatInt(time.Since(turn.avatarWorkerAt).Milliseconds(), 10),
						)
						log.Printf("TTFF voice pipeline session=%s turn=%d first_video_chunk=%.3fs", sessionID, turn.seq, time.Since(turn.turnStart).Seconds())
						if !speakingBroadcasted && session.IsCurrentPipeline(pipelineSeq) && session.IsCurrentTurn(turn.seq) {
							speakingBroadcasted = true
							session.SetState(StateSpeaking)
							o.broadcastStatusTurn(sessionID, "speaking", turn.seq)
						}
					}
					if turnRec == nil && o.recorder != nil && turn.recTurnID != "" && nf > 0 {
						turnRec = o.recorder.BeginTurn(turn.sessionDir, turn.recTurnID, int(chunk.GetWidth()), int(chunk.GetHeight()), fps)
					}

					segVideo = append(segVideo, chunk.GetData()...)
					segFrames += nf
					segWidth = int(chunk.GetWidth())
					segHeight = int(chunk.GetHeight())
					segFPS = fps
					segCount++
					flushVoiceSeg(chunk.GetIsFinal())
					if chunk.GetIsFinal() && turnRec != nil {
						_ = turnRec.Finish()
						turnRec = nil
					}
				}
			}
		}(turn)
	}

	closeTurnInput := func(turn *voicePipelineTurn) {
		if turn == nil || !turn.avatarStarted || turn.avatarInputClosed {
			return
		}
		sampleRate := turn.recAudioSR
		if sampleRate <= 0 {
			if _, _, _, bufferedSR := turn.syncBuf.snapshot(); bufferedSR > 0 {
				sampleRate = bufferedSR
			}
		}
		silence := buildTrailingSilence(sampleRate)
		if dropped := turn.syncBuf.appendPCM(silence.GetData(), int(silence.GetSampleRate())); dropped > 0 {
			bufferedSamples, _, _, _ := turn.syncBuf.snapshot()
			log.Printf("voice sync buffer overflow for session %s: dropped=%d bytes, buffered_samples=%d", sessionID, dropped, bufferedSamples)
		}
		select {
		case turn.avatarAudioCh <- silence:
		case <-turn.avatarCtx.Done():
		}
		close(turn.avatarAudioCh)
		turn.avatarInputClosed = true
		turn.avatarInputClosedAt = time.Now()
		if turn.firstVideoAt.IsZero() {
			logVoiceTrace("go_avatar_input_closed", sessionID, turn.seq, turn.replyID, turn.questionID, turn.userFinalAt)
		}
	}

	currentStatusTurnSeq := func() uint64 {
		if currentTurn != nil {
			return currentTurn.seq
		}
		if pendingTurnSeq != 0 {
			return pendingTurnSeq
		}
		return session.CurrentTurnSeq()
	}

	defer func() {
		if currentTurn != nil {
			abortTurn(currentTurn, false)
		}
		session.MarkPipelineFinished(pipelineSeq)
		if session.IsCurrentPipeline(pipelineSeq) {
			session.SetState(StateListening)
			o.broadcastStatusTurn(sessionID, "idle", currentStatusTurnSeq())
		}
	}()

	if initialTurnSeq == 0 && session.IsCurrentPipeline(pipelineSeq) {
		session.SetState(StateListening)
	}

	for outputCh != nil || currentTurn != nil || errCh != nil {
		select {
		case <-ctx.Done():
			return
		case result := <-currentTurnDone:
			if currentTurn == nil || result.turn != currentTurn {
				continue
			}
			turn := currentTurn
			currentTurn = nil
			currentTurnDone = nil
			if result.err != nil && !turn.aborted {
				log.Printf("Avatar stream error for session %s (voice_llm): %v", sessionID, result.err)
				if session.IsCurrentPipeline(pipelineSeq) {
					o.broadcastError(sessionID, "Avatar generation failed")
				}
			}
			if turn.aborted {
				continue
			}
			saveCompletedTurn(turn)
			setIdleIfCurrent(turn.seq)
		case output, ok := <-outputCh:
			if !ok {
				outputCh = nil
				continue
			}
			outputQuestionID := output.GetQuestionId()
			outputReplyID := output.GetReplyId()

			if output.GetBargeIn() {
				if currentTurn != nil || pendingTurnSeq == 0 {
					seq := reservePendingTurn()
					if outputQuestionID != "" {
						pendingQuestionID = outputQuestionID
					}
					if outputReplyID != "" {
						pendingReplyID = outputReplyID
					}
					pendingTurnAssistantReady = false
					logVoiceTrace("go_barge_in_received", sessionID, seq, pendingReplyID, pendingQuestionID, time.Time{})
					if currentTurn != nil {
						abortTurn(currentTurn, true)
						currentTurn = nil
						currentTurnDone = nil
						pendingAssistantMustMatchKey = true
						broadcastProcessing(seq)
					}
				}
				continue
			}

			if currentTurn != nil {
				if outputQuestionID != "" {
					currentTurn.questionID = outputQuestionID
				}
				if outputReplyID != "" {
					currentTurn.replyID = outputReplyID
				}
			} else {
				if outputQuestionID != "" {
					pendingQuestionID = outputQuestionID
				}
				if outputReplyID != "" {
					pendingReplyID = outputReplyID
				}
			}

			if userText := strings.TrimSpace(output.GetUserTranscript()); userText != "" {
				if currentTurn != nil {
					abortTurn(currentTurn, true)
					currentTurn = nil
					currentTurnDone = nil
					pendingAssistantMustMatchKey = true
				}
				if outputQuestionID != "" {
					pendingQuestionID = outputQuestionID
				}
				if outputReplyID != "" {
					pendingReplyID = outputReplyID
				}
				seq := reservePendingTurn()
				pendingTurnAssistantReady = true
				pendingUserFinalAt = time.Now()
				logVoiceTrace(
					"go_user_transcript_received",
					sessionID,
					seq,
					pendingReplyID,
					pendingQuestionID,
					pendingUserFinalAt,
				)
				session.AddMessage(ChatMessage{Role: "user", Content: userText, TurnSeq: seq})
				o.broadcastJSON(sessionID, map[string]any{
					"type":     "transcript",
					"text":     userText,
					"is_final": true,
					"speaker":  "user",
					"turn_seq": seq,
				})
				broadcastProcessing(seq)
			}

			turnKey := voiceOutputTurnKey(output)
			if turnKey != "" {
				if ignoredTurn, ignored := ignoredTurns[turnKey]; ignored {
					if recordIgnoredTurnOutput(ignoredTurn, output) {
						delete(ignoredTurns, turnKey)
					}
					continue
				}
			}

			if pendingTurnSeq != 0 && !pendingTurnAssistantReady && currentTurn == nil && voiceOutputHasAssistantContent(output) {
				continue
			}
			if pendingTurnSeq != 0 && pendingTurnAssistantReady && pendingAssistantMustMatchKey && currentTurn == nil && voiceOutputHasAssistantContent(output) {
				if expectedKey := pendingTurnKey(); expectedKey != "" && turnKey != expectedKey {
					continue
				}
			}

			if !voiceOutputHasAssistantContent(output) && !voiceOutputIsFinal(output) {
				continue
			}
			if currentTurn == nil && !voiceOutputHasAssistantContent(output) {
				continue
			}

			if currentTurn != nil {
				if currentTurn.key == "" && turnKey != "" {
					currentTurn.key = turnKey
				} else if turnKey != "" && currentTurn.key != "" && turnKey != currentTurn.key {
					abortTurn(currentTurn, true)
					currentTurn = nil
					currentTurnDone = nil
				}
			}

			if currentTurn == nil {
				if pendingTurnSeq != 0 && !pendingTurnAssistantReady {
					continue
				}
				currentTurn = startTurn(turnKey)
			}
			if currentTurn.key == "" && turnKey != "" {
				currentTurn.key = turnKey
			}

			isFinal := voiceOutputIsFinal(output)
			if transcript := output.GetTranscript(); transcript != "" {
				o.broadcastJSON(sessionID, map[string]any{
					"type":     "transcript",
					"text":     transcript,
					"is_final": isFinal,
					"speaker":  "assistant",
					"turn_seq": currentTurn.seq,
				})
				if isFinal {
					currentTurn.assistantText = transcript
				} else {
					currentTurn.assistantText += transcript
				}
			}

			audio := output.GetAudio()
			if audio != nil && len(audio.GetData()) > 0 {
				if currentTurn.firstAudioAt.IsZero() {
					currentTurn.firstAudioAt = time.Now()
					logVoiceTrace(
						"go_first_voice_audio_received",
						sessionID,
						currentTurn.seq,
						currentTurn.replyID,
						currentTurn.questionID,
						currentTurn.userFinalAt,
					)
				}
				if !currentTurn.avatarStarted {
					startAvatarWorker(currentTurn)
				}
				if dropped := currentTurn.syncBuf.appendPCM(audio.GetData(), int(audio.GetSampleRate())); dropped > 0 {
					bufferedSamples, _, _, _ := currentTurn.syncBuf.snapshot()
					log.Printf("voice sync buffer overflow for session %s: dropped=%d bytes, buffered_samples=%d", sessionID, dropped, bufferedSamples)
				}
				currentTurn.recAudioBuf = append(currentTurn.recAudioBuf, audio.GetData()...)
				if int(audio.GetSampleRate()) > 0 {
					currentTurn.recAudioSR = int(audio.GetSampleRate())
				}
				audioClone := proto.Clone(audio).(*pb.AudioChunk)
				if currentTurn.firstAvatarAudioAt.IsZero() {
					currentTurn.firstAvatarAudioAt = time.Now()
					logVoiceTrace(
						"go_avatar_first_audio_enqueued",
						sessionID,
						currentTurn.seq,
						currentTurn.replyID,
						currentTurn.questionID,
						currentTurn.userFinalAt,
					)
				}
				select {
				case currentTurn.avatarAudioCh <- audioClone:
				case <-currentTurn.avatarCtx.Done():
				}
			}

			if !isFinal {
				continue
			}
			if currentTurn.audioFinalAt.IsZero() {
				currentTurn.audioFinalAt = time.Now()
				if currentTurn.firstVideoAt.IsZero() {
					logVoiceTrace("go_voice_audio_final_received", sessionID, currentTurn.seq, currentTurn.replyID, currentTurn.questionID, currentTurn.userFinalAt)
				}
			}
			saveAssistantMessage(currentTurn)
			saveTurnRawAudio(currentTurn)
			saveTurnTranscript(currentTurn)
			saveTurnConversation(currentTurn)

			if currentTurn.avatarStarted {
				closeTurnInput(currentTurn)
				continue
			}

			turn := currentTurn
			currentTurn = nil
			currentTurnDone = nil
			saveCompletedTurn(turn)
			setIdleIfCurrent(turn.seq)
		case err, ok := <-errCh:
			if !ok {
				errCh = nil
				continue
			}
			if err != nil && !errors.Is(err, context.Canceled) {
				streamErr = err
			}
			errCh = nil
		}
	}

	if streamErr != nil {
		log.Printf("VoiceLLM stream error for session %s: %v", sessionID, streamErr)
		if session.IsCurrentPipeline(pipelineSeq) {
			o.broadcastError(sessionID, "Voice conversation failed")
		}
	}
}

// Interrupt cancels the current pipeline for a session.
func (o *Orchestrator) Interrupt(sessionID string) error {
	session, err := o.sessionMgr.Get(sessionID)
	if err != nil {
		return err
	}

	turnSeq := session.MarkTurnStarted()
	o.advancePlaybackEpoch(sessionID, turnSeq)
	o.cancelPipeline(session)

	// Also interrupt VoiceLLM on the inference side
	if session.Mode == ModeVoiceLLM {
		_ = o.inference.Interrupt(context.Background(), sessionID)
	}

	session.SetState(StateListening)
	o.broadcastStatusTurn(sessionID, "idle", turnSeq)
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

func (o *Orchestrator) advancePlaybackEpoch(sessionID string, turnSeq uint64) {
	if turnSeq == 0 {
		return
	}
	o.mu.RLock()
	peer := o.peers[sessionID]
	o.mu.RUnlock()
	if peer != nil {
		peer.AdvancePlaybackEpoch(turnSeq)
	}
}

// broadcastStatus sends an avatar_status message to all WebSocket clients.
func (o *Orchestrator) broadcastStatus(sessionID, status string) {
	o.broadcastJSON(sessionID, map[string]string{
		"type":   "avatar_status",
		"status": status,
	})
}

func (o *Orchestrator) broadcastStatusTurn(sessionID, status string, turnSeq uint64) {
	payload := map[string]any{
		"type":   "avatar_status",
		"status": status,
	}
	if turnSeq > 0 {
		payload["turn_seq"] = turnSeq
	}
	o.broadcastJSON(sessionID, payload)
}

// broadcastError sends an error message to all WebSocket clients.
func (o *Orchestrator) broadcastError(sessionID, message string) {
	o.broadcastJSON(sessionID, map[string]string{
		"type":    "error",
		"message": message,
	})
}

// PersistSessionConversation writes the current session history to session.json.
func (o *Orchestrator) PersistSessionConversation(session *Session) (bool, error) {
	return o.persistSessionConversation(session)
}

func (o *Orchestrator) persistSessionConversation(session *Session) (bool, error) {
	if o == nil || o.charStore == nil || session == nil {
		return false, nil
	}

	sessionID, characterID, startedAt, endedAt, history := session.ConversationSnapshot()
	if characterID == "" || len(history) == 0 {
		return false, nil
	}

	messages := make([]map[string]any, len(history))
	for i, m := range history {
		ts := m.Timestamp
		if ts.IsZero() {
			ts = startedAt
		}
		messages[i] = map[string]any{
			"role":      m.Role,
			"content":   m.Content,
			"timestamp": ts.UTC().Format(time.RFC3339Nano),
		}
		if m.TurnSeq > 0 {
			messages[i]["turn_seq"] = m.TurnSeq
		}
	}

	if err := o.charStore.SaveConversation(characterID, sessionID, startedAt, endedAt, messages); err != nil {
		return false, err
	}
	return true, nil
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
	if o.wsHub == nil {
		return
	}
	data, err := json.Marshal(v)
	if err != nil {
		log.Printf("Failed to marshal broadcast: %v", err)
		return
	}
	o.wsHub.Broadcast(sessionID, data)
}
