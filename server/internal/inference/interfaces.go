package inference

import (
	"context"

	pb "github.com/cyberverse/server/internal/pb"
)

type ImageFrame struct {
	Data        []byte
	MimeType    string
	Width       int32
	Height      int32
	Source      string
	TimestampMS int64
	FrameSeq    int64
}

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string
	Content string
	Images  []ImageFrame
}

// LLMConfig holds parameters for LLM generation.
type LLMConfig struct {
	Model       string
	Temperature float32
	MaxTokens   int32
	Provider    string
}

type TTSConfig struct {
	Provider      string
	Voice         string
	SpeakingStyle string
	Language      string
	SessionID     string
}

type ASRConfig struct {
	Provider  string
	Language  string
	SessionID string
}

// VoiceLLMSessionConfig holds per-session character config for VoiceLLM.
type VoiceLLMDialogContextItem struct {
	Role      string
	Text      string
	Timestamp int64
}

type VoiceLLMSessionConfig struct {
	SessionID      string
	Provider       string
	SystemPrompt   string
	Voice          string // maps to voice_type / speaker
	BotName        string
	SpeakingStyle  string
	WelcomeMessage string
	DialogContext  []VoiceLLMDialogContextItem
}

// VoiceLLMInputEvent is one input item for a VoiceLLM conversation stream.
// Exactly one of Audio or Text should be set.
type VoiceLLMInputEvent struct {
	Audio []byte
	Text  string
}

// InferenceService defines the interface for communicating with the Python
// inference layer. Using an interface allows tests to inject mocks.
type InferenceService interface {
	HealthCheck(ctx context.Context) error
	AvatarInfo(ctx context.Context) (*pb.AvatarInfo, error)

	// Avatar
	SetAvatar(ctx context.Context, sessionID string, imageData []byte, format string) error
	GenerateAvatarStream(ctx context.Context, audioCh <-chan *pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error)
	GenerateAvatar(ctx context.Context, audioChunks []*pb.AudioChunk) (<-chan *pb.VideoChunk, <-chan error)

	// LLM
	GenerateLLMStream(ctx context.Context, sessionID string, messages []ChatMessage, config LLMConfig) (<-chan *pb.LLMChunk, <-chan error)

	// TTS
	SynthesizeSpeechStream(ctx context.Context, textCh <-chan string, config TTSConfig) (<-chan *pb.AudioChunk, <-chan error)

	// ASR
	TranscribeStream(ctx context.Context, audioCh <-chan []byte, config ASRConfig) (<-chan *pb.TranscriptEvent, <-chan error)

	// VoiceLLM
	CheckVoice(ctx context.Context, config VoiceLLMSessionConfig) (string, error)
	ConverseStream(ctx context.Context, inputCh <-chan VoiceLLMInputEvent, config VoiceLLMSessionConfig) (<-chan *pb.VoiceLLMOutput, <-chan error)
	Interrupt(ctx context.Context, sessionID string) error

	Close() error
}
