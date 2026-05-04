package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	LiveKit   LiveKitConfig   `yaml:"livekit"`
	Session   SessionConfig   `yaml:"session"`
	Pipeline  PipelineConfig  `yaml:"pipeline"`
	Inference InferenceConfig `yaml:"inference_grpc"`
	Plugins   PluginDefaults  `yaml:"inference"`
	Recording RecordingConfig `yaml:"recording"`
}

type RecordingConfig struct {
	Enabled   bool   `yaml:"enabled"`
	OutputDir string `yaml:"output_dir"`
	CRF       int    `yaml:"crf"`
}

type InferenceConfig struct {
	Addr string `yaml:"addr"`
}

type PluginDefaults struct {
	LLM PluginDefaultSection `yaml:"llm"`
	ASR PluginDefaultSection `yaml:"asr"`
	TTS PluginDefaultSection `yaml:"tts"`
}

type PluginDefaultSection struct {
	Default string `yaml:"default"`
}

type ServerConfig struct {
	Host        string   `yaml:"host"`
	HTTPPort    int      `yaml:"http_port"`
	GRPCPort    int      `yaml:"grpc_port"`
	CORSOrigins []string `yaml:"cors_origins"`
}

type LiveKitConfig struct {
	URL       string `yaml:"url"`
	APIKey    string `yaml:"api_key"`
	APISecret string `yaml:"api_secret"`
}

type SessionConfig struct {
	MaxConcurrent int `yaml:"max_concurrent"`
	IdleTimeoutS  int `yaml:"idle_timeout_s"`
	MaxDurationS  int `yaml:"max_duration_s"`
}

type PipelineConfig struct {
	DefaultMode     string            `yaml:"default_mode"`
	StreamingMode   string            `yaml:"streaming_mode"` // "direct" (default, P2P WebRTC) or "livekit"
	VisualInput     VisualInputConfig `yaml:"visual_input,omitempty"`
	ICEServers      []ICEServer       `yaml:"ice_servers,omitempty"`
	ICETCPPort      int               `yaml:"ice_tcp_port,omitempty"`      // Deprecated: use TURN instead
	ICEPublicIP     string            `yaml:"ice_public_ip,omitempty"`     // Public IP or hostname (also used by TURN)
	ICENetworkTypes []string          `yaml:"ice_network_types,omitempty"` // Deprecated: use TURN instead
	TURNEnabled     bool              `yaml:"turn_enabled,omitempty"`      // Enable embedded TURN-over-TCP server
	TURNPort        int               `yaml:"turn_port,omitempty"`         // TCP port for TURN (default 3478)
	TURNRealm       string            `yaml:"turn_realm,omitempty"`        // TURN realm (default "cyberverse")
	TURNUsername    string            `yaml:"turn_username,omitempty"`     // TURN username (default "cyberverse")
	TURNPassword    string            `yaml:"turn_password,omitempty"`     // TURN password (required when enabled)
	DefaultLLM      string            `yaml:"-"`
	DefaultASR      string            `yaml:"-"`
	DefaultTTS      string            `yaml:"-"`
}

type VisualInputConfig struct {
	Enabled           *bool   `yaml:"enabled,omitempty"`
	FrameIntervalMS   int     `yaml:"frame_interval_ms,omitempty"`
	MaxWidth          int     `yaml:"max_width,omitempty"`
	MaxHeight         int     `yaml:"max_height,omitempty"`
	JPEGQuality       float64 `yaml:"jpeg_quality,omitempty"`
	MaxFrameBytes     int     `yaml:"max_frame_bytes,omitempty"`
	WSMaxMessageBytes int64   `yaml:"ws_max_message_bytes,omitempty"`
	MaxRecentFrames   int     `yaml:"max_recent_frames,omitempty"`
	FrameTTLMS        int     `yaml:"frame_ttl_ms,omitempty"`
}

func (c VisualInputConfig) IsEnabled() bool {
	return c.Enabled == nil || *c.Enabled
}

// ICEServer configures STUN/TURN servers for direct WebRTC mode.
type ICEServer struct {
	URLs       []string `yaml:"urls"`
	Username   string   `yaml:"username,omitempty"`
	Credential string   `yaml:"credential,omitempty"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Expand environment variables
	expanded := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, err
	}

	// Apply defaults
	if cfg.Server.Host == "" {
		cfg.Server.Host = "0.0.0.0"
	}
	if cfg.Server.HTTPPort == 0 {
		cfg.Server.HTTPPort = 8080
	}
	if cfg.Server.GRPCPort == 0 {
		cfg.Server.GRPCPort = 50051
	}
	if cfg.Session.MaxConcurrent == 0 {
		cfg.Session.MaxConcurrent = 4
	}
	if cfg.Pipeline.DefaultMode == "" {
		cfg.Pipeline.DefaultMode = "standard"
	}
	if cfg.Pipeline.StreamingMode == "" {
		cfg.Pipeline.StreamingMode = "direct"
	}
	if cfg.Pipeline.VisualInput.FrameIntervalMS == 0 {
		cfg.Pipeline.VisualInput.FrameIntervalMS = 1000
	}
	if cfg.Pipeline.VisualInput.MaxWidth == 0 {
		cfg.Pipeline.VisualInput.MaxWidth = 1280
	}
	if cfg.Pipeline.VisualInput.MaxHeight == 0 {
		cfg.Pipeline.VisualInput.MaxHeight = 720
	}
	if cfg.Pipeline.VisualInput.JPEGQuality == 0 {
		cfg.Pipeline.VisualInput.JPEGQuality = 0.78
	}
	if cfg.Pipeline.VisualInput.MaxFrameBytes == 0 {
		cfg.Pipeline.VisualInput.MaxFrameBytes = 512 * 1024
	}
	if cfg.Pipeline.VisualInput.WSMaxMessageBytes == 0 {
		cfg.Pipeline.VisualInput.WSMaxMessageBytes = 1024 * 1024
	}
	if cfg.Pipeline.VisualInput.MaxRecentFrames == 0 {
		cfg.Pipeline.VisualInput.MaxRecentFrames = 2
	}
	if cfg.Pipeline.VisualInput.FrameTTLMS == 0 {
		cfg.Pipeline.VisualInput.FrameTTLMS = 10000
	}
	if len(cfg.Pipeline.ICEServers) == 0 {
		cfg.Pipeline.ICEServers = []ICEServer{
			{URLs: []string{"stun:stun.l.google.com:19302"}},
		}
	}
	if cfg.Pipeline.TURNPort == 0 {
		cfg.Pipeline.TURNPort = 3478
	}
	if cfg.Pipeline.TURNRealm == "" {
		cfg.Pipeline.TURNRealm = "cyberverse"
	}
	if cfg.Pipeline.TURNUsername == "" {
		cfg.Pipeline.TURNUsername = "cyberverse"
	}
	if cfg.Recording.OutputDir == "" {
		cfg.Recording.OutputDir = "./recordings"
	}
	if cfg.Recording.CRF == 0 {
		cfg.Recording.CRF = 23
	}
	if cfg.Plugins.LLM.Default == "" {
		cfg.Plugins.LLM.Default = "qwen"
	}
	if cfg.Plugins.ASR.Default == "" {
		cfg.Plugins.ASR.Default = "qwen"
	}
	if cfg.Plugins.TTS.Default == "" {
		cfg.Plugins.TTS.Default = "qwen"
	}
	cfg.Pipeline.DefaultLLM = cfg.Plugins.LLM.Default
	cfg.Pipeline.DefaultASR = cfg.Plugins.ASR.Default
	cfg.Pipeline.DefaultTTS = cfg.Plugins.TTS.Default

	// Inference gRPC address: env var takes precedence
	if addr := os.Getenv("GRPC_INFERENCE_ADDR"); addr != "" {
		cfg.Inference.Addr = addr
	}
	if cfg.Inference.Addr == "" {
		cfg.Inference.Addr = fmt.Sprintf("localhost:%d", cfg.Server.GRPCPort)
	}

	return &cfg, nil
}
